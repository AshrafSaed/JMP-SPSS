"""ss_spc2.py - Advanced SPC & MSA: EWMA, CUSUM, Laney P'/U', Pareto, run chart, Hotelling T2, Gage Type 1,
linearity & bias, nested (destructive) Gage R&R, variability/multivari chart, acceptance sampling OC curves.
JMP: Control Chart Builder (EWMA/CUSUM/Presummarize/Laney), Pareto Plot, Multivariate Control Chart, MSA (Type 1 Gage, Linearity & Bias, Nested), Variability Chart.
"""
from __future__ import annotations
import itertools, math
import numpy as np, pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from ss_common import HE, Result, Lang, save_fig, PALETTE
from ss_spc import d2, nelson_rules

HE.update({"EWMA chart": "תרשים EWMA", "CUSUM chart": "תרשים CUSUM", "Laney chart": "תרשים Laney", "Pareto chart": "תרשים פארטו", "Run chart": "תרשים ריצה",
           "Hotelling T2 chart": "תרשים Hotelling T²", "Gage Type 1 study": "מחקר Gage סוג 1", "Linearity and bias": "ליניאריות והטיה", "Nested Gage R&R": "Gage R&R מקונן",
           "Variability chart": "תרשים שונות", "Acceptance sampling": "דגימת קבלה", "Subgroup": "תת-קבוצה", "Out of control": "מחוץ לבקרה"})


def _sigma_individuals(x):
    mr = np.abs(np.diff(x)); return float(mr.mean() / 1.128)


def _prep(df, params):
    col = params["column"]; sub = params.get("subgroup"); d = df[[col] + ([sub] if sub else [])].dropna()
    if sub:
        g = d.groupby(sub, sort=False)[col]; xb = g.mean().values; n = int(g.size().mode()[0]); r = g.apply(lambda s: s.max() - s.min()).values
        sw = float(np.mean(r) / d2(n)) if 2 <= n <= 25 else float(np.sqrt(g.var().mean())); sizes = g.size().values
        return xb, sw / math.sqrt(n), sw, list(g.groups.keys()), n
    x = d[col].values.astype(float); return x, _sigma_individuals(x), _sigma_individuals(x), list(range(1, len(x) + 1)), 1


def ewma(df, params, L: Lang, outdir):
    """column; [subgroup]; lambda (0.2); L (3); target (default phase-I mean); sigma (default estimated)."""
    lam = float(params.get("lambda", 0.2)); Lk = float(params.get("L", 3.0)); x, s_mean, sw, labels, n = _prep(df, params)
    mu = float(params.get("target", x.mean())); s = float(params["sigma"]) / math.sqrt(n) if params.get("sigma") else s_mean
    z = np.zeros(len(x)); prev = mu
    for i, v in enumerate(x): prev = lam * v + (1 - lam) * prev; z[i] = prev
    t = np.arange(1, len(x) + 1); w = Lk * s * np.sqrt(lam / (2 - lam) * (1 - (1 - lam) ** (2 * t))); ucl, lcl = mu + w, mu - w; viol = np.where((z > ucl) | (z < lcl))[0]
    res = Result(L.t("EWMA chart"))
    res.table("EWMA parameters", pd.DataFrame([{"Lambda": lam, "L (sigma multiple)": Lk, "Center": mu, "Sigma of plotted statistic (single obs)": s, "Asymptotic UCL": mu + Lk * s * math.sqrt(lam / (2 - lam)), "Asymptotic LCL": mu - Lk * s * math.sqrt(lam / (2 - lam)), "Points out": len(viol)}]))
    if len(viol): res.table("Out-of-control points", pd.DataFrame({"Index": [labels[i] for i in viol], "EWMA": z[viol], "UCL": ucl[viol], "LCL": lcl[viol]}))
    fig, ax = plt.subplots(figsize=(9, 3.8)); ax.plot(t, x, ".", color="#AAAAAA", label="data"); ax.plot(t, z, "o-", ms=3, color=PALETTE[0], label="EWMA"); ax.plot(t, ucl, color=PALETTE[3]); ax.plot(t, lcl, color=PALETTE[3]); ax.axhline(mu, color="green")
    ax.plot(t[viol], z[viol], "o", color="red", ms=6); ax.set_title(L.fig("EWMA chart")); ax.legend(fontsize=7); res.fig("EWMA", save_fig(fig, outdir, "ewma"))
    res.summary.update({"n_out": int(len(viol)), "first_signal": int(viol[0] + 1) if len(viol) else None, "center": mu})
    res.note("Phase I limits use the data themselves; if the series already contains a shift, supply 'target' and 'sigma' from an in-control reference period.")
    return res


def cusum(df, params, L: Lang, outdir):
    """column; [subgroup]; k (0.5 sigma); h (5 sigma); target; sigma."""
    k = float(params.get("k", 0.5)); h = float(params.get("h", 5.0)); x, s_mean, sw, labels, n = _prep(df, params)
    mu = float(params.get("target", x.mean())); s = float(params["sigma"]) / math.sqrt(n) if params.get("sigma") else s_mean
    K, H = k * s, h * s; cp = np.zeros(len(x)); cm = np.zeros(len(x)); a = b = 0.0; np_ = np.zeros(len(x), int); nm_ = np.zeros(len(x), int); ca = cb = 0
    for i, v in enumerate(x):
        a = max(0.0, v - (mu + K) + a); b = max(0.0, (mu - K) - v + b); ca = ca + 1 if a > 0 else 0; cb = cb + 1 if b > 0 else 0; cp[i], cm[i], np_[i], nm_[i] = a, b, ca, cb
    up, dn = np.where(cp > H)[0], np.where(cm > H)[0]
    res = Result(L.t("CUSUM chart"))
    res.table("CUSUM parameters", pd.DataFrame([{"Target": mu, "Sigma (plotted stat)": s, "k": k, "h": h, "K (=k*sigma)": K, "H (=h*sigma)": H, "Upper signals": len(up), "Lower signals": len(dn)}]))
    est = []
    if len(up): i = up[0]; est.append({"Side": "upper", "Signal at": labels[i], "Estimated shift start": labels[i - np_[i] + 1], "Estimated new mean": mu + K + cp[i] / np_[i]})
    if len(dn): i = dn[0]; est.append({"Side": "lower", "Signal at": labels[i], "Estimated shift start": labels[i - nm_[i] + 1], "Estimated new mean": mu - K - cm[i] / nm_[i]})
    if est: res.table("First signal & shift estimate", pd.DataFrame(est))
    t = np.arange(1, len(x) + 1); fig, ax = plt.subplots(figsize=(9, 3.8)); ax.step(t, cp, where="mid", label="C+", color=PALETTE[0]); ax.step(t, -cm, where="mid", label="C-", color=PALETTE[1]); ax.axhline(H, color=PALETTE[3]); ax.axhline(-H, color=PALETTE[3]); ax.axhline(0, color="grey", lw=.6)
    ax.plot(t[up], cp[up], "o", color="red", ms=4); ax.plot(t[dn], -cm[dn], "o", color="red", ms=4); ax.legend(fontsize=7); ax.set_title(L.fig("CUSUM chart")); res.fig("CUSUM", save_fig(fig, outdir, "cusum"))
    res.summary.update({"upper_signals": int(len(up)), "lower_signals": int(len(dn)), "first_upper": int(up[0] + 1) if len(up) else None, "first_lower": int(dn[0] + 1) if len(dn) else None})
    res.note("Defaults k=0.5, h=5 (sigma units) detect ~1 sigma shifts with ARL0 ~ 465 for normal data (Montgomery); choose k = half the shift you want to detect.")
    return res


def laney(df, params, L: Lang, outdir):
    """type p|u; count (defects/defectives column); size (n or area column).  Laney P'/U' for overdispersed attribute data."""
    typ = params.get("type", "p"); d = df[[params["count"], params["size"]]].dropna(); c = d[params["count"]].values.astype(float); n = d[params["size"]].values.astype(float)
    pb = c.sum() / n.sum(); sd_i = np.sqrt(pb * (1 - pb) / n) if typ == "p" else np.sqrt(pb / n); z = (c / n - pb) / sd_i; sig_z = _sigma_individuals(z)
    ucl = pb + 3 * sig_z * sd_i; lcl = np.clip(pb - 3 * sig_z * sd_i, 0, None); ucl0 = pb + 3 * sd_i; lcl0 = np.clip(pb - 3 * sd_i, 0, None); y = c / n; viol = np.where((y > ucl) | (y < lcl))[0]; v0 = np.where((y > ucl0) | (y < lcl0))[0]
    res = Result(L.t("Laney chart"))
    res.table("Laney parameters", pd.DataFrame([{"Type": f"{typ.upper()}'", "Center": pb, "Sigma_z (overdispersion factor)": sig_z, "Points out (Laney)": len(viol), "Points out (classical)": len(v0), "Subgroups": len(c)}]))
    t = np.arange(1, len(y) + 1); fig, ax = plt.subplots(figsize=(9, 3.8)); ax.plot(t, y, "o-", ms=3, color=PALETTE[0]); ax.step(t, ucl, where="mid", color=PALETTE[3], label="Laney limits"); ax.step(t, lcl, where="mid", color=PALETTE[3]); ax.step(t, ucl0, where="mid", color="grey", ls=":", label="classical"); ax.step(t, lcl0, where="mid", color="grey", ls=":")
    ax.axhline(pb, color="green"); ax.plot(t[viol], y[viol], "o", color="red"); ax.legend(fontsize=7); ax.set_title(L.fig("Laney chart")); res.fig("Laney", save_fig(fig, outdir, "laney"))
    res.summary.update({"sigma_z": float(sig_z), "n_out_laney": int(len(viol)), "n_out_classical": int(len(v0))}); res.note("Sigma_z ~ 1 means classical limits are adequate; >1 indicates overdispersion (classical charts give false alarms).")
    return res


def pareto(df, params, L: Lang, outdir):
    """category; [value] (count/cost column; default = row counts); top (n categories, rest -> Other)."""
    cat = params["category"]; val = params.get("value"); s = df.groupby(cat)[val].sum() if val else df[cat].value_counts(); s = s.sort_values(ascending=False); top = params.get("top")
    if top and len(s) > top: s = pd.concat([s.iloc[:top], pd.Series({"Other": s.iloc[top:].sum()})])
    pct = 100 * s / s.sum(); cum = pct.cumsum(); res = Result(L.t("Pareto chart")); res.table("Pareto table", pd.DataFrame({"Category": s.index.astype(str), "Value": s.values, "Percent": pct.values, "Cumulative %": cum.values}))
    fig, ax = plt.subplots(figsize=(8, 4)); ax.bar(range(len(s)), s.values, color=PALETTE[0]); ax.set_xticks(range(len(s))); ax.set_xticklabels(s.index.astype(str), rotation=30, ha="right"); ax2 = ax.twinx(); ax2.plot(range(len(s)), cum.values, "o-", color=PALETTE[3]); ax2.set_ylim(0, 105); ax2.axhline(80, color="grey", ls=":"); ax2.grid(False)
    ax.set_title(L.fig("Pareto chart")); res.fig("Pareto", save_fig(fig, outdir, "pareto")); vital = int((cum.shift(fill_value=0) < 80).sum()); res.summary.update({"top_category": str(s.index[0]), "categories_to_80pct": vital}); return res


def run_chart(df, params, L: Lang, outdir):
    """params: column. Runs about the median and runs up/down tests with a run chart."""
    col = params["column"]; x = df[col].dropna().values.astype(float); n = len(x); med = np.median(x); s = x[x != med] > med; n1, n2 = int(s.sum()), int((~s).sum()); runs = 1 + int(np.sum(s[1:] != s[:-1]))
    mu = 2 * n1 * n2 / (n1 + n2) + 1; var = 2 * n1 * n2 * (2 * n1 * n2 - n1 - n2) / ((n1 + n2) ** 2 * (n1 + n2 - 1)); z = (runs - mu) / math.sqrt(var)
    dsg = np.sign(np.diff(x)); dsg = dsg[dsg != 0]; ru = 1 + int(np.sum(dsg[1:] != dsg[:-1])); mu2 = (2 * (len(dsg) + 1) - 1) / 3; var2 = (16 * (len(dsg) + 1) - 29) / 90; z2 = (ru - mu2) / math.sqrt(var2)
    res = Result(L.t("Run chart"))
    res.table("Runs about the median", pd.DataFrame([{"Median": med, "Runs observed": runs, "Runs expected": mu, "z": z, "p-value": 2 * stats.norm.sf(abs(z)), "Interpretation": "clustering/shift" if z < 0 and 2 * stats.norm.sf(abs(z)) < .05 else ("mixtures/oscillation" if z > 0 and 2 * stats.norm.sf(abs(z)) < .05 else "no evidence")}]))
    res.table("Runs up and down", pd.DataFrame([{"Runs observed": ru, "Runs expected": mu2, "z": z2, "p-value": 2 * stats.norm.sf(abs(z2)), "Interpretation": "trends" if z2 < 0 and 2 * stats.norm.sf(abs(z2)) < .05 else ("oscillation" if z2 > 0 and 2 * stats.norm.sf(abs(z2)) < .05 else "no evidence")}]))
    fig, ax = plt.subplots(figsize=(9, 3.6)); ax.plot(range(1, n + 1), x, "o-", ms=3); ax.axhline(med, color=PALETTE[3]); ax.set_title(L.fig("Run chart")); res.fig("run chart", save_fig(fig, outdir, "run_chart")); res.summary.update({"p_median": float(2 * stats.norm.sf(abs(z))), "p_updown": float(2 * stats.norm.sf(abs(z2)))}); return res


def hotelling_t2(df, params, L: Lang, outdir):
    """columns (p variables, individual observations); phase 1|2; alpha (0.0027); for phase 2 give 'ref_rows' count of in-control reference rows at the start (default: all)."""
    cols = params["columns"]; d = df[cols].dropna(); X = d.values; n, p = X.shape; a = float(params.get("alpha", 0.0027)); phase = int(params.get("phase", 1))
    nref = int(params.get("ref_rows", n)) if phase == 2 else n; ref = X[:nref]; mu = ref.mean(0); S = np.cov(ref.T); Si = np.linalg.inv(S); T2 = np.einsum("ij,jk,ik->i", X - mu, Si, X - mu)
    if phase == 1: ucl = (n - 1) ** 2 / n * stats.beta.ppf(1 - a, p / 2, (n - p - 1) / 2)
    else: m = nref; ucl = p * (m + 1) * (m - 1) / (m * (m - p)) * stats.f.ppf(1 - a, p, m - p)
    viol = np.where(T2 > ucl)[0]; res = Result(L.t("Hotelling T2 chart"))
    res.table("T2 chart", pd.DataFrame([{"Phase": phase, "p": p, "n": n, "alpha": a, "UCL": ucl, "Points out": len(viol)}]))
    if len(viol):
        sdv = np.sqrt(np.diag(S)); zz = (X[viol] - mu) / sdv; rows = []
        for r, z_ in zip(viol, zz): o = np.argsort(-np.abs(z_)); rows.append({"Row": d.index[r], "T2": T2[r], "Largest contributor": cols[o[0]], "z of contributor": z_[o[0]], "Second": cols[o[1]] if p > 1 else "", "z second": z_[o[1]] if p > 1 else np.nan})
        res.table("Out-of-control points (contributors by standardized deviation)", pd.DataFrame(rows))
    fig, ax = plt.subplots(figsize=(9, 3.8)); ax.plot(T2, "o-", ms=3); ax.axhline(ucl, color=PALETTE[3]); ax.plot(viol, T2[viol], "o", color="red"); ax.set_title(L.fig("Hotelling T2 chart")); res.fig("T2", save_fig(fig, outdir, "t2"))
    res.summary.update({"ucl": float(ucl), "n_out": int(len(viol))}); res.note("Contributors are simple standardized deviations (not the full Mason-Young-Tracy decomposition); investigate correlated variables together.")
    return res


def gage_type1(df, params, L: Lang, outdir):
    """column (repeated measurements of one reference part); reference; tolerance (USL-LSL); k_tol (0.2); k_sd (6)."""
    col = params["column"]; x = df[col].dropna().values.astype(float); n = len(x); ref = float(params["reference"]); tol = float(params["tolerance"]); k = float(params.get("k_tol", 0.2)); ks = float(params.get("k_sd", 6)); m, s = x.mean(), x.std(ddof=1)
    cg = k * tol / (ks * s); cgk = (k / 2 * tol - abs(m - ref)) / (ks / 2 * s); t = (m - ref) / (s / math.sqrt(n)); p = 2 * stats.t.sf(abs(t), n - 1)
    res = Result(L.t("Gage Type 1 study"))
    res.table("Type 1 study", pd.DataFrame([{"n": n, "Reference": ref, "Mean": m, "Std Dev": s, "Bias": m - ref, "Bias t": t, "Bias p-value": p, "Cg": cg, "Cgk": cgk, "%Var (Repeatability)": 100 * ks * s / tol, "%Var (Rep.+Bias)": 100 * (ks * s + 2 * abs(m - ref)) / tol}]))
    fig, ax = plt.subplots(figsize=(9, 3.8)); ax.plot(x, "o-", ms=3); ax.axhline(ref, color="green"); ax.axhline(ref + k * tol / 2, color=PALETTE[3], ls="--"); ax.axhline(ref - k * tol / 2, color=PALETTE[3], ls="--"); ax.axhline(m, color="grey", ls=":"); ax.set_title(L.fig("Gage Type 1 study")); res.fig("Type 1", save_fig(fig, outdir, "gage_type1"))
    res.summary.update({"Cg": float(cg), "Cgk": float(cgk), "bias_p": float(p)}); res.note("Cg/Cgk >= 1.33 is the common (VDA 5 / Minitab-style) acceptance convention with K=20%, 6s; it is not a universal standard - confirm customer requirement.")
    return res


def gage_linearity_bias(df, params, L: Lang, outdir):
    """reference (true value of each part); measurement; process_variation (e.g. 6*sigma process or tolerance) for %linearity."""
    rf, ms = params["reference"], params["measurement"]; d = df[[rf, ms]].dropna(); x = d[rf].values.astype(float); y = d[ms].values.astype(float) - x; pv = params.get("process_variation")
    sl = stats.linregress(x, y); n = len(x); tcrit = stats.t.ppf(.975, n - 2); g = d.assign(bias=y).groupby(rf)["bias"]; per = pd.DataFrame({"Reference": g.mean().index, "N": g.size().values, "Mean bias": g.mean().values, "Std Dev": g.std().values})
    per["t"] = per["Mean bias"] / (per["Std Dev"] / np.sqrt(per["N"])); per["p-value"] = 2 * stats.t.sf(abs(per["t"]), per["N"] - 1)
    if pv: per["%Bias of process variation"] = 100 * abs(per["Mean bias"]) / float(pv)
    res = Result(L.t("Linearity and bias")); res.table("Bias by reference", per)
    row = {"Slope": sl.slope, "Slope p-value": sl.pvalue, "Slope low 95%": sl.slope - tcrit * sl.stderr, "Slope high 95%": sl.slope + tcrit * sl.stderr, "Intercept (avg bias at ref=0)": sl.intercept, "R2": sl.rvalue ** 2, "Average bias": y.mean(), "Average bias p-value": stats.ttest_1samp(y, 0).pvalue}
    if pv: row.update({"Linearity |slope|*PV": abs(sl.slope) * float(pv), "% Linearity": 100 * abs(sl.slope)}); row["% Bias (avg)"] = 100 * abs(y.mean()) / float(pv)
    res.table("Linearity regression (bias ~ reference)", pd.DataFrame([row]))
    xs = np.linspace(x.min(), x.max(), 50); se = np.sqrt(np.sum((y - (sl.intercept + sl.slope * x)) ** 2) / (n - 2) * (1 / n + (xs - x.mean()) ** 2 / np.sum((x - x.mean()) ** 2))); fig, ax = plt.subplots(figsize=(6.2, 4)); ax.plot(x, y, "o", ms=4, alpha=.6); ax.plot(g.mean().index, g.mean().values, "s", color=PALETTE[3]); ax.plot(xs, sl.intercept + sl.slope * xs, color="k"); ax.fill_between(xs, sl.intercept + sl.slope * xs - tcrit * se, sl.intercept + sl.slope * xs + tcrit * se, color="grey", alpha=.25); ax.axhline(0, color="green", ls="--"); ax.set_xlabel("Reference"); ax.set_ylabel("Bias"); ax.set_title(L.fig("Linearity and bias")); res.fig("linearity", save_fig(fig, outdir, "gage_linearity"))
    res.summary.update({"slope": float(sl.slope), "slope_p": float(sl.pvalue), "avg_bias": float(y.mean())}); res.note("Linearity is not shown if the 95% band around the fitted line includes the bias=0 line everywhere (AIAG MSA 4th ed.).")
    return res


def gage_nested(df, params, L: Lang, outdir):
    """operator; part (unique within operator - destructive testing); measurement; tolerance (optional)."""
    op, pt, ms = params["operator"], params["part"], params["measurement"]; d = df[[op, pt, ms]].dropna().copy(); d[pt] = d[op].astype(str) + "|" + d[pt].astype(str); a = d[op].nunique(); parts = d.groupby(op)[pt].nunique(); r = d.groupby(pt).size()
    bal = parts.nunique() == 1 and r.nunique() == 1; y = d[ms].values.astype(float); gm = y.mean()
    if bal:
        b, rr = int(parts.iloc[0]), int(r.iloc[0]); om = d.groupby(op)[ms].transform("mean").values; pm = d.groupby(pt)[ms].transform("mean").values
        ssA = float(((om - gm) ** 2).sum()); ssB = float(((pm - om) ** 2).sum()); ssE = float(((y - pm) ** 2).sum()); msA, msB, msE = ssA / (a - 1), ssB / (a * (b - 1)), ssE / (a * b * (rr - 1))
        v_rep = msE; v_part = max((msB - msE) / rr, 0.0); v_op = max((msA - msB) / (b * rr), 0.0)
        res = Result(L.t("Nested Gage R&R")); res.table("Nested ANOVA", pd.DataFrame([{"Source": "Operator", "DF": a - 1, "SS": ssA, "MS": msA, "F": msA / msB, "p": stats.f.sf(msA / msB, a - 1, a * (b - 1))}, {"Source": "Part(Operator)", "DF": a * (b - 1), "SS": ssB, "MS": msB, "F": msB / msE, "p": stats.f.sf(msB / msE, a * (b - 1), a * b * (rr - 1))}, {"Source": "Repeatability", "DF": a * b * (rr - 1), "SS": ssE, "MS": msE, "F": np.nan, "p": np.nan}]))
    else:
        from ss_models2 import mixed
        sub = mixed(d, {"response": ms, "random": [op, pt]}, L, outdir); vc = sub.summary["variance_components"]; v_op, v_part, v_rep = vc[op], vc[pt], vc["Residual"]; res = Result(L.t("Nested Gage R&R")); res.note("Unbalanced data: components by REML.")
    v_grr = v_rep + v_op; tot = v_grr + v_part; tol = params.get("tolerance")
    rows = [("Repeatability", v_rep), ("Reproducibility (operator)", v_op), ("Gage R&R", v_grr), ("Part-to-part (within operator)", v_part), ("Total", tot)]
    res.table("Variance components", pd.DataFrame([{"Source": n, "Variance": v, "Std Dev": math.sqrt(v), "% Contribution": 100 * v / tot, "% Study Var (6s)": 100 * math.sqrt(v / tot), **({"% Tolerance": 100 * 6 * math.sqrt(v) / float(tol)} if tol else {})} for n, v in rows]))
    ndc = int(math.floor(1.41 * math.sqrt(v_part) / math.sqrt(v_grr))) if v_grr > 0 else None
    res.table("Number of distinct categories", pd.DataFrame([{"ndc": ndc, "Note": "ndc >= 5 commonly required (AIAG)"}]))
    fig, ax = plt.subplots(figsize=(6.5, 3.8)); ax.bar([n for n, _ in rows], [100 * v / tot for _, v in rows], color=PALETTE[0]); plt.setp(ax.get_xticklabels(), rotation=25, ha="right"); ax.set_ylabel("% contribution"); ax.set_title(L.fig("Nested Gage R&R")); res.fig("nested GRR", save_fig(fig, outdir, "gage_nested"))
    res.summary.update({"grr_pct_contrib": float(100 * v_grr / tot), "grr_pct_study": float(100 * math.sqrt(v_grr / tot)), "ndc": ndc}); res.note("Nested design: each part is measured by only one operator (destructive test). Part variation is therefore confounded with batch homogeneity.")
    return res


def variability_chart(df, params, L: Lang, outdir):
    """response; factors: list (outer->inner, up to 3).  Multi-vari / JMP Variability chart with cell means and variance components."""
    y, fs = params["response"], params["factors"]; d = df[[y] + fs].dropna().copy(); d["_k"] = d[fs].astype(str).agg("|".join, axis=1); keys = list(dict.fromkeys(d.sort_values(fs, key=lambda c: c.astype(str))["_k"])); pos = {k: i for i, k in enumerate(keys)}
    fig, ax = plt.subplots(figsize=(max(7, .35 * len(keys) + 3), 4.2)); xs = d["_k"].map(pos).values + np.random.default_rng(1).uniform(-.15, .15, len(d)); ax.plot(xs, d[y], "o", ms=3, alpha=.6); cm = d.groupby("_k")[y].mean(); ax.plot([pos[k] for k in cm.index], cm.values, "-s", color=PALETTE[3], ms=4)
    ax.set_xticks(range(len(keys))); ax.set_xticklabels([k.replace("|", "\n") for k in keys], fontsize=7); ax.set_ylabel(y); ax.set_title(L.fig("Variability chart")); res = Result(L.t("Variability chart")); res.fig("variability chart", save_fig(fig, outdir, "variability"))
    res.table("Cell statistics", d.groupby(fs)[y].agg(["count", "mean", "std", "min", "max"]).reset_index())
    from ss_models2 import nested_anova
    try:
        sub = nested_anova(d, {"response": y, "factors": fs}, L, outdir); res.table("Variance components (nested, REML)", sub.tables[1][1]); res.summary = sub.summary
    except Exception as e: res.note(f"Variance components not computed: {e}")
    return res


def sampling_plan(df, params, L: Lang, outdir):
    """Attribute single sampling plan. Give n and c to get OC curve; or aql, alpha, ltpd, beta to search the smallest plan; optional lot_size N for AOQ/ATI."""
    if params.get("n") is None:
        aql, ltpd, al, be = float(params["aql"]), float(params["ltpd"]), float(params.get("alpha", .05)), float(params.get("beta", .10)); found = None
        for n in range(1, 5000):
            for c in range(0, n):
                if stats.binom.cdf(c, n, aql) >= 1 - al and stats.binom.cdf(c, n, ltpd) <= be: found = (n, c); break
            if found: break
        if not found: raise ValueError("no plan found up to n=5000")
        n, c = found
    else: n, c = int(params["n"]), int(params["c"]); aql, ltpd = params.get("aql"), params.get("ltpd")
    N = params.get("lot_size"); p = np.linspace(0, min(1, (c + 1) / n * 4 + .02), 200); pa = stats.binom.cdf(c, n, p); res = Result(L.t("Acceptance sampling"))
    row = {"n": n, "c": c, "Reject if defectives >": c}
    if aql is not None: row.update({"AQL": float(aql), "P(accept) at AQL": float(stats.binom.cdf(c, n, float(aql))), "Producer risk": float(1 - stats.binom.cdf(c, n, float(aql)))})
    if ltpd is not None: row.update({"LTPD": float(ltpd), "P(accept) at LTPD (consumer risk)": float(stats.binom.cdf(c, n, float(ltpd)))})
    res.table("Plan", pd.DataFrame([row])); tb = pd.DataFrame({"p (lot fraction defective)": p[::20], "P(accept)": pa[::20]})
    fig, ax = plt.subplots(figsize=(6.4, 4)); ax.plot(p, pa, color=PALETTE[0], label="OC"); ax.set_xlabel("lot fraction defective"); ax.set_ylabel("P(accept)")
    if N:
        N = int(N); aoq = p * pa * (N - n) / N; ati = n * pa + N * (1 - pa); tb["AOQ"] = aoq[::20]; tb["ATI"] = ati[::20]; ax.plot(p, aoq / aoq.max(), "--", color=PALETTE[2], label="AOQ (scaled)"); res.summary["AOQL"] = float(aoq.max())
    ax.legend(); ax.set_title(L.fig("Acceptance sampling")); res.table("OC curve points", tb); res.fig("OC curve", save_fig(fig, outdir, "oc_curve")); res.summary.update({"n": n, "c": c}); res.note("Binomial (infinite-lot) OC curve; for small lots use hypergeometric. ISO 2859-1 / ANSI Z1.4 tables should be consulted for contractual plans.")
    return res
