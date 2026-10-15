"""ss_spc.py - Control charts, process capability, Gage R&R (JMP: Control Chart Builder, Process Capability, Measurement Systems Analysis; SPSS: Control Charts, Capability).
Constants follow AIAG/ASTM; Gage R&R = crossed ANOVA method (AIAG MSA 4th ed., interaction alpha=0.25).
"""
from __future__ import annotations
import numpy as np, pandas as pd
from scipy import stats
from scipy.special import gammaln
import matplotlib.pyplot as plt
from ss_common import Result, Lang, save_fig, PALETTE

D2 = [1.128, 1.693, 2.059, 2.326, 2.534, 2.704, 2.847, 2.970, 3.078, 3.173, 3.258, 3.336, 3.407, 3.472, 3.532, 3.588, 3.640, 3.689, 3.735, 3.778, 3.819, 3.858, 3.895, 3.931]
D3C = [0.853, 0.888, 0.880, 0.864, 0.848, 0.833, 0.820, 0.808, 0.797, 0.787, 0.778, 0.770, 0.763, 0.756, 0.750, 0.744, 0.739, 0.734, 0.729, 0.724, 0.720, 0.716, 0.712, 0.708]


def d2(n): return D2[n - 2]
def d3(n): return D3C[n - 2]
def c4(n): return float(np.sqrt(2 / (n - 1)) * np.exp(gammaln(n / 2) - gammaln((n - 1) / 2)))


def spc_constants(n):
    c = {"d2": d2(n), "A2": 3 / (d2(n) * np.sqrt(n)), "D3": max(0, 1 - 3 * d3(n) / d2(n)), "D4": 1 + 3 * d3(n) / d2(n)}
    c4n = c4(n)
    c.update({"c4": c4n, "A3": 3 / (c4n * np.sqrt(n)), "B3": max(0, 1 - 3 * np.sqrt(1 - c4n ** 2) / c4n), "B4": 1 + 3 * np.sqrt(1 - c4n ** 2) / c4n})
    return c


def nelson_rules(x, cl, sigma, rules=(1, 2, 3, 5, 6)):
    """Return {index: [rule numbers]} for points violating tests (Nelson/Western Electric subset). sigma may be array (variable limits)."""
    x = np.asarray(x, float); sig = np.broadcast_to(np.asarray(sigma, float), x.shape)
    z = (x - cl) / sig
    v = {}
    def add(i, r): v.setdefault(int(i), []).append(r)
    n = len(x)
    if 1 in rules:
        for i in np.where(np.abs(z) > 3)[0]: add(i, 1)
    if 2 in rules:
        for i in range(8, n):
            w = z[i - 8:i + 1]
            if (w > 0).all() or (w < 0).all(): add(i, 2)
    if 3 in rules:
        for i in range(5, n):
            d = np.diff(x[i - 5:i + 1])
            if (d > 0).all() or (d < 0).all(): add(i, 3)
    if 4 in rules:
        for i in range(13, n):
            d = np.sign(np.diff(x[i - 13:i + 1]))
            if (d[1:] * d[:-1] < 0).all(): add(i, 4)
    if 5 in rules:
        for i in range(2, n):
            w = z[i - 2:i + 1]
            if (w > 2).sum() >= 2 or (w < -2).sum() >= 2: add(i, 5)
    if 6 in rules:
        for i in range(4, n):
            w = z[i - 4:i + 1]
            if (w > 1).sum() >= 4 or (w < -1).sum() >= 4: add(i, 6)
    return v


RULE_TXT = {1: "1 pt > 3σ", 2: "9 in a row same side", 3: "6 trending", 4: "14 alternating", 5: "2 of 3 > 2σ", 6: "4 of 5 > 1σ"}


def _plot_chart(ax, x, cl, ucl, lcl, viol, title, L, xlabel="", ylabel="", labels=None, sig_lines=True):
    idx = np.arange(1, len(x) + 1)
    ucl_a = np.broadcast_to(np.asarray(ucl, float), x.shape); lcl_a = np.broadcast_to(np.asarray(lcl, float), x.shape)
    ax.plot(idx, x, "o-", ms=4, lw=1, color=PALETTE[0])
    ax.step(idx, ucl_a, where="mid", color=PALETTE[3], lw=1.3); ax.step(idx, lcl_a, where="mid", color=PALETTE[3], lw=1.3)
    ax.axhline(cl, color="#2E7D32", lw=1.3)
    for i, r in viol.items():
        ax.plot(i + 1, x[i], "o", ms=8, color=PALETTE[3], mfc="none", mew=2)
        ax.annotate(",".join(map(str, r)), (i + 1, x[i]), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=7, color=PALETTE[3])
    xr = idx[-1]
    for val, nm in ((cl, "CL"), (ucl_a[-1], "UCL"), (lcl_a[-1], "LCL")):
        ax.text(xr + 0.3, val, f"{nm}={val:.4g}", fontsize=7, va="center")
    ax.set_title(title); ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.set_xlim(0.5, xr + max(2, len(x) * 0.12))


def control_chart(df, params, L: Lang, outdir):
    """params: type (imr|xbar_r|xbar_s|p|np|c|u), column (measurement or defect count), subgroup (column name with subgroup id) OR subgroup_size (int),
    n (column with sample size, for p/np/u), rules (list, default [1,2,3,5,6]), phase_split (optional row index to freeze limits before)."""
    ct = params.get("type", "imr"); col = params["column"]; rules = params.get("rules", [1, 2, 3, 5, 6])
    res = Result(f"{L.t('Control Chart')} ({ct.upper()}): {col}")
    rows = []
    if ct == "imr":
        x = df[col].dropna().values.astype(float)
        mr = np.abs(np.diff(x)); mrb = mr.mean(); sig = mrb / 1.128; cl = x.mean()
        ucl, lcl = cl + 3 * sig, cl - 3 * sig
        v1 = nelson_rules(x, cl, sig, rules)
        v2 = {i + 1: [1] for i in np.where(mr > 3.267 * mrb)[0]}
        fig, axs = plt.subplots(2, 1, figsize=(11, 7))
        _plot_chart(axs[0], x, cl, ucl, lcl, v1, f"{L.fig('Individual')}: {col}", L)
        _plot_chart(axs[1], mr, mrb, 3.267 * mrb, 0, {k - 1: r for k, r in v2.items()}, L.fig("Moving Range"), L)
        fig.tight_layout()
        res.fig(L.t("Control Chart"), save_fig(fig, outdir, f"imr_{col}"))
        res.table("Limits", pd.DataFrame([{"Chart": "I", L.t("Center line"): cl, L.t("UCL"): ucl, L.t("LCL"): lcl, "sigma (MRbar/1.128)": sig},
                                          {"Chart": "MR", L.t("Center line"): mrb, L.t("UCL"): 3.267 * mrb, L.t("LCL"): 0.0, "sigma (MRbar/1.128)": sig}]))
        for i, r in v1.items(): rows.append({"Chart": "I", L.t("Subgroup"): i + 1, L.t("Value"): x[i], "Tests": ",".join(RULE_TXT[k] for k in r)})
        res.summary = {"mean": cl, "sigma_within": sig, "UCL": ucl, "LCL": lcl, "n_ooc": len(v1)}
    elif ct in ("xbar_r", "xbar_s"):
        sg = params.get("subgroup")
        if sg:
            g = [v.dropna().values.astype(float) for _, v in df.groupby(sg, sort=False)[col]]
        else:
            n = int(params["subgroup_size"]); xs = df[col].dropna().values.astype(float)
            g = [xs[i:i + n] for i in range(0, len(xs) - n + 1, n)]
        sizes = np.array([len(a) for a in g]); n = int(round(sizes.mean()))
        if len(set(sizes)) > 1:
            res.note(f"Unequal subgroup sizes {sorted(set(sizes))}: limits use per-subgroup constants for Xbar.")
        xb = np.array([a.mean() for a in g]); rr = np.array([np.ptp(a) for a in g]); ss = np.array([a.std(ddof=1) for a in g])
        cl = float(np.average(xb, weights=sizes))
        if ct == "xbar_r":
            rb = rr.mean(); sig = rb / d2(n)
            sig_i = np.array([rb / d2(len(a)) / np.sqrt(len(a)) for a in g])
            c = spc_constants(n); spread, sp_cl, sp_u, sp_l, nm = rr, rb, c["D4"] * rb, c["D3"] * rb, "Range"
        else:
            sb = ss.mean(); sig = sb / c4(n)
            sig_i = np.array([sb / c4(len(a)) / np.sqrt(len(a)) for a in g])
            c = spc_constants(n); spread, sp_cl, sp_u, sp_l, nm = ss, sb, c["B4"] * sb, c["B3"] * sb, "S"
        ucl, lcl = cl + 3 * sig_i, cl - 3 * sig_i
        v1 = nelson_rules(xb, cl, sig_i, rules)
        v2 = {i: [1] for i in np.where((spread > sp_u) | (spread < sp_l))[0]}
        fig, axs = plt.subplots(2, 1, figsize=(11, 7))
        _plot_chart(axs[0], xb, cl, ucl, lcl, v1, f"Xbar: {col}", L, xlabel=L.fig("Subgroup"))
        _plot_chart(axs[1], spread, sp_cl, sp_u, sp_l, v2, L.fig("Range") if ct == "xbar_r" else "S", L, xlabel=L.fig("Subgroup"))
        fig.tight_layout()
        res.fig(L.t("Control Chart"), save_fig(fig, outdir, f"{ct}_{col}"))
        res.table("Limits", pd.DataFrame([{"Chart": "Xbar", L.t("Center line"): cl, L.t("UCL"): ucl[0], L.t("LCL"): lcl[0], "sigma_within": sig, "n": n},
                                          {"Chart": nm, L.t("Center line"): sp_cl, L.t("UCL"): sp_u, L.t("LCL"): sp_l, "sigma_within": sig, "n": n}]))
        for i, r in v1.items(): rows.append({"Chart": "Xbar", L.t("Subgroup"): i + 1, L.t("Value"): xb[i], "Tests": ",".join(RULE_TXT[k] for k in r)})
        for i in v2: rows.append({"Chart": nm, L.t("Subgroup"): i + 1, L.t("Value"): spread[i], "Tests": RULE_TXT[1]})
        res.summary = {"grand_mean": cl, "sigma_within": sig, "n_subgroups": len(g), "subgroup_size": n, "n_ooc": len(rows)}
    else:  # attribute charts
        d = df.dropna(subset=[col]); x = d[col].values.astype(float)
        if ct in ("p", "np", "u"):
            nn = d[params["n"]].values.astype(float) if params.get("n") else np.full(len(x), float(params["sample_size"]))
        if ct == "p":
            pb = x.sum() / nn.sum(); y = x / nn; sg = np.sqrt(pb * (1 - pb) / nn); cl_ = pb; ttl = "P"
        elif ct == "np":
            pb = x.sum() / nn.sum(); y = x; sg = np.sqrt(nn * pb * (1 - pb)); cl_ = nn * pb; ttl = "NP"
        elif ct == "c":
            cb = x.mean(); y = x; sg = np.full(len(x), np.sqrt(cb)); cl_ = np.full(len(x), cb); ttl = "C"
        else:
            ub = x.sum() / nn.sum(); y = x / nn; sg = np.sqrt(ub / nn); cl_ = np.full(len(x), ub); ttl = "U"
        cl_a = np.broadcast_to(cl_, y.shape) if np.ndim(cl_) else np.full(len(y), cl_)
        ucl = cl_a + 3 * sg; lcl = np.clip(cl_a - 3 * sg, 0, None)
        if ct == "p": ucl = np.clip(ucl, None, 1)
        z = (y - cl_a) / sg
        v1 = nelson_rules(y, cl_a, sg, rules)
        fig, ax = plt.subplots(figsize=(11, 4.2))
        idx = np.arange(1, len(y) + 1)
        ax.plot(idx, y, "o-", ms=4, lw=1, color=PALETTE[0]); ax.step(idx, ucl, where="mid", color=PALETTE[3]); ax.step(idx, lcl, where="mid", color=PALETTE[3]); ax.plot(idx, cl_a, color="#2E7D32")
        for i, r in v1.items():
            ax.plot(i + 1, y[i], "o", ms=8, color=PALETTE[3], mfc="none", mew=2)
        ax.set_title(f"{ttl} {L.fig('Control Chart')}: {col}")
        res.fig(L.t("Control Chart"), save_fig(fig, outdir, f"{ct}_{col}"))
        res.table("Limits", pd.DataFrame({L.t("Subgroup"): idx, L.t("Value"): y, L.t("Center line"): cl_a, L.t("UCL"): ucl, L.t("LCL"): lcl}).head(60))
        for i, r in v1.items(): rows.append({"Chart": ttl, L.t("Subgroup"): i + 1, L.t("Value"): y[i], "Tests": ",".join(RULE_TXT[k] for k in r)})
        res.summary = {"center": float(np.mean(cl_a)), "n_ooc": len(v1)}
    res.table(L.t("Out-of-control points"), pd.DataFrame(rows) if rows else pd.DataFrame([{L.t("Verdict"): "No out-of-control signals"}]))
    res.note("Rules (Nelson/Western Electric): " + "; ".join(f"{k}={RULE_TXT[k]}" for k in rules))
    res.note("Limits are computed from the data supplied; use only after confirming the process was stable (Phase I) before applying them to future data (Phase II).")
    return res


# ----------------------------------------------------------------------------
def capability(df, params, L: Lang, outdir):
    """params: column, LSL, USL (at least one), target (optional), subgroup (column) or subgroup_size (default 1 -> MR-based sigma), distribution (normal|lognormal|weibull)."""
    col = params["column"]; LSL = params.get("LSL"); USL = params.get("USL"); T = params.get("target")
    x = df[col].dropna().values.astype(float); n = len(x)
    if LSL is None and USL is None:
        raise ValueError("Provide LSL and/or USL")
    mean, s_all = x.mean(), x.std(ddof=1)
    sg = params.get("subgroup"); m = int(params.get("subgroup_size", 1))
    if sg:
        g = [v.dropna().values.astype(float) for _, v in df.groupby(sg, sort=False)[col]]
        m = int(round(np.mean([len(a) for a in g])))
        sw = np.sqrt(np.sum([(len(a) - 1) * a.var(ddof=1) for a in g]) / np.sum([len(a) - 1 for a in g])) if m > 1 else s_all
        w_method = "pooled within-subgroup sd"
    elif m > 1:
        g = [x[i:i + m] for i in range(0, n - m + 1, m)]
        sw = np.sqrt(np.mean([a.var(ddof=1) for a in g])); w_method = "pooled within-subgroup sd"
    else:
        sw = np.abs(np.diff(x)).mean() / 1.128; w_method = "MRbar/1.128"
    dist = params.get("distribution", "normal")
    def idx(sig, mu):
        cp = (USL - LSL) / (6 * sig) if (LSL is not None and USL is not None) else np.nan
        cpu = (USL - mu) / (3 * sig) if USL is not None else np.nan
        cpl = (mu - LSL) / (3 * sig) if LSL is not None else np.nan
        return cp, cpu, cpl, np.nanmin([cpu, cpl])
    cp, cpu, cpl, cpk = idx(sw, mean); pp, ppu, ppl, ppk = idx(s_all, mean)
    cpm = (USL - LSL) / (6 * np.sqrt(s_all ** 2 + (mean - T) ** 2)) if (T is not None and LSL is not None and USL is not None) else np.nan
    z = stats.norm.ppf(.975)
    def ci(v, nn): return (v * (1 - z / np.sqrt(2 * (nn - 1))), v * (1 + z / np.sqrt(2 * (nn - 1))))
    def ci_k(v, nn): h = z * np.sqrt(1 / (9 * nn) + v ** 2 / (2 * (nn - 1))); return (v - h, v + h)
    def ppm(sig, mu): return ((stats.norm.cdf(LSL, mu, sig) if LSL is not None else 0) * 1e6, (stats.norm.sf(USL, mu, sig) if USL is not None else 0) * 1e6)
    pw, pv = ppm(sw, mean), ppm(s_all, mean)
    obs_lo = (x < LSL).sum() / n * 1e6 if LSL is not None else 0; obs_hi = (x > USL).sum() / n * 1e6 if USL is not None else 0
    res = Result(f"{L.t('Process Capability')}: {col}")
    res.table("Specs / Statistics", pd.DataFrame([{L.t("LSL"): LSL, L.t("Target"): T, L.t("USL"): USL, "N": n, L.t("Mean"): mean, L.t("Overall sigma"): s_all, L.t("Within sigma"): sw, L.t("Method"): w_method}]))
    res.table("Capability indices", pd.DataFrame([
        {"Index": "Cp", "Value": cp, "Lower 95%": ci(cp, n)[0] if not np.isnan(cp) else np.nan, "Upper 95%": ci(cp, n)[1] if not np.isnan(cp) else np.nan},
        {"Index": "CPU", "Value": cpu}, {"Index": "CPL", "Value": cpl},
        {"Index": "Cpk", "Value": cpk, "Lower 95%": ci_k(cpk, n)[0], "Upper 95%": ci_k(cpk, n)[1]},
        {"Index": "Pp", "Value": pp, "Lower 95%": ci(pp, n)[0] if not np.isnan(pp) else np.nan, "Upper 95%": ci(pp, n)[1] if not np.isnan(pp) else np.nan},
        {"Index": "PPU", "Value": ppu}, {"Index": "PPL", "Value": ppl},
        {"Index": "Ppk", "Value": ppk, "Lower 95%": ci_k(ppk, n)[0], "Upper 95%": ci_k(ppk, n)[1]},
        {"Index": "Cpm", "Value": cpm}]))
    res.table("PPM", pd.DataFrame([
        {L.t("Method"): "Observed", L.t("PPM below LSL"): obs_lo, L.t("PPM above USL"): obs_hi, L.t("PPM total"): obs_lo + obs_hi},
        {L.t("Method"): "Expected within (normal)", L.t("PPM below LSL"): pw[0], L.t("PPM above USL"): pw[1], L.t("PPM total"): sum(pw)},
        {L.t("Method"): "Expected overall (normal)", L.t("PPM below LSL"): pv[0], L.t("PPM above USL"): pv[1], L.t("PPM total"): sum(pv)}]))
    sw_p = stats.shapiro(x).pvalue if n <= 5000 else np.nan
    res.table(L.t("Normality tests"), pd.DataFrame([{"Shapiro-Wilk p": sw_p, L.t("Skewness"): stats.skew(x), L.t("Kurtosis"): stats.kurtosis(x)}]))
    summ = {"Cp": cp, "Cpk": cpk, "Pp": pp, "Ppk": ppk, "mean": mean, "sigma_within": sw, "sigma_overall": s_all, "shapiro_p": sw_p}
    if dist != "normal" or (not np.isnan(sw_p) and sw_p < .05):
        if dist == "normal":
            res.note("Normality rejected (Shapiro-Wilk p<.05): normal-based indices may mislead. Re-run with distribution='lognormal' or 'weibull' (percentile/ISO method) or transform the data.")
        else:
            D = stats.lognorm if dist == "lognormal" else stats.weibull_min
            pr = D.fit(x, floc=0) if dist else None
            p_lo, p_med, p_hi = D.ppf([.00135, .5, .99865], *pr)
            ppk_np = min((USL - p_med) / (p_hi - p_med) if USL is not None else np.inf, (p_med - LSL) / (p_med - p_lo) if LSL is not None else np.inf)
            ppu_ = (USL - p_med) / (p_hi - p_med) if USL is not None else np.nan; ppl_ = (p_med - LSL) / (p_med - p_lo) if LSL is not None else np.nan
            pp_np = (USL - LSL) / (p_hi - p_lo) if (USL is not None and LSL is not None) else np.nan
            ppm_np = ((D.cdf(LSL, *pr) if LSL is not None else 0) + (D.sf(USL, *pr) if USL is not None else 0)) * 1e6
            res.table(f"Non-normal capability ({dist}, percentile method)", pd.DataFrame([{"Pp": pp_np, "PPU": ppu_, "PPL": ppl_, "Ppk": ppk_np, "PPM expected": ppm_np}]))
            summ.update({"Ppk_nonnormal": ppk_np, "dist": dist})
    fig, ax = plt.subplots(figsize=(9, 4.4))
    ax.hist(x, bins="auto", density=True, color=PALETTE[0], alpha=.6, edgecolor="white")
    xs = np.linspace(min(x.min(), LSL if LSL is not None else x.min()) - s_all, max(x.max(), USL if USL is not None else x.max()) + s_all, 300)
    ax.plot(xs, stats.norm.pdf(xs, mean, sw), color="#2E7D32", lw=2, label="Within")
    ax.plot(xs, stats.norm.pdf(xs, mean, s_all), color=PALETTE[1], lw=2, ls="--", label="Overall")
    for v, nm in ((LSL, "LSL"), (USL, "USL"), (T, "Target")):
        if v is not None:
            ax.axvline(v, color=PALETTE[3] if nm != "Target" else "#444", lw=2, ls="-" if nm != "Target" else ":"); ax.text(v, ax.get_ylim()[1] * .95, nm, rotation=90, va="top", ha="right", fontsize=8)
    ax.set_title(f"{L.fig('Capability histogram')}: {col}   Cpk={cpk:.2f}  Ppk={ppk:.2f}"); ax.legend()
    res.fig(L.t("Process Capability"), save_fig(fig, outdir, f"cap_{col}"))
    fig, ax = plt.subplots(figsize=(5.5, 4.5)); (osm, osr), (s_, i_, r_) = stats.probplot(x, dist="norm")
    ax.plot(osr, osm, "o", ms=3.5, color=PALETTE[0]); ax.plot(s_ * np.asarray(osm) + i_, osm, color=PALETTE[3]); ax.set_title(L.fig("Normal Q-Q plot"))
    for v in (LSL, USL):
        if v is not None: ax.axvline(v, color=PALETTE[3], ls="--")
    res.fig(L.t("Normal Q-Q plot"), save_fig(fig, outdir, f"cap_qq_{col}"))
    verdict = "capable (Cpk >= 1.33)" if cpk >= 1.33 else ("marginal (1.00 <= Cpk < 1.33)" if cpk >= 1 else "NOT capable (Cpk < 1.00)")
    res.note(f"Cpk={cpk:.3f}: {verdict}. Thresholds (1.33 / 1.67 for critical characteristics) are customer/industry conventions - confirm against your customer requirement.")
    if abs(cpk - cp) / cp > 0.1 if not np.isnan(cp) else False:
        res.note("Cp noticeably larger than Cpk: the process is off-center; re-centering improves capability without reducing variation.")
    res.summary = summ
    return res


# ----------------------------------------------------------------------------
def gage_rr(df, params, L: Lang, outdir):
    """Crossed Gage R&R (ANOVA). params: part, operator, measurement, tolerance (USL-LSL, optional), study_sigma (6 default), alpha_int (0.25)."""
    pc, oc, mc = params["part"], params["operator"], params["measurement"]
    tol = params.get("tolerance"); K = float(params.get("study_sigma", 6)); a_int = float(params.get("alpha_int", .25))
    d = df[[pc, oc, mc]].dropna().copy(); d[pc] = d[pc].astype(str); d[oc] = d[oc].astype(str)
    import re as _re
    _nat = lambda v: [int(t) if t.isdigit() else t.lower() for t in _re.split(r"(\d+)", str(v))]
    parts = sorted(d[pc].unique(), key=_nat); ops = sorted(d[oc].unique(), key=_nat)
    p, o = len(parts), len(ops)
    cnt = d.groupby([pc, oc]).size()
    if cnt.nunique() != 1 or len(cnt) != p * o:
        raise ValueError("Crossed ANOVA Gage R&R needs a balanced design: every part measured by every operator the same number of times. Check counts per Part x Operator.")
    r = int(cnt.iloc[0]); N = len(d); y = d[mc].values; gm = y.mean()
    mp = d.groupby(pc)[mc].mean(); mo = d.groupby(oc)[mc].mean(); mpo = d.groupby([pc, oc])[mc].mean()
    ss_p = o * r * ((mp - gm) ** 2).sum(); ss_o = p * r * ((mo - gm) ** 2).sum()
    ss_t = ((y - gm) ** 2).sum(); ss_cell = r * ((mpo - gm) ** 2).sum()
    ss_i = ss_cell - ss_p - ss_o; ss_e = ss_t - ss_cell
    df_p, df_o, df_i, df_e = p - 1, o - 1, (p - 1) * (o - 1), p * o * (r - 1)
    ms_p, ms_o, ms_i = ss_p / df_p, ss_o / df_o, ss_i / df_i
    ms_e = ss_e / df_e if df_e > 0 else np.nan
    if df_e <= 0 or r < 2:
        raise ValueError("Need at least 2 repeated measurements per part/operator (replicates).")
    F_p, F_o, F_i = ms_p / ms_i, ms_o / ms_i, ms_i / ms_e
    p_p, p_o, p_i = stats.f.sf(F_p, df_p, df_i), stats.f.sf(F_o, df_o, df_i), stats.f.sf(F_i, df_i, df_e)
    aov = pd.DataFrame([
        {"Source": pc, "DF": df_p, "Sum of Squares": ss_p, "Mean Square": ms_p, "F Ratio": F_p, "Prob > F": p_p},
        {"Source": oc, "DF": df_o, "Sum of Squares": ss_o, "Mean Square": ms_o, "F Ratio": F_o, "Prob > F": p_o},
        {"Source": f"{pc}*{oc}", "DF": df_i, "Sum of Squares": ss_i, "Mean Square": ms_i, "F Ratio": F_i, "Prob > F": p_i},
        {"Source": "Repeatability (Error)", "DF": df_e, "Sum of Squares": ss_e, "Mean Square": ms_e, "F Ratio": np.nan, "Prob > F": np.nan},
        {"Source": "Total", "DF": N - 1, "Sum of Squares": ss_t, "Mean Square": np.nan, "F Ratio": np.nan, "Prob > F": np.nan}])
    int_sig = p_i < a_int
    if int_sig:
        v_rep = ms_e; v_int = max(0, (ms_i - ms_e) / r); v_op = max(0, (ms_o - ms_i) / (p * r)); v_part = max(0, (ms_p - ms_i) / (o * r))
    else:
        ms_pool = (ss_i + ss_e) / (df_i + df_e)
        v_rep = ms_pool; v_int = 0.0; v_op = max(0, (ms_o - ms_pool) / (p * r)); v_part = max(0, (ms_p - ms_pool) / (o * r))
    v_repro = v_op + v_int; v_grr = v_rep + v_repro; v_tot = v_grr + v_part
    sd = lambda v: np.sqrt(v)
    comps = [("Gage R&R", v_grr), (L.t("Repeatability"), v_rep), (L.t("Reproducibility"), v_repro), (f"  {L.t('Operator')}", v_op), (f"  {L.t('Operator')}*{L.t('Part')}", v_int), (L.t("Part-to-Part"), v_part), (L.t("Total"), v_tot)]
    rows = []
    for nm, v in comps:
        rows.append({L.t("Source"): nm, L.t("Variance Component"): v, L.t("% of Total Variance"): 100 * v / v_tot, L.t("StdDev"): sd(v), "Study Var": K * sd(v),
                     L.t("% Study Var"): 100 * sd(v) / sd(v_tot), L.t("% Tolerance"): (100 * K * sd(v) / tol) if tol else np.nan})
    ndc = int(np.floor(1.41 * sd(v_part) / sd(v_grr))) if v_grr > 0 else None
    pct_sv = 100 * sd(v_grr) / sd(v_tot); pct_tol = 100 * K * sd(v_grr) / tol if tol else None
    crit = pct_tol if pct_tol is not None else pct_sv
    verdict = "ACCEPTABLE (<10%)" if crit < 10 else ("MARGINAL (10-30%) - may be acceptable depending on application/cost" if crit <= 30 else "UNACCEPTABLE (>30%) - improve the measurement system")
    res = Result(f"{L.t('Gage R&R')}: {mc}")
    res.table(L.t("Analysis of Variance") + f" (Part x Operator, interaction {'kept' if int_sig else 'pooled'}, alpha={a_int})", aov)
    res.table(L.t("Gage R&R"), pd.DataFrame(rows))
    res.table(L.t("Number of Distinct Categories"), pd.DataFrame([{"ndc": ndc, L.t("Verdict"): verdict, "p x o x r": f"{p} x {o} x {r}"}]))
    # variability chart (JMP style)
    fig, axs = plt.subplots(2, 3, figsize=(15, 8.5))
    ax = axs[0, 0]; labs = ["Gage R&R", "Repeat.", "Reprod.", "Part-to-Part"]; vals = [v_grr, v_rep, v_repro, v_part]
    w = .25; xs_ = np.arange(4)
    ax.bar(xs_ - w, [100 * v / v_tot for v in vals], w, label="% Contribution", color=PALETTE[0]); ax.bar(xs_, [100 * sd(v) / sd(v_tot) for v in vals], w, label="% Study Var", color=PALETTE[1])
    if tol: ax.bar(xs_ + w, [100 * K * sd(v) / tol for v in vals], w, label="% Tolerance", color=PALETTE[2])
    ax.set_xticks(xs_); ax.set_xticklabels(labs, fontsize=8); ax.legend(fontsize=7); ax.set_title("Components of Variation")
    ax = axs[0, 1]
    for j, op in enumerate(ops):
        dd = d[d[oc] == op]; xx = [parts.index(a) + (j - (o - 1) / 2) * .18 for a in dd[pc]]
        ax.plot(xx, dd[mc], "o", ms=3, color=PALETTE[j % 8], alpha=.7, label=op)
    ax.plot(range(p), [mp[a] for a in parts], "k-", lw=1.5, label="Part mean"); ax.set_xticks(range(p)); ax.set_xticklabels(parts, fontsize=7); ax.legend(fontsize=7, ncol=2); ax.set_title(f"{mc} by {pc} / {oc}")
    ax = axs[0, 2]
    for j, op in enumerate(ops):
        ax.plot(range(p), [mpo[(a, op)] for a in parts], "o-", color=PALETTE[j % 8], label=op)
    ax.set_xticks(range(p)); ax.set_xticklabels(parts, fontsize=7); ax.legend(fontsize=7); ax.set_title(f"{pc} * {oc} interaction")
    ax = axs[1, 0]; ax.boxplot([d.loc[d[oc] == op, mc] for op in ops], tick_labels=ops); ax.set_title(f"{mc} by {oc}")
    ax = axs[1, 1]; ax.boxplot([d.loc[d[pc] == a, mc] for a in parts], tick_labels=parts); ax.tick_params(axis="x", labelsize=7); ax.set_title(f"{mc} by {pc}")
    ax = axs[1, 2]
    rg = d.groupby([pc, oc])[mc].agg(lambda s: np.ptp(s)).reset_index(); rb = rg[mc].mean(); cc = spc_constants(r) if 2 <= r <= 25 else None
    for j, op in enumerate(ops):
        z_ = rg[rg[oc] == op]; ax.plot(range(p), [z_.loc[z_[pc] == a, mc].values[0] for a in parts], "o-", color=PALETTE[j % 8], label=op)
    if cc: ax.axhline(rb, color="#2E7D32"); ax.axhline(cc["D4"] * rb, color=PALETTE[3])
    ax.set_xticks(range(p)); ax.set_xticklabels(parts, fontsize=7); ax.set_title("Range chart (repeatability consistency)"); ax.legend(fontsize=7)
    fig.tight_layout(); res.fig("Gage R&R report", save_fig(fig, outdir, f"grr_{mc}"))
    res.note(f"Verdict: %Study Var = {pct_sv:.1f}%" + (f", %Tolerance = {pct_tol:.1f}%" if pct_tol is not None else "") + f", ndc = {ndc}. AIAG guideline: <10% acceptable, 10-30% marginal, >30% unacceptable; ndc >= 5 required.")
    res.note(f"Study variation uses {K:g} sigma. Part sample must cover the real process range; ndc/%Study Var are misleading if parts do not span the process variation.")
    if not int_sig:
        res.note("Part x Operator interaction not significant (p >= alpha): pooled into repeatability per AIAG.")
    res.summary = {"pct_study_var": pct_sv, "pct_tolerance": pct_tol, "pct_contribution": 100 * v_grr / v_tot, "ndc": ndc, "verdict": verdict,
                   "sd_repeatability": sd(v_rep), "sd_reproducibility": sd(v_repro), "sd_part": sd(v_part), "interaction_p": p_i}
    return res
