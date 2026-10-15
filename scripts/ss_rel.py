"""ss_rel.py - Reliability & survival: Weibull / lognormal / exponential MLE with right censoring, distribution comparison,
Kaplan-Meier with log-rank test, probability plots. (JMP: Life Distribution, Survival; SPSS: Kaplan-Meier, Survival)."""
from __future__ import annotations
import numpy as np, pandas as pd
from scipy import stats, optimize
from scipy.special import gamma as G
import matplotlib.pyplot as plt
from ss_common import Result, Lang, save_fig, PALETTE


def _data(df, params):
    tc = params["time"]; ec = params.get("event")
    d = df[[tc] + ([ec] if ec else []) + ([params["group"]] if params.get("group") else [])].dropna().copy()
    d = d[d[tc] > 0]
    d["_e"] = d[ec].astype(int) if ec else 1
    return d, tc


def _nll(dist, th, t, e):
    if dist == "weibull":
        b, eta = np.exp(th[0]), np.exp(th[1])
        return -(np.sum(e * (np.log(b) - b * np.log(eta) + (b - 1) * np.log(t))) - np.sum((t / eta) ** b))
    if dist == "lognormal":
        mu, s = th[0], np.exp(th[1])
        z = (np.log(t) - mu) / s
        lp = stats.norm.logpdf(z) - np.log(s * t); ls = stats.norm.logsf(z)
        return -(np.sum(e * lp + (1 - e) * ls))
    lam = np.exp(th[0])
    return -(np.sum(e * np.log(lam)) - lam * np.sum(t))


def _hess(f, x, h=1e-4):
    n = len(x); H = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            xpp = x.copy(); xpm = x.copy(); xmp = x.copy(); xmm = x.copy()
            xpp[i] += h; xpp[j] += h; xpm[i] += h; xpm[j] -= h; xmp[i] -= h; xmp[j] += h; xmm[i] -= h; xmm[j] -= h
            H[i, j] = (f(xpp) - f(xpm) - f(xmp) + f(xmm)) / (4 * h * h)
    return H


def fit_dist(dist, t, e):
    t = np.asarray(t, float); e = np.asarray(e, float)
    if dist == "weibull":
        x0 = np.array([0.0, np.log(t.mean())])
    elif dist == "lognormal":
        x0 = np.array([np.log(t).mean(), np.log(np.log(t).std() + 1e-6)])
    else:
        x0 = np.array([np.log(e.sum() / t.sum())])
    r = optimize.minimize(lambda th: _nll(dist, th, t, e), x0, method="Nelder-Mead", options={"xatol": 1e-9, "fatol": 1e-11, "maxiter": 5000})
    th = r.x
    H = _hess(lambda z: _nll(dist, z, t, e), th)
    try:
        cov = np.linalg.inv(H); se = np.sqrt(np.clip(np.diag(cov), 0, None))
    except np.linalg.LinAlgError:
        cov = np.full_like(H, np.nan); se = np.full(len(th), np.nan)
    ll = -r.fun; k = len(th); n = len(t)
    aicc = -2 * ll + 2 * k + (2 * k * (k + 1) / (n - k - 1) if n - k - 1 > 0 else np.nan)
    return {"dist": dist, "theta": th, "se": se, "cov": cov, "ll": ll, "k": k, "n": n, "aicc": aicc, "bic": -2 * ll + k * np.log(n)}


def sf(fit, t):
    th = fit["theta"]; d = fit["dist"]; t = np.asarray(t, float)
    if d == "weibull": return np.exp(-(t / np.exp(th[1])) ** np.exp(th[0]))
    if d == "lognormal": return stats.norm.sf((np.log(t) - th[0]) / np.exp(th[1]))
    return np.exp(-np.exp(th[0]) * t)


def quantile(fit, p):  # failure probability p -> time
    th = fit["theta"]; d = fit["dist"]
    if d == "weibull": return np.exp(th[1]) * (-np.log(1 - p)) ** (1 / np.exp(th[0]))
    if d == "lognormal": return np.exp(th[0] + np.exp(th[1]) * stats.norm.ppf(p))
    return -np.log(1 - p) / np.exp(th[0])


def mttf(fit):
    th = fit["theta"]; d = fit["dist"]
    if d == "weibull": return np.exp(th[1]) * G(1 + 1 / np.exp(th[0]))
    if d == "lognormal": return np.exp(th[0] + np.exp(2 * th[1]) / 2)
    return 1 / np.exp(th[0])


def _boot_ci(fn, fit, t, e, B=0):
    return None


def km(t, e):
    t = np.asarray(t, float); e = np.asarray(e, int)
    order = np.argsort(t); t, e = t[order], e[order]
    ut = np.unique(t[e == 1]); S = 1.0; var_sum = 0.0; rows = []
    for u in ut:
        n_i = (t >= u).sum(); d_i = ((t == u) & (e == 1)).sum()
        S *= 1 - d_i / n_i
        var_sum += d_i / (n_i * (n_i - d_i)) if n_i > d_i else 0
        se = S * np.sqrt(var_sum)
        # log-log CI
        if S > 0 and var_sum > 0:
            se_ll = np.sqrt(var_sum) / abs(np.log(S)); lo = S ** np.exp(1.96 * se_ll); hi = S ** np.exp(-1.96 * se_ll)
        else:
            lo = hi = S
        rows.append({"Time": u, "At risk": n_i, "Events": d_i, "Survival": S, "Std Err": se, "Lower 95%": lo, "Upper 95%": hi})
    return pd.DataFrame(rows)


def logrank(groups):
    """groups: list of (t, e). Returns chi2, df, p."""
    allt = np.concatenate([g[0] for g in groups]); alle = np.concatenate([g[1] for g in groups]); gid = np.concatenate([[i] * len(g[0]) for i, g in enumerate(groups)])
    k = len(groups); ut = np.unique(allt[alle == 1])
    O = np.zeros(k); E = np.zeros(k); V = np.zeros((k, k))
    for u in ut:
        at = allt >= u; n = at.sum(); d = ((allt == u) & (alle == 1)).sum()
        ni = np.array([(at & (gid == i)).sum() for i in range(k)]); di = np.array([((allt == u) & (alle == 1) & (gid == i)).sum() for i in range(k)])
        O += di; E += d * ni / n
        if n > 1:
            V += d * (n - d) / (n - 1) * (np.diag(ni / n) - np.outer(ni, ni) / n ** 2)
    O_E = (O - E)[:-1]; Vr = V[:-1, :-1]
    chi2 = float(O_E @ np.linalg.pinv(Vr) @ O_E)
    return chi2, k - 1, stats.chi2.sf(chi2, k - 1)


def life_distribution(df, params, L: Lang, outdir):
    """params: time, event (1=failure,0=right-censored; omit if all failures), distributions (default [weibull,lognormal,exponential]), group (optional), at_times (list for reliability at time), unit (label)."""
    d, tc = _data(df, params)
    dists = params.get("distributions", ["weibull", "lognormal", "exponential"]); unit = params.get("unit", "")
    t, e = d[tc].values, d["_e"].values
    if e.sum() < 2:
        raise ValueError("Need at least 2 failures to fit a life distribution.")
    res = Result(f"{L.t('Weibull')} / Life Distribution: {tc}")
    res.table(L.t("Data"), pd.DataFrame([{L.t("Observations"): len(t), L.t("Failures"): int(e.sum()), L.t("Censored"): int(len(t) - e.sum())}]))
    fits = {dn: fit_dist(dn, t, e) for dn in dists}
    cmp_ = pd.DataFrame([{L.t("Distribution"): dn, L.t("Log-Likelihood"): f["ll"], L.t("AICc"): f["aicc"], L.t("BIC"): f["bic"]} for dn, f in fits.items()]).sort_values(L.t("AICc"))
    res.table("Distribution comparison (lower AICc/BIC = better)", cmp_)
    best = cmp_.iloc[0][L.t("Distribution")]
    at = params.get("at_times", [])
    prm_rows, qrows = [], []
    for dn, f in fits.items():
        th, se = f["theta"], f["se"]
        if dn == "weibull":
            b, eta = np.exp(th), None
            prm_rows += [{L.t("Distribution"): dn, "Parameter": L.t("Shape (beta)"), L.t("Estimate"): np.exp(th[0]), "Lower 95%": np.exp(th[0] - 1.96 * se[0]), "Upper 95%": np.exp(th[0] + 1.96 * se[0])},
                         {L.t("Distribution"): dn, "Parameter": L.t("Scale (eta)"), L.t("Estimate"): np.exp(th[1]), "Lower 95%": np.exp(th[1] - 1.96 * se[1]), "Upper 95%": np.exp(th[1] + 1.96 * se[1])}]
        elif dn == "lognormal":
            prm_rows += [{L.t("Distribution"): dn, "Parameter": "mu (log scale)", L.t("Estimate"): th[0], "Lower 95%": th[0] - 1.96 * se[0], "Upper 95%": th[0] + 1.96 * se[0]},
                         {L.t("Distribution"): dn, "Parameter": "sigma", L.t("Estimate"): np.exp(th[1]), "Lower 95%": np.exp(th[1] - 1.96 * se[1]), "Upper 95%": np.exp(th[1] + 1.96 * se[1])}]
        else:
            prm_rows += [{L.t("Distribution"): dn, "Parameter": "lambda (rate)", L.t("Estimate"): np.exp(th[0]), "Lower 95%": np.exp(th[0] - 1.96 * se[0]), "Upper 95%": np.exp(th[0] + 1.96 * se[0])}]
        row = {L.t("Distribution"): dn, L.t("Mean life (MTTF)"): mttf(f), "B1": quantile(f, .01), "B5": quantile(f, .05), L.t("B10 life"): quantile(f, .10), L.t("Median"): quantile(f, .5), "B63.2": quantile(f, .632)}
        for tt in at: row[f"R({tt})"] = float(sf(f, tt))
        qrows.append(row)
    res.table("Parameters (MLE, Wald 95% CI)", pd.DataFrame(prm_rows))
    res.table("Life metrics", pd.DataFrame(qrows))
    # probability plot (Weibull) with Johnson adjusted ranks + survival + hazard
    order = np.argsort(t); ts, es = t[order], e[order]
    n = len(ts); rev = np.arange(n, 0, -1); inc = 0; prev = 0; rank = []
    Fv, Tv = [], []
    for i in range(n):
        if es[i] == 1:
            inc = (n + 1 - prev) / (1 + rev[i]); prev += inc; Fv.append((prev - .3) / (n + .4)); Tv.append(ts[i])
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.4))
    ax = axs[0]; Fv = np.array(Fv); Tv = np.array(Tv)
    ax.plot(np.log(Tv), np.log(-np.log(1 - Fv)), "o", color=PALETTE[0], ms=5)
    tt = np.linspace(Tv.min() * .7, Tv.max() * 1.3, 200)
    for i, (dn, f) in enumerate(fits.items()):
        if dn == "weibull":
            ax.plot(np.log(tt), np.log(-np.log(sf(f, tt))), color=PALETTE[1], lw=2, label="Weibull")
    ax.set_title(L.fig("Probability plot") + " (Weibull)"); ax.set_xlabel("ln(t)"); ax.set_ylabel("ln(-ln(1-F))"); ax.legend(fontsize=8)
    ax = axs[1]
    kmd = km(t, e)
    ax.step(np.r_[0, kmd["Time"]], np.r_[1, kmd["Survival"]], where="post", color="k", lw=1.5, label="Kaplan-Meier")
    for i, (dn, f) in enumerate(fits.items()):
        ax.plot(tt, sf(f, tt), color=PALETTE[i % 8], lw=2, label=dn)
    ax.set_title(L.fig("Survival")); ax.set_xlabel(f"{L.fig('Time')} {unit}"); ax.legend(fontsize=8)
    ax = axs[2]
    for i, (dn, f) in enumerate(fits.items()):
        h = 1e-4 * tt; pdf = (sf(f, tt) - sf(f, tt + h)) / h; ax.plot(tt, pdf / np.clip(sf(f, tt), 1e-12, None), color=PALETTE[i % 8], lw=2, label=dn)
    ax.set_title("Hazard rate"); ax.set_xlabel(f"{L.fig('Time')} {unit}"); ax.legend(fontsize=8)
    fig.tight_layout(); res.fig(L.t("Weibull"), save_fig(fig, outdir, f"life_{tc}"))
    w = fits.get("weibull")
    if w:
        b = float(np.exp(w["theta"][0]))
        phase = "infant mortality (decreasing hazard)" if b < .95 else ("random failures (~constant hazard)" if b < 1.05 else "wear-out (increasing hazard)")
        res.note(f"Weibull shape beta = {b:.2f}: {phase}. Maintenance/replacement policies only make sense when beta > 1.")
    res.note(f"Best distribution by AICc: {best}. A small sample (<~20 failures) gives wide confidence limits - inspect the intervals, not just the point estimates.")
    if e.sum() < 10: res.note("Fewer than 10 failures: MLE parameters/CI are unreliable (Wald intervals are only asymptotic).")
    res.summary = {"best": best, **{f"{dn}_loglik": f["ll"] for dn, f in fits.items()}}
    if w: res.summary.update({"weibull_beta": float(np.exp(w["theta"][0])), "weibull_eta": float(np.exp(w["theta"][1])), "B10": float(quantile(w, .1)), "MTTF": float(mttf(w))})
    return res


def survival(df, params, L: Lang, outdir):
    """Kaplan-Meier + log-rank.  params: time, event, group (optional), unit."""
    d, tc = _data(df, params); g = params.get("group"); unit = params.get("unit", "")
    res = Result(f"{L.t('Kaplan-Meier')}: {tc}")
    fig, ax = plt.subplots(figsize=(7.5, 5))
    groups = [("All", d)] if not g else [(str(k), v) for k, v in d.groupby(g)]
    summ = []
    data_g = []
    for i, (name, dd) in enumerate(groups):
        k_ = km(dd[tc].values, dd["_e"].values); data_g.append((dd[tc].values, dd["_e"].values))
        res.table(f"{L.t('Kaplan-Meier')} - {name}", k_)
        ax.step(np.r_[0, k_["Time"]], np.r_[1, k_["Survival"]], where="post", color=PALETTE[i % 8], lw=2, label=f"{name} (n={len(dd)})")
        ax.fill_between(np.r_[0, k_["Time"]], np.r_[1, k_["Lower 95%"]], np.r_[1, k_["Upper 95%"]], step="post", color=PALETTE[i % 8], alpha=.12)
        cens = dd[dd["_e"] == 0][tc].values
        if len(cens):
            ax.plot(cens, [np.interp(c, k_["Time"], k_["Survival"], left=1) for c in cens], "|", color=PALETTE[i % 8], ms=9)
        below = k_[k_["Survival"] <= .5]
        summ.append({L.t("Group"): name, "N": len(dd), L.t("Failures"): int(dd["_e"].sum()), L.t("Censored"): int((1 - dd["_e"]).sum()), "Median": below["Time"].iloc[0] if len(below) else np.nan,
                     "Mean (restricted)": float(np.trapezoid(np.r_[1, k_["Survival"]], np.r_[0, k_["Time"]]))})
    ax.set_title(L.fig("Kaplan-Meier")); ax.set_xlabel(f"{L.fig('Time')} {unit}"); ax.set_ylabel(L.fig("Survival")); ax.set_ylim(0, 1.02); ax.legend()
    res.fig(L.t("Kaplan-Meier"), save_fig(fig, outdir, f"km_{tc}"))
    res.table("Summary", pd.DataFrame(summ))
    if g and len(groups) > 1:
        chi2, dfree, p = logrank(data_g)
        res.table(L.t("Log-rank test"), pd.DataFrame([{"Chi2": chi2, "DF": dfree, "p-value": p}]))
        res.summary = {"logrank_chi2": chi2, "logrank_p": p}
        res.note("'|' ticks mark censored observations. Log-rank assumes proportional hazards - check that the curves do not cross.")
    return res
