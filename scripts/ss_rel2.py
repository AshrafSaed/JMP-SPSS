"""ss_rel2.py - Cox proportional hazards, repairable systems (Crow-AMSAA NHPP + MCF), accelerated life testing,
degradation, interval-censored life data, reliability test planning (success-run / Weibayes).
JMP: Fit Proportional Hazards, Reliability Growth / Recurrence Analysis, Fit Life by X, Degradation, Life Distribution (interval censoring), Reliability Test Plans.
SPSS: Cox Regression.
"""
from __future__ import annotations
import math
import numpy as np, pandas as pd
from scipy import stats, optimize
from scipy.special import gamma as G
import matplotlib, matplotlib.ticker
import matplotlib.pyplot as plt
from ss_common import HE, Result, Lang, save_fig, PALETTE
from ss_rel import _hess, km, fit_dist

HE.update({"Cox proportional hazards": "רגרסיית Cox (סיכונים פרופורציונליים)", "Repairable system (Crow-AMSAA)": "מערכת ניתנת לתיקון (Crow-AMSAA)",
           "Accelerated life test": "מבחן חיים מואץ", "Degradation analysis": "ניתוח דעיכה", "Interval-censored life data": "נתוני חיים עם צנזור קטעים",
           "Reliability test plan": "תכנון מבחן אמינות", "Hazard ratio": "יחס סיכונים", "Mean cumulative function": "פונקציית מצטברת ממוצעת"})


# ============================================================================
# Cox PH (Efron / Breslow)
# ============================================================================
def _cox_ll(beta, X, t, e, ties="efron", want_derivs=True):
    n, p = X.shape; eta = X @ beta; w = np.exp(eta - eta.max()); shift = eta.max()
    ll = 0.0; g = np.zeros(p); H = np.zeros((p, p))
    for u in np.unique(t[e == 1]):
        R = t >= u; D = (t == u) & (e == 1); d = int(D.sum())
        S0 = w[R].sum(); S1 = (w[R, None] * X[R]).sum(0); S2 = (X[R].T * w[R]) @ X[R]
        A0 = w[D].sum(); A1 = (w[D, None] * X[D]).sum(0); A2 = (X[D].T * w[D]) @ X[D]
        ll += eta[D].sum()
        g += X[D].sum(0)
        for l in range(d):
            f = (l / d) if ties == "efron" else 0.0
            ph = S0 - f * A0; p1 = S1 - f * A1; p2 = S2 - f * A2
            ll -= math.log(ph) + shift
            g -= p1 / ph
            H -= p2 / ph - np.outer(p1, p1) / ph ** 2
    return ll, g, H  # H is Hessian of loglik (negative definite)


def cox_fit(X, t, e, ties="efron", tol=1e-9, maxit=50):
    p = X.shape[1]; b = np.zeros(p); ll0, g0, H0 = _cox_ll(b, X, t, e, ties); ll = ll0
    for _ in range(maxit):
        ll, g, H = _cox_ll(b, X, t, e, ties); step = np.linalg.solve(-H, g); nb = b + step
        lln = _cox_ll(nb, X, t, e, ties)[0]; k = 0
        while lln < ll - 1e-12 and k < 30: step /= 2; nb = b + step; lln = _cox_ll(nb, X, t, e, ties)[0]; k += 1
        b = nb
        if np.max(np.abs(step)) < tol: break
    ll, g, H = _cox_ll(b, X, t, e, ties); cov = np.linalg.inv(-H)
    score0 = float(g0 @ np.linalg.solve(-H0, g0))
    return b, cov, ll, ll0, score0


def cox_ph(df, params, L: Lang, outdir):
    """time, event, covariates (list); categorical (list; default non-numeric); ties efron|breslow."""
    tc, ec, xs = params["time"], params["event"], params["covariates"]; d = df[[tc, ec] + xs].dropna(); d = d[d[tc] > 0]
    cat = params.get("categorical") or [c for c in xs if not pd.api.types.is_numeric_dtype(d[c])]; X = pd.get_dummies(d[xs], columns=cat, drop_first=True, dtype=float); t = d[tc].values.astype(float); e = d[ec].values.astype(int)
    ties = params.get("ties", "efron"); b, cov, ll, ll0, score = cox_fit(X.values, t, e, ties); se = np.sqrt(np.diag(cov)); z = b / se; p = len(b)
    res = Result(L.t("Cox proportional hazards"))
    res.table("Parameter estimates", pd.DataFrame({"Term": X.columns, "Coef": b, "Std Error": se, "z": z, "p-value": 2 * stats.norm.sf(abs(z)), "Hazard ratio": np.exp(b), "HR low 95%": np.exp(b - 1.96 * se), "HR high 95%": np.exp(b + 1.96 * se)}))
    lr = 2 * (ll - ll0); wald = float(b @ np.linalg.solve(cov, b))
    # concordance
    eta = X.values @ b; conc = disc = tie = 0
    for i in np.where(e == 1)[0]:
        m = t > t[i]; conc += (eta[m] < eta[i]).sum(); disc += (eta[m] > eta[i]).sum(); tie += (eta[m] == eta[i]).sum()
    cidx = (conc + .5 * tie) / max(conc + disc + tie, 1)
    res.table("Model tests", pd.DataFrame([{"Ties": ties, "n": len(d), "Events": int(e.sum()), "LogLik (model)": ll, "LogLik (null)": ll0, "LR chi-square": lr, "LR p": stats.chi2.sf(lr, p), "Wald chi-square": wald, "Wald p": stats.chi2.sf(wald, p), "Score chi-square": score, "Score p": stats.chi2.sf(score, p), "Concordance": cidx, "AIC": -2 * ll + 2 * p}]))
    # Schoenfeld residual check (unscaled; correlation with rank of event time)
    ev_t = np.unique(t[e == 1]); rows = []; sch = []
    w = np.exp(eta - eta.max())
    for u in ev_t:
        R = t >= u; D = np.where((t == u) & (e == 1))[0]; xbar = (w[R, None] * X.values[R]).sum(0) / w[R].sum()
        for i in D: sch.append((u, X.values[i] - xbar))
    if len(sch) > 5:
        tt = np.array([s[0] for s in sch]); S_ = np.array([s[1] for s in sch]); rk = stats.rankdata(tt)
        for j, nm in enumerate(X.columns):
            r_, pv = stats.spearmanr(rk, S_[:, j]) if S_[:, j].std() > 0 else (np.nan, np.nan); rows.append({"Term": nm, "Spearman r (Schoenfeld vs time rank)": r_, "p-value": pv, "PH assumption": "questionable" if pv < .05 else "no evidence against"})
        res.table("Proportional-hazards check (approximate)", pd.DataFrame(rows))
    # Breslow baseline
    H0 = np.array([(( (t == u) & (e == 1)).sum()) / w[t >= u].sum() for u in ev_t]); Hc = np.cumsum(H0)
    xm = X.values.mean(0); S0 = np.exp(-Hc * math.exp((xm @ b) - eta.max()))
    fig, axs = plt.subplots(1, 2, figsize=(11, 4)); yy = np.arange(len(b)); axs[0].errorbar(np.exp(b), yy, xerr=[np.exp(b) - np.exp(b - 1.96 * se), np.exp(b + 1.96 * se) - np.exp(b)], fmt="o", color=PALETTE[0], capsize=3); axs[0].axvline(1, color=PALETTE[3], ls="--"); axs[0].set_xscale("log"); axs[0].xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter()); axs[0].xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter()); axs[0].set_yticks(yy); axs[0].set_yticklabels(X.columns); axs[0].set_title(L.fig("Hazard ratio"))
    axs[1].step(np.concatenate([[0], ev_t]), np.concatenate([[1], S0]), where="post", color=PALETTE[0]); axs[1].set_title("Baseline survival at mean covariates"); axs[1].set_ylim(0, 1.02); res.fig("Cox", save_fig(fig, outdir, "cox"))
    res.summary.update({"coef": {k: float(v) for k, v in zip(X.columns, b)}, "hr": {k: float(math.exp(v)) for k, v in zip(X.columns, b)}, "score_chi2": score, "concordance": float(cidx)})
    res.note("Efron tie handling by default (as R/SPSS-compatible Breslow vs Efron may differ slightly). PH check is the simple unscaled-Schoenfeld rank correlation, not the full Grambsch-Therneau test.")
    return res


# ============================================================================
# Repairable systems
# ============================================================================
def repairable(df, params, L: Lang, outdir):
    """system (id column, optional); time (cumulative age at each failure); end (observation end per system: column name or number; default max time per system)."""
    tc = params["time"]; sc = params.get("system"); d = df[[tc] + ([sc] if sc else [])].dropna(); d["_s"] = d[sc].astype(str) if sc else "1"
    if isinstance(params.get("end"), dict): ends = {str(k): float(v) for k, v in params["end"].items()}
    elif isinstance(params.get("end"), (int, float)): ends = {s: float(params["end"]) for s in d["_s"].unique()}
    elif isinstance(params.get("end"), str): ends = df.groupby(df[sc].astype(str))[params["end"]].max().to_dict()
    else: ends = d.groupby("_s")[tc].max().to_dict()
    d = d[d[tc] > 0]; N = len(d); T = np.array([ends[s] for s in d["_s"]]); beta = N / np.sum(np.log(T / d[tc].values)); Ts = np.array(list(ends.values())); lam = N / np.sum(Ts ** beta)
    beta_u = beta * (N - 1) / N if len(Ts) == 1 else beta
    chi_lo, chi_hi = stats.chi2.ppf(.025, 2 * N), stats.chi2.ppf(.975, 2 * N); b_ci = (beta * chi_lo / (2 * N), beta * chi_hi / (2 * N))
    # Laplace trend test (combined systems)
    U = (np.sum(d[tc].values) / N - Ts[0] / 2) / (Ts[0] * math.sqrt(1 / (12 * N))) if len(Ts) == 1 else (np.sum(d[tc].values / T) - N / 2) / math.sqrt(N / 12)
    Tmax = float(Ts.max()); inst = 1 / (lam * beta * Tmax ** (beta - 1))
    res = Result(L.t("Repairable system (Crow-AMSAA)"))
    res.table("Power-law NHPP (MLE)", pd.DataFrame([{"Systems": len(Ts), "Failures N": N, "Beta": beta, "Beta (unbiased, 1 system)": beta_u, "Beta low 95%": b_ci[0], "Beta high 95%": b_ci[1], "Lambda": lam, "Instantaneous MTBF at end": inst, "Cumulative MTBF at end (per system)": 1 / (lam * Tmax ** (beta - 1)), "Laplace U": U, "Trend p-value": 2 * stats.norm.sf(abs(U)),
                                         "Interpretation": "reliability growth (beta<1)" if b_ci[1] < 1 else ("deterioration (beta>1)" if b_ci[0] > 1 else "no significant trend (beta~1)")}]))
    # MCF (Nelson)
    systems = list(ends); times = np.sort(d[tc].unique()); mcf = []; c = 0.0
    for u in times:
        at = sum(1 for s in systems if ends[s] >= u); dn = ((d[tc] == u)).sum(); c += dn / at; mcf.append((u, at, dn, c))
    mc = pd.DataFrame(mcf, columns=["Time", "At risk", "Events", "MCF"]); res.table("Mean cumulative function (Nelson)", mc)
    fig, axs = plt.subplots(1, 2, figsize=(11, 4)); axs[0].step(mc["Time"], mc["MCF"], where="post", color=PALETTE[0], label="MCF"); tt = np.linspace(0, Tmax, 100); axs[0].plot(tt, lam * tt ** beta / len(Ts), color=PALETTE[3], label="Crow-AMSAA (per system)"); axs[0].legend(); axs[0].set_xlabel("age"); axs[0].set_title(L.fig("Mean cumulative function"))
    axs[1].loglog(mc["Time"], mc["MCF"], "o-", ms=3); axs[1].set_title("log-log MCF (straight line => power law)"); res.fig("Crow-AMSAA", save_fig(fig, outdir, "repairable"))
    res.summary.update({"beta": float(beta), "lambda": float(lam), "laplace_U": float(U), "N": N}); res.note("Pooled-system estimates assume all systems are identical and start at age 0; cumulative MTBF shown for one-system scale. Use with caution for different system populations.")
    return res


# ============================================================================
# Accelerated life test
# ============================================================================
def _xfun(rel, s):
    s = np.asarray(s, float)
    if rel == "arrhenius": return 11604.5 / (s + 273.15)   # s in deg C ; slope b = Ea (eV)
    if rel == "power": return np.log(s)
    return s


def alt(df, params, L: Lang, outdir):
    """time, event, stress; relationship arrhenius (stress in C) | power (inverse power law, ln stress) | linear; dist weibull|lognormal; use_stress; B-life p (default 0.1)."""
    tc, ec, sc = params["time"], params["event"], params["stress"]; rel = params.get("relationship", "arrhenius"); dist = params.get("dist", "weibull"); d = df[[tc, ec, sc]].dropna(); d = d[d[tc] > 0]
    t, e, s = d[tc].values.astype(float), d[ec].values.astype(int), d[sc].values.astype(float); x = _xfun(rel, s); xm = x.mean()
    def nll(th):
        a, b, ls = th; sh = math.exp(ls); mu = a + b * (x - xm)
        if dist == "weibull":
            eta = np.exp(mu); z = (t / eta) ** sh; return -(np.sum(e * (ls - mu + (sh - 1) * (np.log(t) - mu))) - np.sum(z))
        z = (np.log(t) - mu) / sh; return -np.sum(e * (stats.norm.logpdf(z) - ls - np.log(t)) + (1 - e) * stats.norm.logsf(z))
    best = None
    for b0 in (0.5, -0.5, 1.0):
        x0 = np.array([np.log(np.median(t)), b0, 0.0]); r = optimize.minimize(nll, x0, method="Nelder-Mead", options={"xatol": 1e-9, "fatol": 1e-11, "maxiter": 8000})
        if best is None or r.fun < best.fun: best = r
    th = best.x; H = _hess(nll, th); cov = np.linalg.inv(H); se = np.sqrt(np.diag(cov)); a, b, ls = th; sh = math.exp(ls)
    names = ["Intercept (ln scale at mean x)", "Slope b (per unit x)" + (" = Ea [eV]" if rel == "arrhenius" else ""), "ln shape (sigma)"]
    res = Result(L.t("Accelerated life test"))
    res.table("Parameter estimates", pd.DataFrame({"Parameter": names, "Estimate": th, "Std Error": se, "z": th / se, "p-value": 2 * stats.norm.sf(abs(th / se))}))
    res.table("Model", pd.DataFrame([{"Distribution": dist, "Relationship": rel, "Shape (beta)" if dist == "weibull" else "Sigma": (sh if dist == "weibull" else sh), "-LogLik": best.fun, "AICc": 2 * best.fun + 6 + 24 / max(len(t) - 4, 1), "n": len(t), "Failures": int(e.sum())}]))
    out = {}; p = float(params.get("p", .1))
    if params.get("use_stress") is not None:
        xu = float(_xfun(rel, [params["use_stress"]])[0]); g = np.array([1.0, xu - xm, 0.0]); mu_u = a + b * (xu - xm); se_mu = math.sqrt(g @ cov @ g)
        if dist == "weibull": life = math.exp(mu_u) * (-math.log(1 - p)) ** (1 / sh); eta_u = math.exp(mu_u); bl = math.exp(mu_u + stats.norm.ppf(.05) * se_mu) * (-math.log(1 - p)) ** (1 / sh)
        else: life = math.exp(mu_u + sh * stats.norm.ppf(p)); eta_u = math.exp(mu_u); bl = math.exp(mu_u + sh * stats.norm.ppf(p) + stats.norm.ppf(.05) * se_mu)
        af = {st: math.exp(b * (xu - float(_xfun(rel, [st])[0]))) for st in sorted(set(s))}
        res.table("Use-condition prediction", pd.DataFrame([{"Use stress": params["use_stress"], f"B{int(p*100)} life (point)": life, f"B{int(p*100)} lower 95% (one-sided approx.)": bl, "Characteristic life at use": eta_u}]))
        res.table("Acceleration factors (stress -> use)", pd.DataFrame({"Test stress": list(af), "Acceleration factor": [1 / v for v in af.values()]}))
        out = {"life_use": life, "ea": b}
    fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.2))
    for i, st in enumerate(sorted(set(s))):
        m_ = s == st; tg, eg = t[m_], e[m_]; k = km(tg, eg)
        if len(k): axs[0].step(k["Time"], 1 - k["Survival"], where="post", color=PALETTE[i % 8], label=f"{st:g}")
    axs[0].set_xscale("log"); axs[0].set_xlabel("time"); axs[0].set_ylabel("F(t) (Kaplan-Meier)"); axs[0].legend(title="stress", fontsize=7); axs[0].set_title(L.fig("Accelerated life test"))
    xs = np.linspace(min(x.min(), xu if params.get("use_stress") is not None else x.min()), x.max(), 100); mus = a + b * (xs - xm); lifep = np.exp(mus) * (-math.log(1 - .5)) ** (1 / sh) if dist == "weibull" else np.exp(mus)
    axs[1].semilogy(xs, lifep, color=PALETTE[3]); axs[1].semilogy(x, t, ".", alpha=.5); axs[1].set_xlabel({"arrhenius": "11604.5/(T+273.15)", "power": "ln(stress)"}.get(rel, "stress")); axs[1].set_ylabel("life (median line)"); axs[1].set_title("Life vs stress"); res.fig("ALT", save_fig(fig, outdir, "alt"))
    res.summary.update({"slope": float(b), "slope_p": float(2 * stats.norm.sf(abs(b / se[1]))), "shape": float(sh), **out}); res.note("Single common shape across stress levels is assumed - compare per-stress fits (life_distribution with 'group') before relying on extrapolation. Extrapolation to use stress is a model assumption, not data.")
    return res


# ============================================================================
# Degradation
# ============================================================================
def degradation(df, params, L: Lang, outdir):
    """unit; time; measure; threshold; path linear|exponential (ln y linear); dist weibull|lognormal."""
    u, tc, mc, thr = params["unit"], params["time"], params["measure"], float(params["threshold"]); path = params.get("path", "linear"); dist = params.get("dist", "lognormal")
    d = df[[u, tc, mc]].dropna(); rows = []; tmax = d[tc].max()
    for k, g in d.groupby(u):
        yv = np.log(g[mc].values) if path == "exponential" else g[mc].values; thv = math.log(thr) if path == "exponential" else thr; sl, ic = np.polyfit(g[tc].values, yv, 1)
        Tf = (thv - ic) / sl if sl != 0 else np.inf
        ok = np.isfinite(Tf) and Tf > 0 and ((sl > 0) == (thv > ic))
        rows.append({"Unit": k, "Slope": sl, "Intercept": ic, "Pseudo failure time": Tf if ok else np.nan, "Last time": g[tc].max(), "R2": np.corrcoef(g[tc], yv)[0, 1] ** 2 if len(g) > 2 else np.nan})
    pf = pd.DataFrame(rows); res = Result(L.t("Degradation analysis")); res.table("Per-unit degradation paths", pf)
    tt = pf["Pseudo failure time"].values; cens = np.isnan(tt); tt2 = np.where(cens, pf["Last time"].values, tt); ev = (~cens).astype(int)
    if ev.sum() >= 2:
        f = fit_dist(dist, tt2, ev); th = f["theta"]; se = f["se"]
        if dist == "weibull": est = pd.DataFrame({"Parameter": ["Shape beta", "Scale eta"], "Estimate": np.exp(th), "Std Error (delta)": np.exp(th) * se})
        else: est = pd.DataFrame({"Parameter": ["mu (log mean)", "sigma"], "Estimate": [th[0], np.exp(th[1])], "Std Error (delta)": [se[0], np.exp(th[1]) * se[1]]})
        res.table(f"{dist.capitalize()} fit to pseudo failure times (units not reaching threshold are censored at last time)", est); res.summary["fit"] = {k: float(v) for k, v in zip(est["Parameter"], est["Estimate"])}
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for i, (k, g) in enumerate(d.groupby(u)): ax.plot(g[tc], g[mc], "o-", ms=3, color=PALETTE[i % 8], lw=1)
    ax.axhline(thr, color="k", ls="--"); ax.set_xlabel(tc); ax.set_ylabel(mc); ax.set_title(L.fig("Degradation analysis")); res.fig("degradation paths", save_fig(fig, outdir, "degradation"))
    res.summary["failed_by_extrapolation"] = int(ev.sum()); res.note("Pseudo failure times come from per-unit extrapolation (two-stage approximate method); extrapolating far beyond observed time is an assumption.")
    return res


# ============================================================================
# Interval-censored life data
# ============================================================================
def life_interval(df, params, L: Lang, outdir):
    """lower, upper (failure known to lie in (lower, upper]; lower==upper exact; upper empty/inf = right-censored; lower 0 = left-censored); freq (optional); dist weibull|lognormal|exponential."""
    lo, up = params["lower"], params["upper"]; d = df[[lo, up] + ([params["freq"]] if params.get("freq") else [])].copy(); d[up] = d[up].fillna(np.inf); d[lo] = d[lo].fillna(0); w = d[params["freq"]].values if params.get("freq") else np.ones(len(d)); l, u = d[lo].values.astype(float), d[up].values.astype(float)
    exact = (l == u); res = Result(L.t("Interval-censored life data")); rows = []; fits = {}
    for dist in params.get("distributions", ["weibull", "lognormal"]):
        def cdf(tv, th):
            tv = np.asarray(tv, float)
            if dist == "weibull": return 1 - np.exp(-(np.clip(tv, 0, None) / math.exp(th[1])) ** math.exp(th[0]))
            if dist == "lognormal": return np.where(tv > 0, stats.norm.cdf((np.log(np.clip(tv, 1e-300, None)) - th[0]) / math.exp(th[1])), 0.0)
            return 1 - np.exp(-math.exp(th[0]) * np.clip(tv, 0, None))
        def pdf(tv, th):
            if dist == "weibull": b, eta = math.exp(th[0]), math.exp(th[1]); return b / eta * (tv / eta) ** (b - 1) * np.exp(-(tv / eta) ** b)
            if dist == "lognormal": s_ = math.exp(th[1]); return stats.norm.pdf((np.log(tv) - th[0]) / s_) / (s_ * tv)
            lam = math.exp(th[0]); return lam * np.exp(-lam * tv)
        def nll(th):
            Fl = cdf(l, th); Fu = np.where(np.isfinite(u), cdf(np.where(np.isfinite(u), u, 1.0), th), 1.0)
            ex = np.log(np.clip(pdf(np.where(exact, l, 1.0), th), 1e-300, None)); iv = np.log(np.clip(Fu - Fl, 1e-300, None))
            return -(w * np.where(exact, ex, iv)).sum()
        x0 = {"weibull": [0.0, math.log(np.nanmedian(np.where(np.isfinite(u), (l + u) / 2, l)) + 1e-9)], "lognormal": [math.log(np.nanmedian(np.where(np.isfinite(u), (l + u) / 2, l)) + 1e-9), 0.0], "exponential": [-math.log(np.nanmedian(np.where(np.isfinite(u), (l + u) / 2, l)) + 1e-9)]}[dist]
        r = optimize.minimize(nll, np.array(x0), method="Nelder-Mead", options={"xatol": 1e-9, "fatol": 1e-11, "maxiter": 5000}); H = _hess(nll, r.x); se = np.sqrt(np.clip(np.diag(np.linalg.pinv(H)), 0, None)); k = len(r.x); n = w.sum()
        fits[dist] = r.x
        if dist == "weibull": ests = {"Shape beta": math.exp(r.x[0]), "Scale eta": math.exp(r.x[1])}
        elif dist == "lognormal": ests = {"mu": r.x[0], "sigma": math.exp(r.x[1])}
        else: ests = {"lambda": math.exp(r.x[0])}
        rows.append({"Distribution": dist, **ests, "-LogLik": r.fun, "AICc": 2 * r.fun + 2 * k + 2 * k * (k + 1) / max(n - k - 1, 1), "BIC": 2 * r.fun + k * math.log(n)})
    t = pd.DataFrame(rows); res.table("Fits (best AICc first)", t.sort_values("AICc")); res.summary["best"] = t.sort_values("AICc").iloc[0]["Distribution"]
    tg = np.linspace(max(l[l > 0].min() * .5 if (l > 0).any() else 0.01, 1e-9), (u[np.isfinite(u)].max() if np.isfinite(u).any() else l.max()) * 1.2, 200); fig, ax = plt.subplots(figsize=(6.5, 4))
    for i, (dn, th) in enumerate(fits.items()):
        if dn == "weibull": F = 1 - np.exp(-(tg / math.exp(th[1])) ** math.exp(th[0]))
        elif dn == "lognormal": F = stats.norm.cdf((np.log(tg) - th[0]) / math.exp(th[1]))
        else: F = 1 - np.exp(-math.exp(th[0]) * tg)
        ax.plot(tg, F, label=dn, color=PALETTE[i])
    ax.legend(); ax.set_xlabel("time"); ax.set_ylabel("F(t)"); ax.set_title(L.fig("Interval-censored life data")); res.fig("fits", save_fig(fig, outdir, "life_interval"))
    return res


# ============================================================================
# Reliability test planning
# ============================================================================
def reliability_plan(df, params, L: Lang, outdir):
    """No data.  reliability R, confidence C, allowed failures list; optional Weibull beta + mission time for Weibayes zero-failure test duration; optional n units."""
    R = float(params.get("reliability", .9)); C = float(params.get("confidence", .9)); res = Result(L.t("Reliability test plan")); rows = []
    for c in params.get("failures", [0, 1, 2, 3, 4, 5]):
        n = next(i for i in range(1, 100000) if stats.binom.cdf(c, i, 1 - R) <= 1 - C); rows.append({"Allowed failures c": c, "Required units n": n, "Check: P(accept | R)": float(stats.binom.cdf(c, n, 1 - R))})
    res.table(f"Success-run / binomial demonstration (R={R}, C={C}) - each unit tested for the mission time", pd.DataFrame(rows)); res.summary["n_zero_fail"] = rows[0]["Required units n"]
    if params.get("beta") and params.get("mission_time"):
        beta, tm = float(params["beta"]), float(params["mission_time"]); eta_req = tm / (-math.log(R)) ** (1 / beta); rows = []
        for n in params.get("units", [5, 10, 20, 30, 50]):
            for r_ in (0, 1, 2):
                # test time per unit (all survive except r_ failing at the same time approx): eta_L = (2*sum t^beta / chi2(C,2r+2))^(1/beta)
                tt = (eta_req ** beta * stats.chi2.ppf(C, 2 * r_ + 2) / (2 * n)) ** (1 / beta); rows.append({"Units n": n, "Failures allowed r": r_, "Test duration per unit": tt, "x mission time": tt / tm})
        res.table(f"Weibayes demonstration test (beta={beta}, mission {tm}, R={R}, C={C}; requires eta >= {eta_req:.6g})", pd.DataFrame(rows)); res.summary["eta_required"] = float(eta_req)
        res.note("Weibayes assumes the Weibull shape beta is known; a wrong beta changes the required test time as t ~ n^(-1/beta). Use a conservative (lower) beta if unsure.")
    fig, ax = plt.subplots(figsize=(6, 3.8)); cs = np.arange(0, 11)
    for Rv in (.8, .9, .95, .99): ax.plot(cs, [next(i for i in range(1, 100000) if stats.binom.cdf(c, i, 1 - Rv) <= 1 - C) for c in cs], "o-", ms=3, label=f"R={Rv}")
    ax.set_xlabel("allowed failures"); ax.set_ylabel("units required"); ax.legend(); ax.set_title(L.fig("Reliability test plan")); res.fig("sample sizes", save_fig(fig, outdir, "reliability_plan")); return res
