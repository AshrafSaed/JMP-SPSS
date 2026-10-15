"""ss_ext.py - Generalized linear mixed model (random intercept, adaptive Gauss-Hermite), REML Gage R&R (unbalanced + covariates),
split-plot analysis (whole-plot / sub-plot error strata with Satterthwaite df).
JMP: Generalized Linear Mixed Model, Fit Model REML with random effects, MSA / Variability chart; DOE split-plot analysis.
SPSS: GENLINMIXED, MIXED.   numpy / scipy only.
"""
from __future__ import annotations
import math
import numpy as np, pandas as pd
from scipy import stats, optimize
from scipy.special import gammaln, logsumexp
import matplotlib.pyplot as plt
from ss_common import HE, Result, Lang, save_fig, PALETTE
from ss_linear import OLS
from ss_models2 import REML, _Z, _hessian

HE.update({"Generalized linear mixed model": "מודל לינארי מוכלל מעורב (GLMM)", "Gage R&R (REML)": "Gage R&R (REML)", "Split-plot analysis": "ניתוח Split-plot"})


# ----------------------------------------------------------------------------
# GLMM: random intercept, binomial (logit) / Poisson (log), adaptive Gauss-Hermite quadrature (nAGQ = 1 is the Laplace approximation)
# ----------------------------------------------------------------------------
def _fam(fam):
    if fam == "binomial":
        def ll(eta, y, nn): return y * eta - nn * np.logaddexp(0, eta) + gammaln(nn + 1) - gammaln(y + 1) - gammaln(nn - y + 1)
        def d12(eta, y, nn): mu = 1 / (1 + np.exp(-eta)); return y - nn * mu, nn * mu * (1 - mu)
    elif fam == "poisson":
        def ll(eta, y, nn): return y * eta - np.exp(eta) - gammaln(y + 1)
        def d12(eta, y, nn): mu = np.exp(eta); return y - mu, mu
    else: raise ValueError("family must be binomial|poisson")
    return ll, d12


def glmm_nll(theta, X, y, nn, g, G, fam, nagq=15):
    """Negative marginal log-likelihood; theta = [beta, log sigma_u]."""
    ll, d12 = _fam(fam); p = X.shape[1]; beta = theta[:p]; sig = math.exp(theta[p]); eta0 = X @ beta; u = np.zeros(G); s2 = sig * sig
    for _ in range(40):                                                    # inner Newton for the conditional modes
        l1, l2 = d12(eta0 + u[g], y, nn); grad = np.bincount(g, l1, G) - u / s2; hess = np.bincount(g, l2, G) + 1 / s2
        step = np.clip(grad / hess, -3, 3); u = u + step
        if np.max(np.abs(step)) < 1e-10: break
    _, l2 = d12(eta0 + u[g], y, nn); sc = 1 / np.sqrt(np.bincount(g, l2, G) + 1 / s2)
    x, w = np.polynomial.hermite.hermgauss(nagq); const = -0.5 * math.log(2 * math.pi * s2); terms = np.empty((nagq, G))
    for k in range(nagq):
        uk = u + math.sqrt(2) * sc * x[k]; h = np.bincount(g, ll(eta0 + uk[g], y, nn), G) - uk * uk / (2 * s2) + const
        terms[k] = math.log(w[k]) + x[k] * x[k] + 0.5 * math.log(2) + np.log(sc) + h
    return -float(logsumexp(terms, axis=0).sum())


def glmm(df, params, L: Lang, outdir):
    """response; family binomial|poisson; fixed (list); categorical (list, default non-numeric); group (random-intercept column); trials (column, binomial counts; default 0/1 response);
    positive (level coded 1, binomial 0/1 text response); degree (fixed interaction order, default 1); nagq (quadrature points, default 15; 1 = Laplace)."""
    y_, fam = params["response"], params.get("family", "binomial"); fixed = list(params.get("fixed") or []); gc = params["group"]; tr = params.get("trials"); nagq = int(params.get("nagq", 15))
    d = df[[y_, gc] + fixed + ([tr] if tr else [])].dropna().reset_index(drop=True)
    if fam == "binomial" and not tr:
        pos = params.get("positive")
        if pos is None and not set(pd.unique(d[y_])) <= {0, 1}: pos = sorted(d[y_].unique(), key=str)[-1]
        yv = (d[y_].astype(str) == str(pos)).astype(float).values if pos is not None else d[y_].values.astype(float); nn = np.ones(len(d))
    else: yv = d[y_].values.astype(float); nn = d[tr].values.astype(float) if tr else np.ones(len(d))
    d["_y"] = yv
    cat = params.get("categorical") or [c for c in fixed if not pd.api.types.is_numeric_dtype(d[c])]; cov = [c for c in fixed if c not in cat]
    o = OLS(d, "_y", factors=cat, covariates=cov, degree=int(params.get("degree", 1))); X, names = o.X, o.names
    glab = d[gc].astype(str); lv = sorted(glab.unique()); g = glab.map({l: i for i, l in enumerate(lv)}).values; G = len(lv); p = X.shape[1]
    ll, d12 = _fam(fam)
    m0 = float(np.clip(yv.sum() / nn.sum(), 1e-3, 1 - 1e-3)) if fam == "binomial" else max(yv.mean(), 1e-3)
    x0 = np.zeros(p + 1); x0[0] = math.log(m0 / (1 - m0)) if fam == "binomial" else math.log(m0)
    if not o.has_int: x0[0] = 0.0
    x0[p] = math.log(.5); f = lambda th: glmm_nll(th, X, yv, nn, g, G, fam, nagq)
    best = None
    for s0 in (math.log(.5), math.log(1.5), math.log(.1)):
        x0[p] = s0; r = optimize.minimize(f, x0, method="L-BFGS-B", bounds=[(None, None)] * p + [(-7, 3)]); best = r if best is None or r.fun < best.fun else best
    th = best.x; nll = float(best.fun); sig = math.exp(th[p])
    H = _hessian(f, th, 1e-4);
    try: C = np.linalg.inv(H)
    except np.linalg.LinAlgError: C = np.full((p + 1, p + 1), np.nan)
    se = np.sqrt(np.where(np.diag(C) > 0, np.diag(C), np.nan)); b = th[:p]; z = b / se[:p]
    # null model without random effect (sigma -> 0): ordinary GLM likelihood
    def nll0(bb): return -float(ll(X @ bb, yv, nn).sum())
    r0 = optimize.minimize(nll0, x0[:p], method="BFGS"); lr = max(2 * (r0.fun - nll), 0.0); p_lr = 0.5 * stats.chi2.sf(lr, 1)
    res = Result(L.t("Generalized linear mixed model"))
    eff = "Odds Ratio" if fam == "binomial" else "Rate Ratio"
    res.table("Fixed effects", pd.DataFrame({"Term": names, "Estimate": b, "Std Error": se[:p], "z": z, "p-value": 2 * stats.norm.sf(np.abs(z)), eff: np.exp(b), "Lower 95%": np.exp(b - 1.96 * se[:p]), "Upper 95%": np.exp(b + 1.96 * se[:p])}))
    sd_se = sig * se[p]; icc = sig ** 2 / (sig ** 2 + math.pi ** 2 / 3) if fam == "binomial" else np.nan
    res.table("Random intercept", pd.DataFrame([{"Group": gc, "Groups": G, "Std Dev (sigma_u)": sig, "Std Error (delta)": sd_se, "Variance": sig ** 2, "Latent-scale ICC": icc}]))
    k = p + 1; res.table("Fit", pd.DataFrame([{"-2 log-lik (marginal)": 2 * nll, "AIC": 2 * nll + 2 * k, "BIC": 2 * nll + k * math.log(len(yv)), "n": len(yv), "Quadrature points": nagq, "LR vs no random effect": lr, "p (boundary mixture 0.5*chi2_1)": p_lr}]))
    # conditional modes (BLUP-like) for the caterpillar plot
    eta0 = X @ b; u = np.zeros(G); s2 = sig ** 2
    for _ in range(50):
        l1, l2 = d12(eta0 + u[g], yv, nn); step = (np.bincount(g, l1, G) - u / s2) / (np.bincount(g, l2, G) + 1 / s2); u += step
    _, l2 = d12(eta0 + u[g], yv, nn); usd = 1 / np.sqrt(np.bincount(g, l2, G) + 1 / s2); oo = np.argsort(u)
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2)); axs[0].errorbar(u[oo], range(G), xerr=1.96 * usd[oo], fmt="o", ms=3, color=PALETTE[0], lw=.8); axs[0].axvline(0, color=PALETTE[3], ls="--"); axs[0].set_yticks(range(G) if G <= 40 else []); axs[0].set_yticklabels(np.array(lv)[oo] if G <= 40 else [], fontsize=6); axs[0].set_title(f"{gc}: conditional modes +/- 1.96 SD")
    mu_c = (1 / (1 + np.exp(-(eta0 + u[g]))) if fam == "binomial" else np.exp(eta0 + u[g])); gm_obs = pd.Series(yv / nn).groupby(g).mean().values; gm_fit = pd.Series(mu_c).groupby(g).mean().values
    axs[1].scatter(gm_fit, gm_obs, s=18, color=PALETTE[1]); lim = [min(gm_fit.min(), gm_obs.min()), max(gm_fit.max(), gm_obs.max())]; axs[1].plot(lim, lim, "--", color="#888"); axs[1].set_xlabel("fitted (conditional) group mean"); axs[1].set_ylabel("observed group mean"); axs[1].set_title("Observed vs fitted by group")
    res.fig("GLMM", save_fig(fig, outdir, "glmm"))
    res.summary.update({"beta": {n: float(v) for n, v in zip(names, b)}, "se": {n: float(v) for n, v in zip(names, se[:p])}, "sigma_u": sig, "neg_loglik": nll, "lr_p": float(p_lr)})
    res.note(f"Random-intercept GLMM by maximum marginal likelihood, adaptive Gauss-Hermite quadrature with {nagq} points (1 = Laplace). Only ONE random intercept (no random slopes or crossed effects). Wald z-tests are large-sample (no small-sample df correction); the LR test for sigma_u uses the 0.5*chi-square(1) boundary distribution.")
    return res


# ----------------------------------------------------------------------------
# Gage R&R by REML (handles unbalanced data and fixed covariates)
# ----------------------------------------------------------------------------
def gage_rr_reml(df, params, L: Lang, outdir):
    """part, operator, measurement; covariates (optional fixed effects, e.g. temperature); categorical (list); tolerance (USL-LSL); study_sigma (6)."""
    pc, oc, mc = params["part"], params["operator"], params["measurement"]; cov = list(params.get("covariates") or []); K = float(params.get("study_sigma", 6)); tol = params.get("tolerance")
    d = df[[pc, oc, mc] + cov].dropna().reset_index(drop=True); d["_P"] = d[pc].astype(str); d["_O"] = d[oc].astype(str); yv = d[mc].values.astype(float)
    if cov:
        cat = params.get("categorical") or [c for c in cov if not pd.api.types.is_numeric_dtype(d[c])]; o = OLS(d, mc, factors=cat, covariates=[c for c in cov if c not in cat]); X, names = o.X, o.names
    else: X, names = np.ones((len(d), 1)), ["Intercept"]
    ZP, lp = _Z(d, "_P"); ZO, lo = _Z(d, "_O"); ZI, li = _Z(d, "_P:_O"); m = REML(yv, X, [ZP, ZO, ZI])
    v_part, v_op, v_int, v_rep = m.v; v_repro = v_op + v_int; v_grr = v_rep + v_repro; v_tot = v_grr + v_part; sd = math.sqrt
    comps = [("Gage R&R", v_grr), ("Repeatability", v_rep), ("Reproducibility", v_repro), ("  Operator", v_op), ("  Operator*Part", v_int), ("Part-to-Part", v_part), ("Total", v_tot)]
    rows = [{"Source": nm, "Variance Component": v, "% of Total Variance": 100 * v / v_tot, "StdDev": sd(v), "Study Var": K * sd(v), "% Study Var": 100 * sd(v) / sd(v_tot), "% Tolerance": (100 * K * sd(v) / tol) if tol else np.nan} for nm, v in comps]
    ndc = int(math.floor(1.41 * sd(v_part) / sd(v_grr))) if v_grr > 0 else None; pct_sv = 100 * sd(v_grr) / sd(v_tot); pct_tol = 100 * K * sd(v_grr) / tol if tol else None
    crit = pct_tol if pct_tol is not None else pct_sv
    verdict = "ACCEPTABLE (<10%)" if crit < 10 else ("MARGINAL (10-30%)" if crit <= 30 else "UNACCEPTABLE (>30%)")
    res = Result(f"{L.t('Gage R&R (REML)')}: {mc}")
    res.table("Variance components (REML)", pd.DataFrame([{"Random effect": n, "Levels": lv_, "Variance component": s, "Std Error (approx.)": se_} for n, lv_, s, se_ in zip(["Part", "Operator", "Part*Operator", "Repeatability (residual)"], [len(lp), len(lo), len(li), len(d)], m.v, m.se_v)]))
    res.table("Gage R&R", pd.DataFrame(rows)); res.table("Number of Distinct Categories", pd.DataFrame([{"ndc": ndc, "Verdict": verdict, "Observations": len(d), "Parts": len(lp), "Operators": len(lo)}]))
    if cov: res.table("Fixed effects (covariates)", pd.DataFrame({"Term": names, "Estimate": m.beta, "Std Error": np.sqrt(np.diag(m.cov_b))}))
    fig, axs = plt.subplots(1, 3, figsize=(14, 4)); lab = ["Gage R&R", "Repeat.", "Reprod.", "Part-to-Part"]; vals = [v_grr, v_rep, v_repro, v_part]; xs_ = np.arange(4); w_ = .27
    axs[0].bar(xs_ - w_ / 2, [100 * v / v_tot for v in vals], w_, color=PALETTE[0], label="% Contribution"); axs[0].bar(xs_ + w_ / 2, [100 * sd(v) / sd(v_tot) for v in vals], w_, color=PALETTE[1], label="% Study Var"); axs[0].set_xticks(xs_); axs[0].set_xticklabels(lab, fontsize=8); axs[0].legend(fontsize=7); axs[0].set_title("Components of variation")
    opsl = sorted(d["_O"].unique()); parts = sorted(d["_P"].unique())
    for j, op in enumerate(opsl):
        dd = d[d["_O"] == op]; axs[1].plot([parts.index(a) + (j - (len(opsl) - 1) / 2) * .15 for a in dd["_P"]], dd[mc], "o", ms=3, color=PALETTE[j % 8], alpha=.7, label=op)
    axs[1].set_xticks(range(len(parts))); axs[1].set_xticklabels(parts, fontsize=6, rotation=45); axs[1].legend(fontsize=7); axs[1].set_title(f"{mc} by part / operator")
    axs[2].scatter(m.fitted_cond, yv - m.fitted_cond, s=12); axs[2].axhline(0, color=PALETTE[3]); axs[2].set_title(L.fig("Residual"))
    fig.tight_layout(); res.fig("Gage R&R REML", save_fig(fig, outdir, f"grr_reml_{mc}"))
    res.summary = {"pct_study_var": pct_sv, "pct_tolerance": pct_tol, "pct_contribution": 100 * v_grr / v_tot, "ndc": ndc, "verdict": verdict, "var": {"part": v_part, "operator": v_op, "interaction": v_int, "repeatability": v_rep}}
    res.note("Crossed Part x Operator model fitted by REML (random Part, Operator, Part*Operator): works with unbalanced data and adjusts for fixed covariates. The interaction is always kept (no AIAG alpha=0.25 pooling), so values can differ slightly from the ANOVA gage_rr. Guidelines (<10 % acceptable, 10-30 % marginal, ndc >= 5) are conventions, not a standard.")
    return res


# ----------------------------------------------------------------------------
# Split-plot analysis (REML, whole-plot random effect; Satterthwaite df identify the right error stratum)
# ----------------------------------------------------------------------------
def split_plot_analyze(df, params, L: Lang, outdir):
    """response; whole_plot (list of hard-to-change factor columns); subplot (list of easy-to-change factor columns); plot (whole-plot ID column);
    block (optional random block column); degree (interaction order, default 2); categorical (list; default non-numeric)."""
    from ss_models2 import mixed
    y = params["response"]; wp = list(params["whole_plot"]); sp = list(params.get("subplot") or []); pid = params["plot"]; blk = params.get("block")
    rand = ([blk] if blk else []) + ([f"{blk}:{pid}"] if blk else [pid])
    res = mixed(df, {"response": y, "fixed": wp + sp, "random": rand, "degree": int(params.get("degree", 2)), "categorical": params.get("categorical")}, L, outdir)
    res.title = L.t("Split-plot analysis") + f": {y}"
    stratum = []
    ft = [t for n, t in res.tables if n.startswith("Fixed-effect tests")][0]
    for _, r in ft.iterrows():
        parts = r["Source"].split("*"); st = "whole plot" if all(p in wp for p in parts) else "sub plot"
        stratum.append({"Effect": r["Source"], "Stratum": st, "Num DF": r["Num DF"], "Den DF (Satterthwaite)": r.iloc[2], "F": r["F"], "p-value": r["p-value"]})
    res.table("Effects by error stratum", pd.DataFrame(stratum))
    res.note("Whole-plot effects (all factors hard-to-change) are tested against whole-plot-to-whole-plot variation; effects involving a sub-plot factor against the residual. The Satterthwaite denominator df show which stratum each test used. Few whole plots => low power for whole-plot effects.")
    return res
