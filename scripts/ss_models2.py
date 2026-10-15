"""ss_models2.py - Mixed models, nested variance components, factor analysis, discriminant, ordinal/multinomial logit,
nonlinear curve fitting, penalized regression, tree/forest/boosting, PLS, time series.
JMP: Fit Model (REML/Random effects), Factor Analysis, Discriminant, Fit Curve/Nonlinear, Generalized Regression, Partition, Bootstrap Forest, PLS, Time Series.
SPSS: MIXED, Factor, Discriminant, PLUM/NOMREG, Curve Estimation/NLR, Forecasting.
numpy / scipy / sklearn only.
"""
from __future__ import annotations
import itertools, math
import numpy as np, pandas as pd
from scipy import stats, optimize
import matplotlib.pyplot as plt
from ss_common import HE, Result, Lang, save_fig, PALETTE, numeric_cols
from ss_linear import OLS, build_terms

HE.update({
    "Mixed model (REML)": "מודל מעורב (REML)", "Variance components": "רכיבי שונות", "Nested ANOVA": "ANOVA מקונן",
    "Factor analysis": "ניתוח גורמים", "Discriminant analysis": "ניתוח דיסקרימיננטי", "Ordinal logistic regression": "רגרסיה לוגיסטית סדרתית",
    "Multinomial logistic regression": "רגרסיה לוגיסטית רב-נומינלית", "Nonlinear curve fit": "התאמת עקומה לא-לינארית",
    "Penalized regression": "רגרסיה מוענשת", "Predictive model": "מודל חיזוי", "PLS regression": "רגרסיית PLS", "Time series": "סדרות עתיות",
    "Forecast": "תחזית", "Autocorrelation": "אוטוקורלציה", "Scree plot": "גרף Scree", "Loadings": "עמסות", "Communality": "קומיוניליטי",
    "Variance component": "רכיב שונות", "% of total": "% מהכלל", "Random effect": "אפקט אקראי", "Fixed effects": "אפקטים קבועים",
})


# ============================================================================
# Mixed model - REML for variance-components models
# ============================================================================
def _expand_random(spec):
    out = []
    for s in spec:
        if "/" in s:
            parts = s.split("/")
            for i in range(1, len(parts) + 1): out.append(":".join(parts[:i]))
        else: out.append(s)
    seen = []
    for o in out:
        if o not in seen: seen.append(o)
    return seen


def _Z(d, term):
    lab = d[term.split(":")].astype(str).agg("|".join, axis=1)
    lv = sorted(lab.unique()); Z = (np.asarray(lab, dtype=object)[:, None] == np.array(lv, dtype=object)[None, :]).astype(float); return Z, lv


class REML:
    def __init__(self, y, X, Zs):
        self.y, self.X, self.Zs = y, X, Zs; self.n, self.p = X.shape
        self.K = [Z @ Z.T for Z in Zs]
        self.fit()

    def _V(self, v):
        V = v[-1] * np.eye(self.n)
        for s, K in zip(v[:-1], self.K): V = V + s * K
        return V

    def nll(self, v):
        V = self._V(v)
        try: Lc = np.linalg.cholesky(V)
        except np.linalg.LinAlgError: return 1e12
        Vi_X = np.linalg.solve(V, self.X); Vi_y = np.linalg.solve(V, self.y)
        XtVX = self.X.T @ Vi_X; b = np.linalg.solve(XtVX, self.X.T @ Vi_y); r = self.y - self.X @ b
        return 0.5 * (2 * np.log(np.diag(Lc)).sum() + np.linalg.slogdet(XtVX)[1] + r @ np.linalg.solve(V, r))

    def fit(self):
        yv = float(np.var(self.y, ddof=1)) or 1.0; k = len(self.Zs)
        best = None
        for frac in (0.5, 0.1, 0.9):
            x0 = np.array([yv * frac / max(k, 1)] * k + [yv * (1 - frac) + 1e-9])
            r = optimize.minimize(self.nll, x0, method="L-BFGS-B", bounds=[(0, None)] * k + [(1e-10 * yv + 1e-12, None)], options={"maxiter": 500})
            if best is None or r.fun < best.fun: best = r
        self.v = best.x; self.neg_reml = best.fun
        V = self._V(self.v); Vi = np.linalg.inv(V); XtVX = self.X.T @ Vi @ self.X
        self.cov_b = np.linalg.inv(XtVX); self.beta = self.cov_b @ self.X.T @ Vi @ self.y
        self.resid_marg = self.y - self.X @ self.beta
        self.u = [s * Z.T @ Vi @ self.resid_marg for s, Z in zip(self.v[:-1], self.Zs)]
        self.fitted_cond = self.X @ self.beta + sum(Z @ u for Z, u in zip(self.Zs, self.u)) if self.Zs else self.X @ self.beta
        # SE of variance components from numeric Hessian (nonboundary only)
        h = 1e-4 * np.maximum(self.v, 1e-3 * yv); m = len(self.v); H = np.zeros((m, m)); f0 = self.nll(self.v)
        for i in range(m):
            for j in range(i, m):
                def f(di, dj):
                    w = self.v.copy(); w[i] += di; w[j] += dj; return self.nll(w)
                H[i, j] = H[j, i] = (f(h[i], h[j]) - f(h[i], -h[j]) - f(-h[i], h[j]) + f(-h[i], -h[j])) / (4 * h[i] * h[j])
        try:
            C = np.linalg.inv(H); se = np.sqrt(np.where(np.diag(C) > 0, np.diag(C), np.nan))
        except np.linalg.LinAlgError: se = np.full(m, np.nan)
        se[:-1][self.v[:-1] < 1e-8 * yv] = np.nan; self.se_v = se
        self.H, self.hstep, self.yv = H, h, yv
        free = np.ones(m, bool); free[:-1] = self.v[:-1] > 1e-8 * yv; self.free = free      # components on the boundary are held fixed
        try: A = np.zeros((m, m)); A[np.ix_(free, free)] = np.linalg.inv(H[np.ix_(free, free)]); self.A = A
        except np.linalg.LinAlgError: self.A = None
        self.cov_v = self.A

    def cov_beta_at(self, v):
        V = self._V(v); Vi = np.linalg.inv(V); return np.linalg.inv(self.X.T @ Vi @ self.X)

    def _sat_one(self, l):
        """Satterthwaite df for the variance of one linear function l'beta: nu = 2 var^2 / (g' A g), g = d var / d v (numerical)."""
        var = float(l @ self.cov_b @ l)
        if self.A is None or var <= 0: return var, np.nan
        g = np.zeros(len(self.v))
        for k in np.where(self.free)[0]:
            hk = self.hstep[k]; vp, vm = self.v.copy(), self.v.copy(); vp[k] += hk; vm[k] = max(vm[k] - hk, 1e-12)
            g[k] = (float(l @ self.cov_beta_at(vp) @ l) - float(l @ self.cov_beta_at(vm) @ l)) / (vp[k] - vm[k])
        den = float(g @ self.A @ g)
        return var, (2 * var * var / den if den > 1e-300 else np.inf)

    def contrast_test(self, Lm):
        """F test of H0: Lm beta = 0 with Satterthwaite denominator df (Fai-Cornelius style for >1 df).
        Returns F, num df, den df, p."""
        Lm = np.atleast_2d(np.asarray(Lm, float)); q = np.linalg.matrix_rank(Lm); Lb = Lm @ self.beta; S = Lm @ self.cov_b @ Lm.T
        if q == 1 and Lm.shape[0] == 1:
            var, nu = self._sat_one(Lm[0]); F = float(Lb[0] ** 2 / var); return F, 1, nu, (stats.f.sf(F, 1, nu) if np.isfinite(nu) else stats.chi2.sf(F, 1))
        D, P = np.linalg.eigh(S); keep = D > 1e-12 * D.max(); D, P = D[keep], P[:, keep]; q = len(D)
        t2 = 0.0; nus = []
        for mth in range(q):
            l = P[:, mth] @ Lm; _, nu = self._sat_one(l); nus.append(nu); t2 += float((l @ self.beta) ** 2 / D[mth])
        F = t2 / q; nus = np.array(nus)
        if not np.all(np.isfinite(nus)): nu = np.inf if np.all(np.isinf(nus)) else float(np.nanmin(nus))
        elif np.all(nus > 2): E = float(np.sum(nus / (nus - 2))); nu = 2 * E / (E - q) if E > q else float(nus.min())
        else: nu = float(nus.min())
        return F, q, nu, (stats.f.sf(F, q, nu) if np.isfinite(nu) else stats.chi2.sf(F * q, q))


def mixed(df, params, L: Lang, outdir):
    """response; fixed (list of columns); categorical (list, default non-numeric); random (list, e.g. ['Batch','Batch/Sample'] or ['A:B']); degree (fixed interactions order, default 1)."""
    y = params["response"]; fixed = params.get("fixed") or []; rand = _expand_random(params.get("random") or [])
    rcols = sorted({c for t in rand for c in t.split(":")}); d = df[[y] + fixed + rcols].dropna().reset_index(drop=True)
    cat = params.get("categorical") or [c for c in fixed if not pd.api.types.is_numeric_dtype(d[c])]
    cov = [c for c in fixed if c not in cat]
    if fixed:
        o = OLS(d, y, factors=cat, covariates=cov, degree=int(params.get("degree", 1))); X, names = o.X, o.names
    else:
        X, names = np.ones((len(d), 1)), ["Intercept"]
    yv = d[y].values.astype(float); Zs, lvs = zip(*[_Z(d, t) for t in rand]) if rand else ((), ())
    m = REML(yv, X, list(Zs)); res = Result(L.t("Mixed model (REML)"))
    tot = m.v.sum()
    vrows = [{"Random effect": t, "Levels": len(l), "Variance component": s, "Std Dev": math.sqrt(s), "% of total": 100 * s / tot, "Std Error (approx.)": se}
             for t, l, s, se in zip(rand, lvs, m.v[:-1], m.se_v[:-1])]
    vrows.append({"Random effect": "Residual", "Levels": len(d), "Variance component": m.v[-1], "Std Dev": math.sqrt(m.v[-1]), "% of total": 100 * m.v[-1] / tot, "Std Error (approx.)": m.se_v[-1]})
    vrows.append({"Random effect": "Total", "Levels": len(d), "Variance component": tot, "Std Dev": math.sqrt(tot), "% of total": 100.0, "Std Error (approx.)": np.nan})
    res.table("Variance components (REML)", pd.DataFrame(vrows))
    se = np.sqrt(np.diag(m.cov_b)); dfm_ = (params.get("df_method") or "satterthwaite").lower(); sat = dfm_ != "wald"
    dfs = np.array([m._sat_one(np.eye(len(m.beta))[i])[1] for i in range(len(m.beta))]) if sat else np.full(len(m.beta), np.inf)
    tv = m.beta / se; pv = np.array([2 * (stats.t.sf(abs(a), b) if np.isfinite(b) else stats.norm.sf(abs(a))) for a, b in zip(tv, dfs)])
    res.table("Fixed effects (t, Satterthwaite df)" if sat else "Fixed effects (Wald z)", pd.DataFrame({"Term": names, "Estimate": m.beta, "Std Error": se, "DF": dfs, "t": tv, "p-value": pv}))
    if fixed:
        rows = []
        for t, idx in o.term_idx.items():
            Lm = np.zeros((len(idx), len(m.beta))); Lm[np.arange(len(idx)), idx] = 1
            F, q_, nu, p_ = m.contrast_test(Lm) if sat else (None, len(idx), np.inf, None)
            b = m.beta[idx]; C = m.cov_b[np.ix_(idx, idx)]; w = float(b @ np.linalg.solve(C, b))
            if not sat: F = w / len(idx); p_ = stats.chi2.sf(w, len(idx))
            rows.append({"Source": "*".join(t), "Num DF": q_, "Den DF (Satterthwaite)" if sat else "Den DF": nu, "F": F, "p-value": p_, "Wald chi-square (large-sample)": w, "Wald p": stats.chi2.sf(w, len(idx))})
        res.table("Fixed-effect tests (F, Satterthwaite df)" if sat else "Fixed-effect tests (Wald chi-square)", pd.DataFrame(rows)); res.summary["fixed_tests"] = {r["Source"]: {"F": float(r["F"]), "df2": float(r[[k for k in r if k.startswith("Den DF")][0]]), "p": float(r["p-value"])} for r in rows}
    k_par = len(m.v) + X.shape[1]
    res.table("Fit", pd.DataFrame([{"-2 REML log-lik": 2 * m.neg_reml, "n": len(d), "Fixed params": X.shape[1], "Variance params": len(m.v), "AIC (REML, comparable only for same fixed effects)": 2 * m.neg_reml + 2 * len(m.v)}]))
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.8))
    axs[0].bar([r["Random effect"] for r in vrows[:-1]], [r["% of total"] for r in vrows[:-1]], color=PALETTE[0]); axs[0].set_ylabel("% of total"); axs[0].set_title(L.fig("Variance components")); plt.setp(axs[0].get_xticklabels(), rotation=25, ha="right")
    axs[1].scatter(m.fitted_cond, yv - m.fitted_cond, s=14); axs[1].axhline(0, color=PALETTE[3]); axs[1].set_title(L.fig("Residual"))
    if rand:
        u = m.u[0]; o2 = np.argsort(u); axs[2].plot(u[o2], range(len(u)), "o", ms=3); axs[2].set_yticks([]); axs[2].set_title(f"BLUP: {rand[0]}")
    res.fig("mixed model diagnostics", save_fig(fig, outdir, "mixed"))
    res.summary.update({"variance_components": {t: float(s) for t, s in zip(rand + ["Residual"], m.v)}, "neg2_reml": float(2 * m.neg_reml)})
    res.note("Variance components by REML (boundary at zero allowed). Fixed-effect t/F tests use Satterthwaite denominator df (multi-df terms: Fai-Cornelius approximation); Kenward-Roger's small-sample covariance adjustment is NOT applied, so with very few random-effect levels p-values can be slightly optimistic. Set df_method='wald' for large-sample Wald tests.")
    return res


def nested_anova(df, params, L: Lang, outdir):
    """response; factors: hierarchical list outer->inner, e.g. ['Plant','Batch','Sample'] (Batch nested in Plant ...). Variance components by REML + classical SS table."""
    y = params["response"]; fs = params["factors"]; d = df[[y] + fs].dropna().reset_index(drop=True); yv = d[y].values.astype(float)
    res = Result(L.t("Nested ANOVA")); rows = []; prev = None; gm = yv.mean(); keys = []
    for i in range(len(fs)):
        key = d[fs[:i + 1]].astype(str).agg("|".join, axis=1); keys.append(key)
    prev_mean = np.full(len(d), gm); prev_df = 0
    for i, (f, key) in enumerate(zip(fs, keys)):
        cm = d.groupby(key)[y].transform("mean").values; ss = float(((cm - prev_mean) ** 2).sum()); ng = key.nunique()
        dfi = ng - (keys[i - 1].nunique() if i else 1); rows.append({"Source": f if i == 0 else f"{f}[{fs[i-1]}]", "DF": dfi, "Sum of Squares": ss, "Mean Square": ss / dfi if dfi else np.nan}); prev_mean = cm
    sse = float(((yv - prev_mean) ** 2).sum()); dfe = len(d) - keys[-1].nunique(); rows.append({"Source": "Error (within last level)", "DF": dfe, "Sum of Squares": sse, "Mean Square": sse / dfe if dfe else np.nan})
    t = pd.DataFrame(rows); mse = t["Mean Square"]
    F = [mse[i] / mse[i + 1] for i in range(len(rows) - 1)] + [np.nan]
    t["F Ratio (vs next level)"] = F; t["Prob > F"] = [stats.f.sf(F[i], t["DF"][i], t["DF"][i + 1]) if t["DF"][i] and t["DF"][i + 1] else np.nan for i in range(len(rows) - 1)] + [np.nan]
    res.table("Nested ANOVA table (sequential SS)", t)
    sub = mixed(d, {"response": y, "random": ["/".join(fs)]}, L, outdir)
    for nm, tb in sub.tables[:1]: res.table("Variance components (REML)", tb)
    for cap, p in sub.figures: res.fig(cap, p)
    res.summary = sub.summary; res.note("F ratios test each level against the next-inner level (random nested factors).")
    return res


# ============================================================================
# Factor analysis
# ============================================================================
def _varimax_once(Lm, R, gamma, q, tol):
    p, k = Lm.shape; d = 0
    for _ in range(q):
        Lr = Lm @ R; u, s, vt = np.linalg.svd(Lm.T @ (Lr ** 3 - gamma / p * Lr @ np.diag((Lr ** 2).sum(0)))); R = u @ vt; d_old, d = d, s.sum()
        if d_old != 0 and d - d_old < tol * max(1.0, d): break
    return R


def varimax(Lm, gamma=1.0, q=2000, tol=1e-12, starts=12):
    """Varimax from the identity plus several random orthogonal starts (the criterion has saddle points); keeps the best."""
    p, k = Lm.shape; rng = np.random.default_rng(0); best = None
    crit = lambda A: float(((A ** 2) ** 2).sum() - gamma / p * ((A ** 2).sum(0) ** 2).sum())
    for i in range(starts if k > 1 else 1):
        R0 = np.eye(k) if i == 0 else np.linalg.qr(rng.normal(size=(k, k)))[0]; R = _varimax_once(Lm, R0, gamma, q, tol); c = crit(Lm @ R)
        if best is None or c > best[0] + 1e-12: best = (c, R)
    return Lm @ best[1], best[1]


def factor_analysis(df, params, L: Lang, outdir):
    """columns; n_factors (default Kaiser eigenvalue>1); method pc|ml; rotation varimax|none."""
    cols = params.get("columns") or numeric_cols(df); d = df[cols].dropna(); n, p = d.shape; Z = (d - d.mean()) / d.std(ddof=1); R = np.corrcoef(Z.values.T)
    ev, evec = np.linalg.eigh(R); o = np.argsort(-ev); ev, evec = ev[o], evec[:, o]
    k = int(params.get("n_factors") or max(1, int((ev > 1).sum())))
    Rinv = np.linalg.inv(R); Dm = np.diag(1 / np.sqrt(np.diag(Rinv))); A = -Dm @ Rinv @ Dm; np.fill_diagonal(A, 0); r2 = R - np.eye(p); np.fill_diagonal(r2, 0)
    kmo_all = (r2 ** 2).sum() / ((r2 ** 2).sum() + (A ** 2).sum()); msa = (r2 ** 2).sum(0) / ((r2 ** 2).sum(0) + (A ** 2).sum(0))
    chi = -(n - 1 - (2 * p + 5) / 6) * np.log(np.linalg.det(R)); bdf = p * (p - 1) / 2
    method = params.get("method", "pc")
    if method == "ml":
        from sklearn.decomposition import FactorAnalysis
        fa = FactorAnalysis(n_components=k).fit(Z.values); load = fa.components_.T
    else:
        load = evec[:, :k] * np.sqrt(ev[:k])
    if params.get("rotation", "varimax") == "varimax" and k > 1: load, _ = varimax(load)
    comm = (load ** 2).sum(1); ss = (load ** 2).sum(0)
    res = Result(L.t("Factor analysis"))
    res.table("KMO & Bartlett", pd.DataFrame([{"KMO (overall)": kmo_all, "Bartlett chi-square": chi, "DF": bdf, "p-value": stats.chi2.sf(chi, bdf), "n": n, "Factors": k, "Method": method}]))
    res.table("Eigenvalues", pd.DataFrame({"Component": range(1, p + 1), "Eigenvalue": ev, "% variance": 100 * ev / p, "Cumulative %": 100 * np.cumsum(ev) / p}))
    res.table("Loadings (rotated)" if params.get("rotation", "varimax") == "varimax" else "Loadings", pd.DataFrame(load, index=cols, columns=[f"Factor{i+1}" for i in range(k)]).assign(**{"Communality": comm, "KMO (MSA)": msa}), index=True)
    res.table("Variance explained by factors", pd.DataFrame({"Factor": [f"Factor{i+1}" for i in range(k)], "SS loadings": ss, "% variance": 100 * ss / p}))
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.9)); axs[0].plot(range(1, p + 1), ev, "o-"); axs[0].axhline(1, color=PALETTE[3], ls="--"); axs[0].set_title(L.fig("Scree plot"))
    im = axs[1].imshow(load, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto"); axs[1].set_yticks(range(p)); axs[1].set_yticklabels(cols); axs[1].set_xticks(range(k)); axs[1].set_xticklabels([f"F{i+1}" for i in range(k)])
    for i in range(p):
        for j in range(k): axs[1].text(j, i, f"{load[i,j]:.2f}", ha="center", va="center", fontsize=7)
    axs[1].grid(False); axs[1].set_title(L.fig("Loadings")); res.fig("scree & loadings", save_fig(fig, outdir, "factor_analysis"))
    res.summary.update({"kmo": float(kmo_all), "factors": k, "eigenvalues": [float(e) for e in ev]})
    res.note("Kaiser eigenvalue>1 and KMO>0.6 are conventional rules of thumb, not standards.")
    return res


# ============================================================================
# Discriminant analysis
# ============================================================================
def discriminant(df, params, L: Lang, outdir):
    """response (class); predictors; method lda|qda; cv: 'loo' or k."""
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA, QuadraticDiscriminantAnalysis as QDA
    from sklearn.model_selection import cross_val_predict, LeaveOneOut, StratifiedKFold
    y, xs = params["response"], params.get("predictors") or [c for c in numeric_cols(df) if c != params["response"]]
    d = df[[y] + xs].dropna(); X = d[xs].values; cls = d[y].astype(str).values; lv = sorted(set(cls))
    mdl = QDA() if params.get("method", "lda") == "qda" else LDA(solver="svd"); mdl.fit(X, cls)
    cvp = params.get("cv", "loo"); cv = LeaveOneOut() if cvp == "loo" else StratifiedKFold(int(cvp), shuffle=True, random_state=1)
    pred_cv = cross_val_predict(type(mdl)() if params.get("method", "lda") == "qda" else LDA(solver="svd"), X, cls, cv=cv); pred = mdl.predict(X)
    res = Result(L.t("Discriminant analysis"))
    res.table("Classification (resubstitution)", pd.crosstab(pd.Series(cls, name="Actual"), pd.Series(pred, name="Predicted")), index=True)
    res.table("Classification (cross-validated)", pd.crosstab(pd.Series(cls, name="Actual"), pd.Series(pred_cv, name="Predicted")), index=True)
    res.table("Accuracy", pd.DataFrame([{"Resubstitution": (pred == cls).mean(), "Cross-validated": (pred_cv == cls).mean(), "Chance (largest class)": pd.Series(cls).value_counts(normalize=True).max(), "n": len(d)}]))
    res.table("Group means", d.groupby(y)[xs].mean().reset_index()); res.table("Prior probabilities", pd.DataFrame({"Class": mdl.classes_, "Prior": mdl.priors_}))
    if params.get("method", "lda") == "lda":
        res.table("Linear discriminant coefficients", pd.DataFrame(mdl.scalings_[:, :min(len(lv) - 1, len(xs))], index=xs, columns=[f"LD{i+1}" for i in range(min(len(lv) - 1, len(xs)))]), index=True)
        res.table("Proportion of trace", pd.DataFrame({"LD": [f"LD{i+1}" for i in range(len(mdl.explained_variance_ratio_))], "Proportion": mdl.explained_variance_ratio_}))
        S = mdl.transform(X); fig, ax = plt.subplots(figsize=(5.8, 4.6))
        for i, l in enumerate(lv): m_ = cls == l; ax.scatter(S[m_, 0], S[m_, 1] if S.shape[1] > 1 else np.zeros(m_.sum()), s=18, color=PALETTE[i % 8], label=l, alpha=.8)
        ax.legend(fontsize=7); ax.set_xlabel("LD1"); ax.set_ylabel("LD2"); ax.set_title(L.fig("Discriminant analysis")); res.fig("canonical scores", save_fig(fig, outdir, "discriminant"))
    res.summary.update({"acc_resub": float((pred == cls).mean()), "acc_cv": float((pred_cv == cls).mean())})
    return res


# ============================================================================
# Ordinal & multinomial logit
# ============================================================================
def _hessian(f, x, eps=1e-4):
    n = len(x); H = np.zeros((n, n)); h = eps * np.maximum(1, np.abs(x))
    for i in range(n):
        for j in range(i, n):
            def g(a, b):
                z = x.copy(); z[i] += a; z[j] += b; return f(z)
            H[i, j] = H[j, i] = (g(h[i], h[j]) - g(h[i], -h[j]) - g(-h[i], h[j]) + g(-h[i], -h[j])) / (4 * h[i] * h[j])
    return H


def _design(d, xs, cat):
    X = pd.get_dummies(d[xs], columns=[c for c in cat if c in xs], drop_first=True, dtype=float); return X


def ordinal(df, params, L: Lang, outdir):
    """response (ordered categories; give 'order' list low->high); predictors; categorical (list)."""
    y, xs = params["response"], params["predictors"]; d = df[[y] + xs].dropna(); order = [str(o) for o in (params.get("order") or sorted(d[y].astype(str).unique()))]
    cat = params.get("categorical") or [c for c in xs if not pd.api.types.is_numeric_dtype(d[c])]; X = _design(d, xs, cat); cats = d[y].astype(str).map({o: i for i, o in enumerate(order)}).values.astype(int)
    K = len(order); Xv = X.values; n, p = Xv.shape
    def unpack(w): th = np.cumsum(np.concatenate([[w[0]], np.exp(w[1:K - 1])])); return th, w[K - 1:]
    def nll(w):
        th, b = unpack(w); eta = Xv @ b; ub = np.concatenate([th, [np.inf]])[cats]; lb = np.concatenate([[-np.inf], th])[cats]
        pr = stats.logistic.cdf(ub - eta) - stats.logistic.cdf(lb - eta); return -np.log(np.clip(pr, 1e-300, None)).sum()
    cum = np.cumsum(np.bincount(cats, minlength=K))[:-1] / n; th0 = np.log(cum / (1 - cum)); w0 = np.concatenate([[th0[0]], np.log(np.clip(np.diff(th0), 1e-3, None)), np.zeros(p)])
    r = optimize.minimize(nll, w0, method="BFGS", options={"gtol": 1e-8, "maxiter": 2000}); w = r.x
    H = _hessian(nll, w); C = np.linalg.pinv(H)
    # delta method for thresholds & betas
    def theta_beta(w): th, b = unpack(w); return np.concatenate([th, b])
    J = np.array([(theta_beta(w + e) - theta_beta(w - e)) / (2e-6) for e in (np.eye(len(w)) * 1e-6)]).T; Cp = J @ C @ J.T; est = theta_beta(w); se = np.sqrt(np.clip(np.diag(Cp), 0, None))
    names = [f"Intercept {order[i]}|{order[i+1]}" for i in range(K - 1)] + list(X.columns); z = est / se
    tab = pd.DataFrame({"Term": names, "Estimate": est, "Std Error": se, "z": z, "p-value": 2 * stats.norm.sf(abs(z))})
    tab["Odds ratio"] = np.where(np.arange(len(est)) >= K - 1, np.exp(est), np.nan)
    nll0 = -(np.bincount(cats, minlength=K) * np.log(np.bincount(cats, minlength=K) / n)).sum(); lr = 2 * (nll0 - r.fun)
    th, b = unpack(w); eta = Xv @ b; cdf = stats.logistic.cdf(th[None, :] - eta[:, None]); P = np.diff(np.hstack([np.zeros((n, 1)), cdf, np.ones((n, 1))]), axis=1); pred = P.argmax(1)
    res = Result(L.t("Ordinal logistic regression"))
    res.table("Parameter estimates (proportional odds: logit P(Y<=j) = theta_j - x*beta)", tab)
    res.table("Model fit", pd.DataFrame([{"-LogLikelihood": r.fun, "LR chi-square vs null": lr, "DF": p, "p-value": stats.chi2.sf(lr, p) if p else np.nan, "McFadden R2": 1 - r.fun / nll0, "AIC": 2 * r.fun + 2 * len(w), "n": n}]))
    res.table("Classification table", pd.crosstab(pd.Series([order[i] for i in cats], name="Actual"), pd.Series([order[i] for i in pred], name="Predicted")), index=True)
    fig, ax = plt.subplots(figsize=(6.2, 3.8)); ax.bar(range(K), np.bincount(cats, minlength=K) / n, alpha=.5, label="observed"); ax.plot(range(K), P.mean(0), "o-", color=PALETTE[3], label="mean predicted"); ax.set_xticks(range(K)); ax.set_xticklabels(order); ax.legend(); ax.set_title(L.fig("Ordinal logistic regression"))
    res.fig("category fit", save_fig(fig, outdir, "ordinal")); res.summary.update({"coef": {k: float(v) for k, v in zip(names, est)}, "loglik": float(-r.fun)})
    res.note("Proportional-odds (parallel lines) assumption is not tested here; check by comparing with the multinomial model.")
    return res


def multinomial(df, params, L: Lang, outdir):
    """response (nominal); predictors; categorical; reference (optional level)."""
    from sklearn.linear_model import LogisticRegression
    y, xs = params["response"], params["predictors"]; d = df[[y] + xs].dropna(); cat = params.get("categorical") or [c for c in xs if not pd.api.types.is_numeric_dtype(d[c])]
    X = _design(d, xs, cat); yy = d[y].astype(str).values; lv = sorted(set(yy)); ref = str(params.get("reference", lv[-1])); others = [l for l in lv if l != ref]
    Xv = np.column_stack([np.ones(len(d)), X.values]); n, p = Xv.shape
    m = LogisticRegression(penalty=None, max_iter=5000, tol=1e-10).fit(X.values, yy); cl = list(m.classes_)
    B = {l: np.concatenate([[m.intercept_[cl.index(l)] - m.intercept_[cl.index(ref)]], m.coef_[cl.index(l)] - m.coef_[cl.index(ref)]]) for l in others}
    Bm = np.column_stack([B[l] for l in others]); Eta = np.column_stack([Xv @ Bm, np.zeros(n)]); P = np.exp(Eta - Eta.max(1, keepdims=True)); P /= P.sum(1, keepdims=True)
    I = np.zeros((p * len(others), p * len(others)))
    for a in range(len(others)):
        for b in range(len(others)):
            w = (P[:, a] * (a == b) - P[:, a] * P[:, b]); I[a * p:(a + 1) * p, b * p:(b + 1) * p] = (Xv * w[:, None]).T @ Xv
    C = np.linalg.pinv(I); se = np.sqrt(np.diag(C)); names = ["Intercept"] + list(X.columns); rows = []
    for a, l in enumerate(others):
        for j, nm in enumerate(names):
            e = B[l][j]; s = se[a * p + j]; rows.append({"Class vs ref": f"{l} vs {ref}", "Term": nm, "Estimate": e, "Std Error": s, "z": e / s, "p-value": 2 * stats.norm.sf(abs(e / s)), "Odds ratio": math.exp(e) if nm != "Intercept" else np.nan})
    ll = np.log(P[np.arange(n), [ (others + [ref]).index(c) for c in yy]]).sum(); cnt = pd.Series(yy).value_counts(); ll0 = (cnt * np.log(cnt / n)).sum(); lr = 2 * (ll - ll0)
    pred = np.array((others + [ref]))[P.argmax(1)]
    res = Result(L.t("Multinomial logistic regression")); res.table("Parameter estimates", pd.DataFrame(rows))
    res.table("Model fit", pd.DataFrame([{"LogLikelihood": ll, "LR chi-square vs null": lr, "DF": p * len(others) - len(others), "p-value": stats.chi2.sf(lr, (p - 1) * len(others)), "McFadden R2": 1 - ll / ll0, "AIC": -2 * ll + 2 * p * len(others), "Accuracy": (pred == yy).mean()}]))
    res.table("Classification table", pd.crosstab(pd.Series(yy, name="Actual"), pd.Series(pred, name="Predicted")), index=True)
    xcol = X.columns[0]; o_ = np.argsort(X[xcol].values); fig, ax = plt.subplots(figsize=(6.2, 3.8)); ax.stackplot(X[xcol].values[o_], P[o_].T, labels=others + [ref], colors=PALETTE[:len(others) + 1], alpha=.8); ax.set_xlabel(xcol); ax.set_ylabel("predicted probability"); ax.legend(fontsize=7); ax.set_title(L.fig("Multinomial logistic regression")); res.fig("class probabilities", save_fig(fig, outdir, "multinomial"))
    res.summary.update({"loglik": float(ll), "accuracy": float((pred == yy).mean())}); return res


# ============================================================================
# Nonlinear curve fitting
# ============================================================================
MODELS = {
    "exponential": (lambda x, a, b: a * np.exp(b * x), ["a", "b"], lambda x, y: [max(y[0], 1e-3) if y[0] != 0 else 1.0, 0.1 / max(np.ptp(x), 1e-9)]),
    "exp_decay": (lambda x, a, b, c: a + b * np.exp(-c * x), ["a", "b", "c"], lambda x, y: [y[-1], y[0] - y[-1], 3 / max(np.ptp(x), 1e-9)]),
    "power": (lambda x, a, b: a * np.power(x, b), ["a", "b"], lambda x, y: [np.mean(y), 1.0]),
    "logistic3": (lambda x, a, b, c: a / (1 + np.exp(-b * (x - c))), ["a", "b", "c"], lambda x, y: [np.max(y), 4 / max(np.ptp(x), 1e-9), np.median(x)]),
    "logistic4": (lambda x, a, b, c, d: d + (a - d) / (1 + np.power(x / c, b)), ["a (min)", "b (slope)", "c (EC50)", "d (max)"], lambda x, y: [np.min(y), 1.0, np.median(x), np.max(y)]),
    "michaelis_menten": (lambda x, vmax, km: vmax * x / (km + x), ["Vmax", "Km"], lambda x, y: [np.max(y), np.median(x)]),
    "gompertz": (lambda x, a, b, c: a * np.exp(-b * np.exp(-c * x)), ["a", "b", "c"], lambda x, y: [np.max(y) * 1.1, 2.0, 3 / max(np.ptp(x), 1e-9)]),
    "weibull_growth": (lambda x, a, b, c, d: a - b * np.exp(-c * np.power(x, d)), ["a", "b", "c", "d"], lambda x, y: [np.max(y) * 1.05, np.max(y) - np.min(y), 0.1, 1.0]),
}


def nonlinear(df, params, L: Lang, outdir):
    """x, y, model in {exponential, exp_decay, power, logistic3, logistic4, michaelis_menten, gompertz, weibull_growth, custom}; custom: expr (e.g. 'a*np.exp(-b*x)+c'), names ['a','b','c'], p0 [..]."""
    xn, yn = params["x"], params["y"]; d = df[[xn, yn]].dropna().sort_values(xn); x, y = d[xn].values.astype(float), d[yn].values.astype(float); mn = params.get("model", "exp_decay")
    if mn == "custom":
        names = params["names"]; expr = params["expr"]; ns = {"np": np, "__builtins__": {}}
        f = eval(f"lambda x,{','.join(names)}: {expr}", ns); p0 = params["p0"]
    else:
        f, names, auto = MODELS[mn]; p0 = params.get("p0") or auto(x, y)
    popt, pcov = optimize.curve_fit(f, x, y, p0=p0, maxfev=20000)
    if not np.all(np.isfinite(pcov)): pcov = np.full((len(popt), len(popt)), np.nan)
    n, k = len(x), len(popt); resid = y - f(x, *popt); sse = float(resid @ resid); dfe = n - k; mse = sse / dfe; se = np.sqrt(np.diag(pcov)); tc = stats.t.ppf(.975, dfe)
    res = Result(L.t("Nonlinear curve fit"))
    res.table("Parameter estimates", pd.DataFrame({"Parameter": names, "Estimate": popt, "Std Error": se, "t Ratio": popt / se, "Prob > |t|": 2 * stats.t.sf(abs(popt / se), dfe), "Lower 95%": popt - tc * se, "Upper 95%": popt + tc * se}))
    tss = float(((y - y.mean()) ** 2).sum())
    res.table("Fit statistics", pd.DataFrame([{"Model": mn, "n": n, "SSE": sse, "MSE": mse, "RMSE": math.sqrt(mse), "R2": 1 - sse / tss, "AIC": n * math.log(sse / n) + 2 * (k + 1), "AICc": n * math.log(sse / n) + 2 * (k + 1) + 2 * (k + 1) * (k + 2) / max(n - k - 2, 1)}]))
    xs = np.linspace(x.min(), x.max(), 300); ys = f(xs, *popt); J = np.array([(f(xs, *(popt + e)) - f(xs, *(popt - e))) / (2 * e[i]) for i, e in enumerate(np.eye(k) * np.maximum(1e-6, 1e-6 * np.abs(popt)))]).T
    band = tc * np.sqrt(np.clip(np.einsum("ij,jk,ik->i", J, np.nan_to_num(pcov), J), 0, None))
    fig, axs = plt.subplots(1, 2, figsize=(11, 4)); axs[0].plot(x, y, "o", ms=4); axs[0].plot(xs, ys, color=PALETTE[3]); axs[0].fill_between(xs, ys - band, ys + band, color=PALETTE[3], alpha=.2); axs[0].set_xlabel(xn); axs[0].set_ylabel(yn); axs[0].set_title(L.fig("Nonlinear curve fit"))
    axs[1].scatter(f(x, *popt), resid, s=14); axs[1].axhline(0, color=PALETTE[3]); axs[1].set_title(L.fig("Residual")); res.fig("fit & residuals", save_fig(fig, outdir, f"nonlinear_{mn}"))
    res.summary.update({"params": {a: float(b) for a, b in zip(names, popt)}, "r2": float(1 - sse / tss), "rmse": math.sqrt(mse)}); res.note("Confidence band/SEs are asymptotic (Jacobian-based); verify convergence from several starting values (p0).")
    return res


# ============================================================================
# Penalized regression
# ============================================================================
def penalized(df, params, L: Lang, outdir):
    """response; predictors; method ridge|lasso|enet; l1_ratio (enet); cv folds."""
    from sklearn.linear_model import RidgeCV, LassoCV, ElasticNetCV, lasso_path, enet_path
    y, xs = params["response"], params.get("predictors") or [c for c in numeric_cols(df) if c != params["response"]]; d = df[[y] + xs].dropna(); X = d[xs].values; yv = d[y].values
    mu, sd = X.mean(0), X.std(0, ddof=1); Xs = (X - mu) / sd; cv = int(params.get("cv", 10)); meth = params.get("method", "lasso")
    if meth == "ridge": m = RidgeCV(alphas=np.logspace(-4, 3, 80), cv=cv).fit(Xs, yv); lam = m.alpha_
    elif meth == "enet": m = ElasticNetCV(l1_ratio=float(params.get("l1_ratio", 0.5)), cv=cv, alphas=100, random_state=1).fit(Xs, yv); lam = m.alpha_
    else: m = LassoCV(cv=cv, alphas=100, random_state=1).fit(Xs, yv); lam = m.alpha_
    b = m.coef_ / sd; b0 = float(m.intercept_ - (mu * b).sum()); pred = m.predict(Xs); r2 = 1 - ((yv - pred) ** 2).sum() / ((yv - yv.mean()) ** 2).sum()
    res = Result(L.t("Penalized regression"))
    res.table("Coefficients (original scale)", pd.DataFrame({"Term": ["Intercept"] + xs, "Estimate": [b0] + list(b), "Standardized": [np.nan] + list(m.coef_), "Selected": [True] + [abs(c) > 1e-10 for c in m.coef_]}))
    cvm = np.nan
    if meth != "ridge": cvm = float(np.min(m.mse_path_.mean(1)))
    else:
        cvm = float(-m.best_score_) if getattr(m, "best_score_", None) is not None else np.nan
    res.table("Fit", pd.DataFrame([{"Method": meth, "Lambda (alpha) chosen by CV": lam, "R2 (training)": r2, "RMSE (training)": math.sqrt(((yv - pred) ** 2).mean()), "CV MSE (min)" if meth != "ridge" else "CV score": cvm, "Nonzero coefs": int((abs(m.coef_) > 1e-10).sum()), "n": len(d)}]))
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.9))
    if meth == "ridge":
        al = np.logspace(-4, 3, 60); coefs = np.array([np.linalg.solve(Xs.T @ Xs + a_ * np.eye(Xs.shape[1]), Xs.T @ (yv - yv.mean())) for a_ in al]); axs[0].semilogx(al, coefs)
    else:
        al, coefs, _ = (lasso_path(Xs, yv) if meth == "lasso" else enet_path(Xs, yv, l1_ratio=float(params.get("l1_ratio", 0.5)))); axs[0].semilogx(al, coefs.T)
    axs[0].axvline(lam, color="k", ls="--"); axs[0].set_title("Coefficient path"); axs[0].set_xlabel("lambda")
    axs[1].scatter(yv, pred, s=14); axs[1].plot([yv.min(), yv.max()], [yv.min(), yv.max()], color=PALETTE[3]); axs[1].set_xlabel("Actual"); axs[1].set_ylabel("Predicted"); axs[1].set_title(L.fig("Penalized regression")); res.fig("path & fit", save_fig(fig, outdir, "penalized"))
    res.summary.update({"lambda": float(lam), "r2": float(r2)}); return res


# ============================================================================
# Trees, forests, boosting (JMP Partition / Bootstrap Forest / Boosted Tree)
# ============================================================================
def predictive(df, params, L: Lang, outdir):
    """response; predictors; method tree|forest|boost; task auto|regression|classification; cv folds."""
    from sklearn import tree as sktree
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier
    from sklearn.model_selection import cross_val_score, cross_val_predict, StratifiedKFold, KFold
    from sklearn.inspection import permutation_importance
    y, xs = params["response"], params.get("predictors") or [c for c in df.columns if c != params["response"]]; d = df[[y] + xs].dropna()
    X = pd.get_dummies(d[xs], drop_first=True, dtype=float); yv = d[y]; task = params.get("task", "auto")
    if task == "auto": task = "regression" if pd.api.types.is_numeric_dtype(yv) and yv.nunique() > 10 else "classification"
    meth = params.get("method", "forest"); seed = 1
    if task == "regression":
        mk = {"tree": lambda: sktree.DecisionTreeRegressor(max_depth=int(params.get("max_depth", 4)), min_samples_leaf=int(params.get("min_leaf", 5)), random_state=seed), "forest": lambda: RandomForestRegressor(300, random_state=seed, n_jobs=1), "boost": lambda: GradientBoostingRegressor(random_state=seed)}[meth]
        cv = KFold(int(params.get("cv", 5)), shuffle=True, random_state=seed); yy = yv.values.astype(float); sc = "r2"
    else:
        mk = {"tree": lambda: sktree.DecisionTreeClassifier(max_depth=int(params.get("max_depth", 4)), min_samples_leaf=int(params.get("min_leaf", 5)), random_state=seed), "forest": lambda: RandomForestClassifier(300, random_state=seed, n_jobs=1), "boost": lambda: GradientBoostingClassifier(random_state=seed)}[meth]
        yy = yv.astype(str).values; cv = StratifiedKFold(int(params.get("cv", 5)), shuffle=True, random_state=seed); sc = "accuracy"
    m = mk().fit(X.values, yy); cvs = cross_val_score(mk(), X.values, yy, cv=cv, scoring=sc); pcv = cross_val_predict(mk(), X.values, yy, cv=cv)
    pi = permutation_importance(m, X.values, yy, n_repeats=10, random_state=seed)
    res = Result(L.t("Predictive model"))
    res.table("Cross-validated performance", pd.DataFrame([{"Method": meth, "Task": task, "Metric": sc, "CV mean": cvs.mean(), "CV SD": cvs.std(ddof=1), "Training score": m.score(X.values, yy), "n": len(d), "Folds": cv.get_n_splits()}]))
    imp = pd.DataFrame({"Predictor": X.columns, "Impurity importance": m.feature_importances_, "Permutation importance": pi.importances_mean}).sort_values("Permutation importance", ascending=False); res.table("Variable importance", imp)
    if task == "classification": res.table("Cross-validated confusion matrix", pd.crosstab(pd.Series(yy, name="Actual"), pd.Series(pcv, name="Predicted")), index=True)
    else:
        rm = math.sqrt(np.mean((yy - pcv) ** 2)); res.table("Cross-validated error", pd.DataFrame([{"RMSE (CV)": rm, "MAE (CV)": np.mean(np.abs(yy - pcv))}]))
    fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.2)); top = imp.head(12)[::-1]; axs[0].barh(top["Predictor"], top["Permutation importance"], color=PALETTE[0]); axs[0].set_title("Permutation importance")
    if meth == "tree": sktree.plot_tree(m, feature_names=list(X.columns), filled=True, ax=axs[1], fontsize=6, max_depth=3); axs[1].set_title("Tree (top 3 levels)")
    elif task == "regression": axs[1].scatter(yy, pcv, s=12); axs[1].plot([yy.min(), yy.max()], [yy.min(), yy.max()], color=PALETTE[3]); axs[1].set_xlabel("Actual"); axs[1].set_ylabel("CV predicted"); axs[1].set_title(L.fig("Predictive model"))
    else: axs[1].axis("off")
    res.fig("importance", save_fig(fig, outdir, f"predictive_{meth}"))
    res.summary.update({"cv_mean": float(cvs.mean()), "metric": sc, "top": imp["Predictor"].iloc[0]}); res.note("Always judge by cross-validated (not training) performance; random seed fixed at 1 for reproducibility.")
    return res


def pls(df, params, L: Lang, outdir):
    """responses (list) ; predictors; max_components; cv folds."""
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.model_selection import KFold
    ys, xs = params["responses"], params.get("predictors") or [c for c in numeric_cols(df) if c not in params["responses"]]; d = df[ys + xs].dropna(); X = d[xs].values; Y = d[ys].values
    A = int(params.get("max_components", min(8, len(xs), len(d) - 2))); cv = KFold(int(params.get("cv", 7)), shuffle=True, random_state=1); press = np.zeros(A)
    for a in range(1, A + 1):
        for tr, te in cv.split(X):
            mx, sx, my, sy = X[tr].mean(0), X[tr].std(0, ddof=1), Y[tr].mean(0), Y[tr].std(0, ddof=1); m = PLSRegression(a, scale=False).fit((X[tr] - mx) / sx, (Y[tr] - my) / sy)
            pr = m.predict((X[te] - mx) / sx); press[a - 1] += (((Y[te] - my) / sy - pr) ** 2).sum()
    Yz = (Y - Y.mean(0)) / Y.std(0, ddof=1); q2 = 1 - press / (Yz ** 2).sum()
    best = int(params.get("n_components") or 0)
    if not best:  # parsimonious rule: add components while cross-validated Q2 improves by more than 0.01
        best = 1
        while best < A and q2[best] - q2[best - 1] > 0.01: best += 1
    Xz = (X - X.mean(0)) / X.std(0, ddof=1); m = PLSRegression(best, scale=False).fit(Xz, Yz); T = m.x_scores_; W = m.x_weights_; Q = m.y_loadings_
    ssy = (T ** 2).sum(0) * (Q ** 2).sum(0); vip = np.sqrt(len(xs) * ((W / np.linalg.norm(W, axis=0)) ** 2 @ ssy) / ssy.sum())
    r2x = ((T @ m.x_loadings_.T) ** 2).sum() / (Xz ** 2).sum(); pred = m.predict(Xz); r2y = 1 - ((Yz - pred) ** 2).sum(0) / (Yz ** 2).sum(0)
    res = Result(L.t("PLS regression"))
    res.table("Components (cross-validated)", pd.DataFrame({"Components": range(1, A + 1), "Q2 (CV)": q2, "PRESS": press, "Chosen": [i + 1 == best for i in range(A)]}))
    res.table("Fit", pd.DataFrame([{"Components": best, "R2X (cum.)": r2x, **{f"R2Y {c}": r for c, r in zip(ys, r2y)}, "Q2 (CV, chosen)": q2[best - 1]}]))
    res.table("VIP scores", pd.DataFrame({"Predictor": xs, "VIP": vip, "Important (VIP>1)": vip > 1}).sort_values("VIP", ascending=False))
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.8)); axs[0].plot(range(1, A + 1), q2, "o-"); axs[0].set_title("Q2 vs components"); o = np.argsort(vip); axs[1].barh(np.array(xs)[o], vip[o], color=PALETTE[0]); axs[1].axvline(1, color=PALETTE[3], ls="--"); axs[1].set_title("VIP")
    axs[2].scatter(Yz[:, 0], pred[:, 0], s=12); axs[2].set_title(L.fig("PLS regression")); res.fig("PLS", save_fig(fig, outdir, "pls")); res.summary.update({"components": best, "q2": float(q2[best - 1])})
    return res


# ============================================================================
# Time series
# ============================================================================
def acf(x, nlags):
    x = np.asarray(x, float) - np.mean(x); n = len(x); c0 = (x @ x) / n; return np.array([(x[:n - k] @ x[k:]) / n / c0 for k in range(nlags + 1)])


def pacf(x, nlags):
    r = acf(x, nlags); out = [1.0]; phi = np.zeros((nlags + 1, nlags + 1))
    for k in range(1, nlags + 1):
        num = r[k] - sum(phi[k - 1, j] * r[k - j] for j in range(1, k)); den = 1 - sum(phi[k - 1, j] * r[j] for j in range(1, k)); phi[k, k] = num / den
        for j in range(1, k): phi[k, j] = phi[k - 1, j] - phi[k, k] * phi[k - 1, k - j]
        out.append(phi[k, k])
    return np.array(out)


def ljung_box(x, lags, nfit=0):
    n = len(x); r = acf(x, lags)[1:]; q = n * (n + 2) * np.sum(r ** 2 / (n - np.arange(1, lags + 1))); return q, stats.chi2.sf(q, max(lags - nfit, 1))


def _diff_poly(d, D, m):
    poly = np.array([1.0])
    for _ in range(d): poly = np.convolve(poly, [1, -1])
    for _ in range(D):
        s = np.zeros(m + 1); s[0] = 1; s[m] = -1; poly = np.convolve(poly, s)
    return poly


def arima_fit(y, p, d, q, D=0, m=1, with_mean=None):
    y = np.asarray(y, float); dp = _diff_poly(d, D, m); w = np.convolve(y, dp[::-1], mode="valid") if len(dp) > 1 else y.copy(); nw = len(w)
    if with_mean is None: with_mean = (d + D == 0)
    def resid(par):
        phi, th, mu = par[:p], par[p:p + q], (par[p + q] if with_mean else 0.0); e = np.zeros(nw); z = w - mu
        for t in range(nw):
            ar = sum(phi[i] * z[t - 1 - i] for i in range(p) if t - 1 - i >= 0); ma = sum(th[j] * e[t - 1 - j] for j in range(q) if t - 1 - j >= 0); e[t] = z[t] - ar - ma
        return e[p:] if p else e
    x0 = np.zeros(p + q + (1 if with_mean else 0)); x0[-1:] = [w.mean()] if with_mean else x0[-1:]
    r = optimize.minimize(lambda par: float(resid(par) @ resid(par)), x0, method="Nelder-Mead" if len(x0) > 1 else "BFGS", options={"maxiter": 4000, "xatol": 1e-8, "fatol": 1e-10}) if len(x0) else None
    par = r.x if r is not None else np.array([]); e = resid(par) if r is not None else w; css = float(e @ e); nn = len(e); s2 = css / nn
    k = p + q + (1 if with_mean else 0) + 1; aic = nn * math.log(s2) + 2 * k + nn * (1 + math.log(2 * math.pi))
    return dict(phi=par[:p], theta=par[p:p + q], mu=(par[p + q] if with_mean else 0.0), sigma2=s2, resid=e, aic=aic, aicc=aic + 2 * k * (k + 1) / max(nn - k - 1, 1), n=nn, dp=dp, w=w, y=y, with_mean=with_mean, p=p, q=q, d=d, D=D, m=m)


def arima_forecast(f, h):
    y, dp, phi, th, mu = f["y"], f["dp"], f["phi"], f["theta"], f["mu"]; p, q = f["p"], f["q"]
    AR = np.convolve(np.concatenate([[1.0], -phi]), dp); a = -AR[1:]; c = (1 - phi.sum()) * mu
    # full-sample residuals aligned to y
    w, nw = f["w"], len(f["w"]); z = w - mu; e = np.zeros(nw)
    for t in range(nw):
        ar = sum(phi[i] * z[t - 1 - i] for i in range(p) if t - 1 - i >= 0); ma = sum(th[j] * e[t - 1 - j] for j in range(q) if t - 1 - j >= 0); e[t] = z[t] - ar - ma
    off = len(y) - nw; efull = np.concatenate([np.zeros(off), e]); yf = list(y); ef = list(efull); out = []
    for _ in range(h):
        val = c + sum(a[i] * yf[-1 - i] for i in range(len(a)) if len(yf) - 1 - i >= 0) + sum(th[j] * ef[-1 - j] for j in range(q) if len(ef) - 1 - j >= 0); out.append(val); yf.append(val); ef.append(0.0)
    psi = [1.0]
    for k in range(1, h):
        v = (th[k - 1] if k - 1 < q else 0.0) + sum(a[i] * psi[k - 1 - i] for i in range(min(len(a), k))); psi.append(v)
    se = np.sqrt(f["sigma2"] * np.cumsum(np.array(psi) ** 2)); return np.array(out), se


def _hw(y, m, trend, seasonal, par):
    n = len(y); al, be, ga = par; l = y[:m].mean() if seasonal else y[0]; b = (y[m:2 * m].mean() - y[:m].mean()) / m if (trend and seasonal and n >= 2 * m) else ((y[1] - y[0]) if trend else 0.0)
    s = list((y[:m] - y[:m].mean()) if seasonal == "add" else (y[:m] / y[:m].mean())) if seasonal else []; fit = np.zeros(n); sse = 0.0
    for t in range(n):
        st = s[t % m] if seasonal else (0 if seasonal != "mul" else 1)
        f = (l + b) + st if seasonal == "add" else ((l + b) * st if seasonal == "mul" else l + b); fit[t] = f; err = y[t] - f; sse += err * err
        ln = al * (y[t] - st) + (1 - al) * (l + b) if seasonal == "add" else (al * (y[t] / st) + (1 - al) * (l + b) if seasonal == "mul" else al * y[t] + (1 - al) * (l + b))
        bn = be * (ln - l) + (1 - be) * b if trend else 0.0
        if seasonal: s[t % m] = ga * (y[t] - ln) + (1 - ga) * st if seasonal == "add" else ga * (y[t] / ln) + (1 - ga) * st
        l, b = ln, bn
    return sse, fit, l, b, s


def timeseries(df, params, L: Lang, outdir):
    """column; mode: explore|decompose|smooth|arima|auto_arima; period (m); horizon; p,d,q and seasonal P,D,Q (arima); method ml (exact Kalman ML, default) | css; trend (bool), seasonal add|mul|none (smooth)."""
    col = params["column"]; y = df[col].dropna().values.astype(float); n = len(y); mode = params.get("mode", "explore"); m = int(params.get("period", 1)); h = int(params.get("horizon", 12)); res = Result(L.t("Time series"))
    nl = int(params.get("nlags", min(40, n // 3)))
    if mode == "explore":
        r, pc = acf(y, nl), pacf(y, nl); bound = 1.96 / math.sqrt(n); q, pv = ljung_box(y, min(10, n // 5))
        res.table("Ljung-Box (white-noise test)", pd.DataFrame([{"Lags": min(10, n // 5), "Q": q, "p-value": pv, "n": n}])); res.table("ACF / PACF", pd.DataFrame({"Lag": range(nl + 1), "ACF": r, "PACF": pc}))
        fig, axs = plt.subplots(3, 1, figsize=(8, 8)); axs[0].plot(y); axs[0].set_title(col)
        for ax, v, t in [(axs[1], r, "ACF"), (axs[2], pc, "PACF")]: ax.bar(range(len(v)), v, width=.3); ax.axhline(bound, color=PALETTE[3], ls="--"); ax.axhline(-bound, color=PALETTE[3], ls="--"); ax.set_title(f"{L.fig('Autocorrelation')} - {t}")
        res.fig("series, ACF, PACF", save_fig(fig, outdir, "ts_explore")); res.summary.update({"ljung_p": float(pv)})
    elif mode == "decompose":
        kind = params.get("kind", "additive"); k = np.ones(m) / m
        if m % 2 == 0: ker = np.concatenate([[.5], np.ones(m - 1), [.5]]) / m
        else: ker = k
        tr = np.convolve(y, ker, mode="valid"); pad = (len(y) - len(tr)) // 2; trend = np.full(n, np.nan); trend[pad:pad + len(tr)] = tr
        det = y - trend if kind == "additive" else y / trend; seas = np.array([np.nanmean(det[i::m]) for i in range(m)]); seas = seas - seas.mean() if kind == "additive" else seas / seas.mean()
        sfull = np.array([seas[i % m] for i in range(n)]); rem = y - trend - sfull if kind == "additive" else y / (trend * sfull)
        res.table("Seasonal indices", pd.DataFrame({"Position": range(1, m + 1), "Index": seas}))
        fig, axs = plt.subplots(4, 1, figsize=(8, 8.5), sharex=True)
        for ax, v, t in zip(axs, [y, trend, sfull, rem], [col, "Trend", "Seasonal", "Remainder"]): ax.plot(v); ax.set_title(t)
        res.fig("decomposition", save_fig(fig, outdir, "ts_decompose")); res.summary["seasonal_amplitude"] = float(np.ptp(seas))
    elif mode == "smooth":
        trend = bool(params.get("trend", True)); seasonal = params.get("seasonal", "none"); seasonal = None if seasonal in (None, "none", False) else seasonal
        if seasonal and m < 2: raise ValueError("period>=2 required for seasonal smoothing")
        best = None
        for x0 in ([.3, .1, .1], [.6, .05, .2], [.1, .01, .05]):
            r = optimize.minimize(lambda p_: _hw(y, m, trend, seasonal, p_)[0], x0, bounds=[(1e-3, .999)] * 3, method="L-BFGS-B")
            if best is None or r.fun < best.fun: best = r
        sse, fit, l, b, s = _hw(y, m, trend, seasonal, best.x); resid = y - fit; sd = resid.std(ddof=1); rng = np.random.default_rng(1); sims = np.zeros((1000, h))
        for i in range(1000):
            ll, bb, ss = l, b, list(s)
            for j in range(h):
                st = ss[j % m] if seasonal else (0 if seasonal != "mul" else 1)
                base = ll + bb; f = base + st if seasonal == "add" else (base * st if seasonal == "mul" else base); e = rng.choice(resid); val = f + e; sims[i, j] = val
                al, be, ga = best.x; ln = al * (val - st) + (1 - al) * base if seasonal == "add" else (al * (val / st) + (1 - al) * base if seasonal == "mul" else al * val + (1 - al) * base)
                bn = be * (ln - ll) + (1 - be) * bb if trend else 0.0
                if seasonal: ss[j % m] = ga * (val - ln) + (1 - ga) * st if seasonal == "add" else ga * (val / ln) + (1 - ga) * st
                ll, bb = ln, bn
        fc = sims.mean(0); lo, hi = np.percentile(sims, [2.5, 97.5], axis=0)
        res.table("Smoothing parameters", pd.DataFrame([{"Alpha (level)": best.x[0], "Beta (trend)": best.x[1] if trend else np.nan, "Gamma (season)": best.x[2] if seasonal else np.nan, "Trend": trend, "Seasonal": seasonal or "none", "SSE": sse, "RMSE": math.sqrt(sse / n), "MAPE %": 100 * np.mean(np.abs(resid / y)) if (y != 0).all() else np.nan}]))
        res.table("Forecast", pd.DataFrame({"Step": range(1, h + 1), "Forecast": fc, "Lower 95%": lo, "Upper 95%": hi}))
        fig, ax = plt.subplots(figsize=(8, 3.8)); ax.plot(range(n), y, label="data"); ax.plot(range(n), fit, label="fit", alpha=.7); ax.plot(range(n, n + h), fc, color=PALETTE[3], label="forecast"); ax.fill_between(range(n, n + h), lo, hi, color=PALETTE[3], alpha=.2); ax.legend(); ax.set_title(L.fig("Forecast"))
        res.fig("smoothing forecast", save_fig(fig, outdir, "ts_smooth")); res.summary.update({"alpha": float(best.x[0]), "rmse": math.sqrt(sse / n)}); res.note("Prediction intervals by residual bootstrap simulation (1000 paths).")
    else:
        D = int(params.get("D", 0)); meth = (params.get("method") or "ml").lower(); P_, Q_ = int(params.get("P", 0)), int(params.get("Q", 0))
        if meth == "ml": from ss_ts2 import sarima_ml
        def _fit(p_, d_, q_):
            return sarima_ml(y, p_, d_, q_, P_, D, Q_, m) if meth == "ml" else arima_fit(y, p_, d_, q_, D, m)
        if mode == "auto_arima":
            d_ = int(params.get("d", 0)); cands = []
            for p_, q_ in itertools.product(range(4), range(4)):
                try: cands.append((_fit(p_, d_, q_)["aicc"], p_, q_))
                except Exception: pass
            _, p, q = min(cands); d = d_
            res.table("Model search (AICc)", pd.DataFrame(sorted(cands), columns=["AICc", "p", "q"]).head(10))
        else: p, d, q = int(params.get("p", 1)), int(params.get("d", 0)), int(params.get("q", 0))
        f = _fit(p, d, q); fc, se = arima_forecast(f, h); e = f["resid"]; lb = ljung_box(f.get("std_resid", e), min(10, len(e) // 5), p + q + P_ + Q_)
        if meth == "ml":
            names, vals, ses = f["names"], list(f["native"]), list(f["native_se"])
            tab = pd.DataFrame({"Parameter": names, "Estimate": vals, "Std Error": ses, "z": np.array(vals) / np.array(ses), "p-value": 2 * stats.norm.sf(np.abs(np.array(vals) / np.array(ses)))}) if names else pd.DataFrame({"Parameter": ["(none)"], "Estimate": [np.nan]})
            lab = f"SARIMA({p},{d},{q})({P_},{D},{Q_}) period {m}" if (P_ or Q_ or D) and m > 1 else f"ARIMA({p},{d},{q})"
            res.table(lab + " - exact Gaussian maximum likelihood", tab)
            res.table("Fit", pd.DataFrame([{"Sigma^2": f["sigma2"], "RMSE": math.sqrt(f["sigma2"]), "-2 log-lik": -2 * f["loglik"], "AIC": f["aic"], "AICc": f["aicc"], "Ljung-Box Q": lb[0], "Ljung-Box p": lb[1], "n used": f["n"]}]))
        else:
            names = [f"AR{i+1}" for i in range(p)] + [f"MA{j+1}" for j in range(q)] + (["Mean"] if f["with_mean"] else []); vals = list(f["phi"]) + list(f["theta"]) + ([f["mu"]] if f["with_mean"] else [])
            res.table(f"ARIMA({p},{d},{q})" + (f"({D}) period {m}" if D else "") + " - conditional least squares", pd.DataFrame({"Parameter": names, "Estimate": vals}) if names else pd.DataFrame({"Parameter": ["(none)"], "Estimate": [np.nan]}))
            res.table("Fit", pd.DataFrame([{"Sigma^2": f["sigma2"], "RMSE": math.sqrt(f["sigma2"]), "AIC (CSS-based)": f["aic"], "AICc": f["aicc"], "Ljung-Box Q": lb[0], "Ljung-Box p": lb[1], "n used": f["n"]}]))
        res.table("Forecast", pd.DataFrame({"Step": range(1, h + 1), "Forecast": fc, "Lower 95%": fc - 1.96 * se, "Upper 95%": fc + 1.96 * se}))
        fig, axs = plt.subplots(2, 1, figsize=(8, 7)); axs[0].plot(range(n), y); axs[0].plot(range(n, n + h), fc, color=PALETTE[3]); axs[0].fill_between(range(n, n + h), fc - 1.96 * se, fc + 1.96 * se, color=PALETTE[3], alpha=.2); axs[0].set_title(L.fig("Forecast"))
        ra = acf(e, min(30, len(e) // 3)); axs[1].bar(range(len(ra)), ra, width=.3); b_ = 1.96 / math.sqrt(len(e)); axs[1].axhline(b_, color=PALETTE[3], ls="--"); axs[1].axhline(-b_, color=PALETTE[3], ls="--"); axs[1].set_title("Residual ACF")
        res.fig("ARIMA forecast", save_fig(fig, outdir, "ts_arima")); res.summary.update({"phi": [float(v) for v in f["phi"]], "theta": [float(v) for v in f["theta"]], "mu": float(f["mu"]), "sigma2": float(f["sigma2"]), "forecast": [float(v) for v in fc]})
        res.note(("Exact Gaussian ML (Kalman filter), multiplicative seasonal terms supported (P, Q, D, period m); " if meth == "ml" else "Conditional least squares (method='css'; no seasonal AR/MA terms); ") + "forecast intervals assume Gaussian innovations and the correct model order, and use the CSS-style residual recursion for the starting innovations.")
    return res
