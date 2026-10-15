"""ss_linear.py - One-way ANOVA, multi-way ANOVA / GLM (Type III), regression, logistic/Poisson GLM, power.
JMP: Fit Y by X, Fit Model (Standard Least Squares / Generalized Regression).  SPSS: One-Way ANOVA, GLM Univariate, Linear Regression, Logistic Regression.
Pure numpy/scipy (no statsmodels needed).
"""
from __future__ import annotations
import itertools
import numpy as np, pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from ss_common import Result, Lang, save_fig, PALETTE, numeric_cols


# ----------------------------------------------------------------------------
# Design matrix with effect (sum-to-zero) coding  -> Type III SS (SPSS/JMP default)
# ----------------------------------------------------------------------------
def _factor_cols(s: pd.Series, name: str, coding: str = "effect"):
    """coding='effect' (sum-to-zero, JMP/Type III) or 'reference' (indicator, last level is the reference - SPSS parameter estimates)."""
    lv = sorted(s.dropna().unique(), key=str)
    cols, names = [], []
    for l in lv[:-1]:
        cols.append(((s == l).astype(float) - (s == lv[-1]).astype(float)).values if coding == "effect" else (s == l).astype(float).values)
        names.append(f"{name}[{l}]")
    return cols, names, lv


def build_terms(variables, degree):
    terms = [(v,) for v in variables]
    for k in range(2, int(degree) + 1):
        terms += list(itertools.combinations(variables, k))
    return terms


class OLS:
    def __init__(self, df, y, factors=(), covariates=(), terms=None, degree=1, intercept=True, coding="effect"):
        self.y_name = y; self.coding = coding
        self.factors, self.covariates = list(factors), list(covariates)
        cols_needed = [y] + self.factors + self.covariates
        d = df[cols_needed].dropna().copy()
        for f in self.factors:
            d[f] = d[f].astype(str)
        self.d = d
        self.terms = terms or build_terms(self.factors + self.covariates, degree)
        base = {}
        for f in self.factors:
            c, n, lv = _factor_cols(d[f], f, coding)
            base[f] = (c, n)
        for v in self.covariates:
            base[v] = ([d[v].values.astype(float)], [v])
        self.levels = {f: sorted(d[f].unique(), key=str) for f in self.factors}
        names, cols, self.term_idx = [], [], {}
        if intercept:
            names.append("Intercept"); cols.append(np.ones(len(d)))
        for t in self.terms:
            tc, tn = [np.ones(len(d))], [""]
            for v in t:
                c, n = base[v]
                tc, tn = ([a * b for a in tc for b in c], [(x + "*" + m) if x else m for x in tn for m in n])
            start = len(cols)
            cols += tc; names += tn
            self.term_idx[t] = list(range(start, start + len(tc)))
        self.names = names
        self.X = np.column_stack(cols)
        self.yv = d[y].values.astype(float)
        self._fit()

    def _fit(self):
        X, y = self.X, self.yv
        self.n = len(y)
        self.rank = np.linalg.matrix_rank(X)
        self.beta = np.linalg.lstsq(X, y, rcond=None)[0]
        self.fitted = X @ self.beta
        self.resid = y - self.fitted
        self.rss = float(self.resid @ self.resid)
        self.dfe = self.n - self.rank
        self.mse = self.rss / self.dfe if self.dfe > 0 else np.nan
        XtXi = np.linalg.pinv(X.T @ X)
        self.cov = self.mse * XtXi
        self.se = np.sqrt(np.clip(np.diag(self.cov), 0, None))
        self.has_int = "Intercept" in self.names
        self.tss = float(((y - y.mean()) ** 2).sum()) if self.has_int else float((y ** 2).sum())
        self.r2 = 1 - self.rss / self.tss
        pm = self.rank - (1 if self.has_int else 0)
        self.adj_r2 = 1 - (self.rss / self.dfe) / (self.tss / (self.n - (1 if self.has_int else 0)))
        self.H = X @ XtXi @ X.T if self.n <= 3000 else None
        self.model_df = pm

    def ss3(self):
        rows = []
        for t, idx in self.term_idx.items():
            keep = [i for i in range(self.X.shape[1]) if i not in idx]
            Xr = self.X[:, keep]
            br = np.linalg.lstsq(Xr, self.yv, rcond=None)[0]
            rss_r = float(((self.yv - Xr @ br) ** 2).sum())
            df_t = self.rank - np.linalg.matrix_rank(Xr)
            ss = rss_r - self.rss
            F = (ss / df_t) / self.mse if df_t > 0 and self.mse > 0 else np.nan
            rows.append({"Source": "*".join(t), "DF": df_t, "Type III SS": ss, "Mean Square": ss / df_t if df_t else np.nan,
                         "F Ratio": F, "Prob > F": stats.f.sf(F, df_t, self.dfe) if df_t else np.nan})
        return pd.DataFrame(rows)

    def anova_table(self):
        ssm = self.tss - self.rss
        F = (ssm / self.model_df) / self.mse
        return pd.DataFrame([
            {"Source": "Model", "DF": self.model_df, "Sum of Squares": ssm, "Mean Square": ssm / self.model_df, "F Ratio": F, "Prob > F": stats.f.sf(F, self.model_df, self.dfe)},
            {"Source": "Error", "DF": self.dfe, "Sum of Squares": self.rss, "Mean Square": self.mse, "F Ratio": np.nan, "Prob > F": np.nan},
            {"Source": "Total", "DF": self.n - 1, "Sum of Squares": self.tss, "Mean Square": np.nan, "F Ratio": np.nan, "Prob > F": np.nan}])

    def params_table(self):
        t = self.beta / self.se
        tc = stats.t.ppf(0.975, self.dfe)
        return pd.DataFrame({"Term": self.names, "Estimate": self.beta, "Std Error": self.se, "t Ratio": t,
                             "Prob > |t|": 2 * stats.t.sf(np.abs(t), self.dfe),
                             "Lower 95%": self.beta - tc * self.se, "Upper 95%": self.beta + tc * self.se})

    def predict_df(self, newdf):
        if self.coding != "effect": raise NotImplementedError("predict_df requires effect coding")
        d = newdf.copy()
        # rebuild using same coding
        cols = []
        if self.has_int:
            cols.append(np.ones(len(d)))
        base = {}
        for f in self.factors:
            s = d[f].astype(str)
            lv = self.levels[f]
            base[f] = [((s == l).astype(float) - (s == lv[-1]).astype(float)).values for l in lv[:-1]]
        for v in self.covariates:
            base[v] = [d[v].values.astype(float)]
        for t in self.terms:
            tc = [np.ones(len(d))]
            for v in t:
                tc = [a * b for a in tc for b in base[v]]
            cols += tc
        return np.column_stack(cols) @ self.beta

    def lsmeans(self, factor):
        rows = []
        for l in self.levels[factor]:
            tmp = self.d.copy(); tmp[factor] = l
            rows.append((l, self.predict_df(tmp).mean()))
        return rows

    def diagnostics(self):
        h = np.diag(self.H) if self.H is not None else np.full(self.n, self.rank / self.n)
        stud = self.resid / np.sqrt(self.mse * np.clip(1 - h, 1e-9, None))
        cook = stud ** 2 * h / (self.rank * np.clip(1 - h, 1e-9, None))
        return h, stud, cook


def _diag_plots(m: OLS, L: Lang, outdir, prefix):
    h, stud, cook = m.diagnostics()
    fig, axs = plt.subplots(2, 3, figsize=(13, 7))
    ax = axs[0, 0]; ax.plot(m.fitted, m.yv, "o", ms=3.5, color=PALETTE[0]); lo, hi = min(m.fitted.min(), m.yv.min()), max(m.fitted.max(), m.yv.max())
    ax.plot([lo, hi], [lo, hi], color=PALETTE[3]); ax.set_title(L.fig("Actual by Predicted")); ax.set_xlabel(L.fig("Predicted")); ax.set_ylabel(L.fig("Actual"))
    ax = axs[0, 1]; ax.plot(m.fitted, m.resid, "o", ms=3.5, color=PALETTE[0]); ax.axhline(0, color=PALETTE[3]); ax.set_title(L.fig("Residual by Predicted"))
    ax = axs[0, 2]; (osm, osr), (s, i, r) = stats.probplot(stud, dist="norm"); ax.plot(osm, osr, "o", ms=3.5, color=PALETTE[0]); ax.plot(osm, s * np.asarray(osm) + i, color=PALETTE[3]); ax.set_title(L.fig("Residual normal Q-Q"))
    ax = axs[1, 0]; ax.hist(m.resid, bins="auto", color=PALETTE[0], edgecolor="white"); ax.set_title(L.fig("Residual histogram"))
    ax = axs[1, 1]; ax.plot(np.arange(1, m.n + 1), m.resid, "o-", ms=3, lw=.6, color=PALETTE[0]); ax.axhline(0, color=PALETTE[3]); ax.set_title(L.fig("Observation order"))
    ax = axs[1, 2]; ax.stem(np.arange(1, m.n + 1), cook, markerfmt="o", basefmt=" "); ax.axhline(4 / m.n, color=PALETTE[3], ls="--"); ax.set_title(L.fig("Cook's D"))
    fig.tight_layout()
    return save_fig(fig, outdir, prefix + "_diagnostics"), (stud, cook)


def _fit_summary(m: OLS, L: Lang):
    return pd.DataFrame([{L.t("R-squared"): m.r2, L.t("Adj R-squared"): m.adj_r2, L.t("RMSE"): np.sqrt(m.mse),
                          L.t("Mean"): m.yv.mean(), L.t("Observations"): m.n}])


def _ptab(m, params, data, y, fa):
    """Parameter table in the requested coding. 'effect' (default) = JMP; 'reference' = SPSS indicator coding (last level is the reference). Tests/SS always use effect coding."""
    if params.get("coding") != "reference": return m.params_table()
    mr = OLS(data, y, m.factors, m.covariates, terms=m.terms, coding="reference"); return mr.params_table()


def _tr_cols(df, L):
    return df.rename(columns={c: L.t(c) for c in df.columns})


# ----------------------------------------------------------------------------
def oneway(df, params, L: Lang, outdir):
    """params: response, factor, posthoc (tukey|dunnett), control (level for dunnett)."""
    y, f = params["response"], params["factor"]
    d = df[[y, f]].dropna().copy(); d[f] = d[f].astype(str)
    lv = sorted(d[f].unique(), key=str)
    groups = [d.loc[d[f] == l, y].values for l in lv]
    k, N = len(groups), len(d)
    m = OLS(d, y, factors=[f])
    res = Result(f"{L.t('Oneway Analysis')}: {y} {L.t('by')} {f}")
    res.table(L.t("Summary of Fit"), _tr_cols(_fit_summary(m, L), L))
    res.table(L.t("Analysis of Variance"), _tr_cols(m.anova_table(), L))
    mt = []
    for l, g in zip(lv, groups):
        se = g.std(ddof=1) / np.sqrt(len(g)) if len(g) > 1 else np.nan
        tc = stats.t.ppf(.975, len(g) - 1) if len(g) > 1 else np.nan
        mt.append({L.t("Level"): l, "N": len(g), L.t("Mean"): g.mean(), L.t("Std Dev"): g.std(ddof=1), L.t("Std Err Mean"): se,
                   L.t("Lower 95%"): g.mean() - tc * se, L.t("Upper 95%"): g.mean() + tc * se})
    res.table(L.t("Means and Std Deviations"), pd.DataFrame(mt))
    lev = stats.levene(*groups, center="median"); bf = stats.levene(*groups, center="mean")
    res.table(L.t("Levene test"), pd.DataFrame([{L.t("Method"): "Brown-Forsythe (median)", L.t("Statistic"): lev.statistic, "p-value": lev.pvalue},
                                                  {L.t("Method"): "Levene (mean)", L.t("Statistic"): bf.statistic, "p-value": bf.pvalue}]))
    n_i = np.array([len(g) for g in groups], float); mu = np.array([g.mean() for g in groups]); v = np.array([g.var(ddof=1) for g in groups])
    w = n_i / v; W = w.sum(); mw = (w * mu).sum() / W
    lam = ((1 - w / W) ** 2 / (n_i - 1)).sum()
    Fw = ((w * (mu - mw) ** 2).sum() / (k - 1)) / (1 + 2 * (k - 2) / (k ** 2 - 1) * lam)
    df2 = (k ** 2 - 1) / (3 * lam)
    res.table(L.t("Welch ANOVA"), pd.DataFrame([{L.t("F Ratio"): Fw, "DF num": k - 1, "DF den": df2, "p-value": stats.f.sf(Fw, k - 1, df2)}]))
    kw = stats.kruskal(*groups)
    res.table(L.t("Kruskal-Wallis"), pd.DataFrame([{L.t("Statistic"): kw.statistic, "DF": k - 1, "p-value": kw.pvalue}]))
    ph = params.get("posthoc", "tukey")
    cmp_rows = []
    if k > 2 and ph == "tukey":
        tk = stats.tukey_hsd(*groups)
        ci = tk.confidence_interval(0.95)
        for i, j in itertools.combinations(range(k), 2):
            cmp_rows.append({L.t("Pair"): f"{lv[i]} - {lv[j]}", L.t("Difference"): mu[i] - mu[j], L.t("Lower 95%"): ci.low[i, j], L.t("Upper 95%"): ci.high[i, j], "p-value": tk.pvalue[i, j]})
        res.table(L.t("Tukey HSD comparisons"), pd.DataFrame(cmp_rows))
    elif k > 2 and ph == "dunnett":
        ctrl = str(params.get("control", lv[0])); ci_ = lv.index(ctrl)
        others = [i for i in range(k) if i != ci_]
        dn = stats.dunnett(*[groups[i] for i in others], control=groups[ci_])
        for a, i in enumerate(others):
            cmp_rows.append({L.t("Pair"): f"{lv[i]} - {ctrl}", L.t("Difference"): mu[i] - mu[ci_], "p-value": dn.pvalue[a]})
        res.table("Dunnett vs control", pd.DataFrame(cmp_rows))
    fig, axs = plt.subplots(1, 2 if cmp_rows and ph == "tukey" else 1, figsize=(11 if cmp_rows else 6, 4.3), squeeze=False)
    ax = axs[0, 0]
    for i, g in enumerate(groups):
        ax.plot(np.random.default_rng(1).normal(i, 0.06, len(g)), g, "o", ms=3.5, alpha=.55, color=PALETTE[i % 8])
        ax.errorbar(i, g.mean(), yerr=stats.sem(g) * stats.t.ppf(.975, len(g) - 1) if len(g) > 1 else 0, fmt="D", color="k", capsize=5, ms=7)
    ax.axhline(d[y].mean(), color="#999", ls="--", lw=1)
    ax.set_xticks(range(k)); ax.set_xticklabels(lv); ax.set_ylabel(y); ax.set_title(f"{L.fig('Oneway Analysis')}: {y} {L.fig('by')} {f}")
    if cmp_rows and ph == "tukey":
        ax = axs[0, 1]
        for r_i, r in enumerate(cmp_rows):
            col = PALETTE[3] if r["p-value"] < .05 else "#666"
            ax.plot([r[L.t("Lower 95%")], r[L.t("Upper 95%")]], [r_i, r_i], color=col, lw=2); ax.plot(r[L.t("Difference")], r_i, "o", color=col)
        ax.axvline(0, color="#999", ls="--"); ax.set_yticks(range(len(cmp_rows))); ax.set_yticklabels([r[L.t("Pair")] for r in cmp_rows], fontsize=8); ax.set_title("Tukey HSD 95%")
    res.fig(L.t("Oneway Analysis"), save_fig(fig, outdir, f"oneway_{y}_{f}"))
    pth, (stud, cook) = _diag_plots(m, L, outdir, f"oneway_{y}_{f}")
    res.fig(L.t("Residuals"), pth)
    Fv = res.tables[1][1].iloc[0]["F Ratio" if not L.rtl else L.t("F Ratio")]
    res.summary = {"F": Fv, "p": res.tables[1][1].iloc[0][L.t("Prob > F")], "R2": m.r2, "welch_p": stats.f.sf(Fw, k - 1, df2), "kruskal_p": kw.pvalue, "levene_p": lev.pvalue}
    if lev.pvalue < .05:
        res.note("Variances differ (Levene p<.05): prefer Welch ANOVA / Games-Howell-type comparisons over the classic F-test.")
    return res


def glm_anova(df, params, L: Lang, outdir):
    """Multi-way ANOVA / ANCOVA / GLM univariate.  params: response, factors (list), covariates (list), degree (interaction order, default 2 if >1 factor)."""
    y = params["response"]; fa = params.get("factors", []); cv = params.get("covariates", [])
    deg = params.get("degree", 2 if len(fa) + len(cv) > 1 else 1)
    m = OLS(df, y, fa, cv, degree=deg)
    res = Result(f"{L.t('Analysis of Variance')}: {y}")
    res.table(L.t("Summary of Fit"), _tr_cols(_fit_summary(m, L), L))
    res.table(L.t("Analysis of Variance"), _tr_cols(m.anova_table(), L))
    res.table(f"{L.t('Effect Tests')} ({L.t('Type III SS')})", _tr_cols(m.ss3(), L))
    res.table(L.t("Parameter Estimates") + (" (reference coding, last level = reference)" if params.get("coding") == "reference" else ""), _tr_cols(_ptab(m, params, m.d, y, fa), L))
    for f_ in fa:
        ls = m.lsmeans(f_)
        res.table(f"LS Means - {f_}", pd.DataFrame(ls, columns=[L.t("Level"), "LS Mean"]))
    # main effect + interaction plots
    if fa:
        n = len(fa)
        fig, axs = plt.subplots(1, n, figsize=(4.2 * n, 3.8), squeeze=False)
        for i, f_ in enumerate(fa):
            ls = m.lsmeans(f_)
            axs[0, i].plot([str(a) for a, _ in ls], [b for _, b in ls], "o-", color=PALETTE[0], lw=2)
            axs[0, i].set_title(f"{L.fig('Main Effects')}: {f_}")
        res.fig(L.t("Main Effects"), save_fig(fig, outdir, "glm_main"))
        if len(fa) >= 2:
            pairs = list(itertools.combinations(fa, 2))[:6]
            fig, axs = plt.subplots(1, len(pairs), figsize=(4.4 * len(pairs), 3.8), squeeze=False)
            for i, (a, b) in enumerate(pairs):
                for j, lb in enumerate(m.levels[b]):
                    ys = []
                    for la in m.levels[a]:
                        tmp = m.d.copy(); tmp[a] = la; tmp[b] = lb
                        ys.append(m.predict_df(tmp).mean())
                    axs[0, i].plot(m.levels[a], ys, "o-", color=PALETTE[j % 8], label=f"{b}={lb}")
                axs[0, i].set_title(f"{L.fig('Interaction plot')}: {a}*{b}"); axs[0, i].legend(fontsize=7)
            res.fig(L.t("Interaction plot"), save_fig(fig, outdir, "glm_interaction"))
    pth, _ = _diag_plots(m, L, outdir, "glm")
    res.fig(L.t("Residuals"), pth)
    res.summary = {"R2": m.r2, "RMSE": float(np.sqrt(m.mse)), "terms_p": {r["Source"]: r["Prob > F"] for _, r in m.ss3().iterrows()}}
    res.note("Type III sums of squares with effect (sum-to-zero) coding - the default of JMP Standard Least Squares and SPSS GLM.")
    return res


def regression(df, params, L: Lang, outdir):
    """params: response, predictors (numeric list), factors (optional), degree (interaction order; 1 default), poly (polynomial degree per numeric predictor, default 1), stepwise (bool, backward), alpha (0.05), predict_at (dict optional)."""
    y = params["response"]; xs = list(params.get("predictors", [])); fa = list(params.get("factors", []))
    d = df.copy()
    poly = int(params.get("poly", 1))
    allx = list(xs)
    if poly > 1:
        for x in xs:
            c = d[x] - d[x].mean()
            for p in range(2, poly + 1):
                nm = f"{x}^{p}"; d[nm] = c ** p; allx.append(nm)
        for x in xs:
            d[x] = d[x] - d[x].mean()
    deg = int(params.get("degree", 1))
    vars_ = fa + allx
    terms = build_terms(fa + xs, deg) + [(a,) for a in allx if a not in xs]
    m = OLS(d, y, fa, allx, terms=terms)
    if params.get("stepwise"):
        alpha = float(params.get("alpha", .05))
        while True:
            t3 = m.ss3()
            cand = t3[t3["Prob > F"] > alpha]
            if cand.empty or len(m.terms) <= 1:
                break
            drop = cand.sort_values("Prob > F").iloc[-1]["Source"]
            tdrop = tuple(drop.split("*"))
            if any(set(tdrop) < set(t) for t in m.terms if t != tdrop):
                break
            terms = [t for t in m.terms if t != tdrop]
            m = OLS(d, y, fa, [c for c in allx if any(c in t for t in terms)], terms=terms)
    res = Result(f"{L.t('Model')}: {y}")
    res.table(L.t("Summary of Fit"), _tr_cols(_fit_summary(m, L), L))
    res.table(L.t("Analysis of Variance"), _tr_cols(m.anova_table(), L))
    res.table(L.t("Parameter Estimates") + (" (reference coding, last level = reference)" if params.get("coding") == "reference" else ""), _tr_cols(_ptab(m, params, m.d, y, fa), L))
    res.table(L.t("Effect Tests"), _tr_cols(m.ss3(), L))
    if len(m.covariates) > 1:
        Xc = m.d[m.covariates]
        vif = []
        for c in m.covariates:
            others = [o for o in m.covariates if o != c]
            if not others:
                continue
            r2 = OLS(Xc, c, [], others).r2
            vif.append({L.t("Variable"): c, "VIF": 1 / (1 - r2) if r2 < 1 else np.inf})
        if vif:
            res.table("VIF (multicollinearity)", pd.DataFrame(vif))
    pth, (stud, cook) = _diag_plots(m, L, outdir, "reg")
    res.fig(L.t("Residuals"), pth)
    if len(xs) == 1 and not fa:
        x = xs[0]
        fig, ax = plt.subplots(figsize=(6, 4.2))
        xs_ = np.linspace(df[x].min(), df[x].max(), 100)
        tmp = pd.DataFrame({x: xs_ - (df[x].mean() if poly > 1 else 0)})
        if poly > 1:
            for p in range(2, poly + 1):
                tmp[f"{x}^{p}"] = tmp[x] ** p
        pr = m.predict_df(tmp)
        se = np.sqrt(np.einsum("ij,jk,ik->i", np.column_stack([np.ones(100)] + [tmp[c].values for c in m.covariates]), m.cov, np.column_stack([np.ones(100)] + [tmp[c].values for c in m.covariates])))
        tc = stats.t.ppf(.975, m.dfe)
        ax.plot(df[x], df[y], "o", ms=4, alpha=.6, color=PALETTE[0]); ax.plot(xs_, pr, color=PALETTE[3], lw=2)
        ax.fill_between(xs_, pr - tc * se, pr + tc * se, color=PALETTE[3], alpha=.15); ax.set_xlabel(x); ax.set_ylabel(y)
        res.fig("Fit + 95% CI", save_fig(fig, outdir, "reg_fit"))
    res.summary = {"R2": m.r2, "adjR2": m.adj_r2, "RMSE": float(np.sqrt(m.mse)), "n": m.n,
                   "params": dict(zip(m.names, m.beta.round(6)))}
    big = np.where(np.abs(stud) > 3)[0]
    if len(big):
        res.note(f"{len(big)} observation(s) with |studentized residual| > 3 (rows {list(map(int, big[:10]))}) - check for outliers.")
    return res


def glm(df, params, L: Lang, outdir):
    """Logistic (binomial) / Poisson regression by IRLS.  params: response, predictors, factors, family (binomial|poisson), positive (level treated as 1 for binomial)."""
    y = params["response"]; xs = list(params.get("predictors", [])); fa = list(params.get("factors", []))
    fam = params.get("family", "binomial")
    d = df[[y] + xs + fa].dropna().copy()
    if fam == "binomial":
        pos = params.get("positive")
        if pos is None:
            pos = sorted(d[y].unique(), key=str)[-1]
        yv = (d[y].astype(str) == str(pos)).astype(float).values
    else:
        yv = d[y].values.astype(float)
    d["_y"] = yv
    base = OLS(d, "_y", fa, xs, degree=1, coding=params.get("coding", "effect"))
    X = base.X
    beta = np.zeros(X.shape[1])
    for _ in range(100):
        eta = X @ beta
        if fam == "binomial":
            mu = 1 / (1 + np.exp(-eta)); W = mu * (1 - mu)
        else:
            mu = np.exp(eta); W = mu
        W = np.clip(W, 1e-9, None)
        z = eta + (yv - mu) / W
        XtW = X.T * W
        nb = np.linalg.solve(XtW @ X + 1e-10 * np.eye(X.shape[1]), XtW @ z)
        if np.max(np.abs(nb - beta)) < 1e-9:
            beta = nb; break
        beta = nb
    eta = X @ beta
    mu = 1 / (1 + np.exp(-eta)) if fam == "binomial" else np.exp(eta)
    W = np.clip(mu * (1 - mu) if fam == "binomial" else mu, 1e-9, None)
    cov = np.linalg.inv((X.T * W) @ X)
    se = np.sqrt(np.diag(cov)); z = beta / se
    def _lfact(v):
        from scipy.special import gammaln
        return gammaln(np.asarray(v, float) + 1)
    if fam == "binomial":
        ll = (yv * np.log(np.clip(mu, 1e-12, 1)) + (1 - yv) * np.log(np.clip(1 - mu, 1e-12, 1))).sum()
        mu0 = yv.mean(); ll0 = len(yv) * (mu0 * np.log(mu0) + (1 - mu0) * np.log(1 - mu0))
    else:
        ll = (yv * np.log(np.clip(mu, 1e-12, None)) - mu - _lfact(yv)).sum()
        mu0 = yv.mean(); ll0 = (yv * np.log(mu0) - mu0 - _lfact(yv)).sum()
    lr = 2 * (ll - ll0)
    k = X.shape[1] - 1
    res = Result(L.t("Logistic Regression") if fam == "binomial" else "Poisson Regression")
    pt = pd.DataFrame({"Term": base.names, "Estimate": beta, "Std Error": se, "z": z, "Prob > |z|": 2 * stats.norm.sf(np.abs(z)),
                       L.t("Odds Ratio") if fam == "binomial" else "Rate Ratio": np.exp(beta),
                       "Lower 95%": np.exp(beta - 1.96 * se), "Upper 95%": np.exp(beta + 1.96 * se)})
    res.table(L.t("Parameter Estimates"), _tr_cols(pt, L))
    res.table("Model fit", pd.DataFrame([{"Log-Likelihood": ll, "LR Chi2": lr, "DF": k, "p-value": stats.chi2.sf(lr, k), "McFadden R2": 1 - ll / ll0, "AIC": -2 * ll + 2 * (k + 1), "BIC": -2 * ll + np.log(len(yv)) * (k + 1)}]))
    res.summary = {"LR_p": stats.chi2.sf(lr, k), "McFadden_R2": 1 - ll / ll0}
    if fam == "binomial":
        from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix
        auc = roc_auc_score(yv, mu); fpr, tpr, _ = roc_curve(yv, mu)
        cm = confusion_matrix(yv, (mu >= .5).astype(int))
        res.table(L.t("Confusion matrix"), pd.DataFrame(cm, index=["Actual 0", "Actual 1"], columns=["Pred 0", "Pred 1"]), index=True)
        res.table(L.t("Accuracy"), pd.DataFrame([{L.t("Accuracy"): np.trace(cm) / cm.sum(), L.t("AUC"): auc}]))
        fig, ax = plt.subplots(figsize=(4.8, 4.5)); ax.plot(fpr, tpr, color=PALETTE[0], lw=2); ax.plot([0, 1], [0, 1], "--", color="#999")
        ax.set_title(f"{L.fig('ROC curve')} ({L.fig('AUC')}={auc:.3f})"); ax.set_xlabel("1 - Specificity"); ax.set_ylabel("Sensitivity")
        res.fig(L.t("ROC curve"), save_fig(fig, outdir, "roc"))
        res.summary["AUC"] = auc
    return res


def power(df, params, L: Lang, outdir):
    """Sample-size / power calculator (no data needed).  params: test (ttest2|ttest1|anova), effect_size (d for t-test, Cohen f for ANOVA), alpha, power (target), k (groups for ANOVA), n (optional - compute power at n)."""
    test = params.get("test", "ttest2"); es = float(params["effect_size"]); a = float(params.get("alpha", .05)); tgt = float(params.get("power", .8)); k = int(params.get("k", 3))

    def pw(n):
        if test == "anova":
            lam = es ** 2 * n * k; df1, df2 = k - 1, k * (n - 1)
            return stats.ncf.sf(stats.f.ppf(1 - a, df1, df2), df1, df2, lam)
        df_ = 2 * (n - 1) if test == "ttest2" else n - 1
        nc = es * np.sqrt(n / 2 if test == "ttest2" else n)
        tc = stats.t.ppf(1 - a / 2, df_)
        return stats.nct.sf(tc, df_, nc) + stats.nct.cdf(-tc, df_, nc)
    ns = np.arange(2, 1000)
    pv = np.array([pw(n) for n in ns])
    need = int(ns[np.argmax(pv >= tgt)]) if (pv >= tgt).any() else None
    res = Result(L.t("Power"))
    rows = [{L.t("Sample size"): need, L.t("Power"): pw(need) if need else np.nan, L.t("Alpha"): a, L.t("Effect size"): es}]
    if params.get("n"):
        rows.append({L.t("Sample size"): int(params["n"]), L.t("Power"): pw(int(params["n"])), L.t("Alpha"): a, L.t("Effect size"): es})
    res.table(L.t("Power"), pd.DataFrame(rows))
    fig, ax = plt.subplots(figsize=(6, 4)); ax.plot(ns[:min(len(ns), (need or 100) * 2 + 5)], pv[:min(len(ns), (need or 100) * 2 + 5)], color=PALETTE[0], lw=2); ax.axhline(tgt, color=PALETTE[3], ls="--")
    ax.set_xlabel(L.fig("Sample size") + (" (per group)" if test != "ttest1" else "")); ax.set_ylabel(L.fig("Power"))
    res.fig(L.t("Power"), save_fig(fig, outdir, "power"))
    res.summary = {"n_required_per_group": need}
    return res
