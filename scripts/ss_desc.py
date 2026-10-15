"""ss_desc.py - Descriptive, distribution, tests, correlation, crosstab, PCA, clustering (JMP Distribution / Analyze; SPSS Explore / Descriptives / Compare Means / Correlate / Chi-square / Factor / Cluster)."""
from __future__ import annotations
import numpy as np, pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from ss_common import Result, Lang, save_fig, PALETTE, numeric_cols, get_param


def _moments(x):
    x = np.asarray(x, float)
    n = len(x)
    sd = x.std(ddof=1) if n > 1 else np.nan
    return {"N": n, "Mean": x.mean(), "Std Dev": sd, "Std Err Mean": sd / np.sqrt(n) if n > 1 else np.nan,
            "Variance": sd ** 2, "Skewness": stats.skew(x, bias=False) if n > 2 else np.nan,
            "Kurtosis": stats.kurtosis(x, bias=False) if n > 3 else np.nan,
            "Minimum": x.min(), "Maximum": x.max()}


def distribution(df, params, L: Lang, outdir):
    """params: columns (list, default all numeric), by (optional grouping column), spec limits not used here."""
    cols = get_param(params, "columns") or numeric_cols(df)
    by = params.get("by")
    res = Result(L.t("Distribution"))
    rows, qrows, nrows = [], [], []
    for c in cols:
        groups = [("", df[c].dropna())] if not by else [(str(g), d[c].dropna()) for g, d in df.groupby(by)]
        for g, x in groups:
            label = c if not g else f"{c} [{g}]"
            m = _moments(x)
            rows.append({L.t("Variable"): label, **{L.t(k): v for k, v in m.items()}})
            qs = x.quantile([0, .01, .05, .10, .25, .5, .75, .90, .95, .99, 1.0])
            qrows.append({L.t("Variable"): label, **{f"{q*100:g}%": v for q, v in qs.items()}})
            if len(x) >= 3:
                sw = stats.shapiro(x) if len(x) <= 5000 else (np.nan, np.nan)
                dag = stats.normaltest(x) if len(x) >= 8 else (np.nan, np.nan)
                ad = stats.anderson(x, dist="norm")
                nrows.append({L.t("Variable"): label, "Shapiro-Wilk W": sw[0], "Shapiro-Wilk p": sw[1],
                              "D'Agostino K2": dag[0], "D'Agostino p": dag[1],
                              "Anderson-Darling A2": ad.statistic, "AD crit. 5%": ad.critical_values[2]})
    res.table(L.t("Summary Statistics"), pd.DataFrame(rows))
    res.table(L.t("Quantiles"), pd.DataFrame(qrows))
    if nrows:
        res.table(L.t("Normality tests"), pd.DataFrame(nrows))
    for c in cols:
        x = df[c].dropna().values
        fig, axs = plt.subplots(1, 3, figsize=(12, 3.6), gridspec_kw={"width_ratios": [2.2, 1, 1.6]})
        if by:
            for i, (g, d) in enumerate(df.groupby(by)):
                axs[0].hist(d[c].dropna(), bins="auto", alpha=0.55, color=PALETTE[i % 8], label=str(g))
            axs[0].legend(fontsize=7)
        else:
            axs[0].hist(x, bins="auto", color=PALETTE[0], alpha=0.8, edgecolor="white", density=True)
            xs = np.linspace(x.min(), x.max(), 200)
            axs[0].plot(xs, stats.norm.pdf(xs, x.mean(), x.std(ddof=1)), color=PALETTE[1], lw=2)
        axs[0].set_title(f"{L.fig('Histogram')}: {c}")
        if by:
            axs[1].boxplot([d[c].dropna() for _, d in df.groupby(by)], tick_labels=[str(g) for g, _ in df.groupby(by)])
        else:
            axs[1].boxplot(x, vert=True)
        axs[1].set_title(L.fig("Box plot"))
        (osm, osr), (slope, inter, r) = stats.probplot(x, dist="norm")
        axs[2].plot(osm, osr, "o", ms=3.5, color=PALETTE[0])
        axs[2].plot(osm, slope * np.asarray(osm) + inter, color=PALETTE[3])
        axs[2].set_title(L.fig("Normal Q-Q plot"))
        res.fig(f"{L.t('Distribution')} - {c}", save_fig(fig, outdir, f"dist_{c}"))
        res.summary[c] = _moments(x)
    return res


def ttest(df, params, L: Lang, outdir):
    """params: mode (one|two|paired), column, group (two), column2 (paired), mu0 (one), equal_var (bool), alternative."""
    mode = params.get("mode", "two")
    alt = params.get("alternative", "two-sided")
    res = Result(L.t("t-test"))
    col = params["column"]
    if mode == "one":
        mu0 = float(params.get("mu0", 0))
        x = df[col].dropna()
        t, p = stats.ttest_1samp(x, mu0, alternative=alt)
        ci = stats.t.interval(0.95, len(x) - 1, loc=x.mean(), scale=stats.sem(x))
        res.table(L.t("One-sample t-test"), pd.DataFrame([{"N": len(x), L.t("Mean"): x.mean(), "mu0": mu0, "t": t, "DF": len(x) - 1, "p-value": p, L.t("Lower 95%"): ci[0], L.t("Upper 95%"): ci[1]}]))
        try:
            w = stats.wilcoxon(x - mu0)
            res.table(L.t("Wilcoxon signed-rank"), pd.DataFrame([{L.t("Statistic"): w.statistic, "p-value": w.pvalue}]))
        except Exception:
            pass
        res.summary = {"t": t, "p": p}
    elif mode == "paired":
        d = df[[col, params["column2"]]].dropna()
        t, p = stats.ttest_rel(d[col], d[params["column2"]], alternative=alt)
        diff = d[col] - d[params["column2"]]
        dz = diff.mean() / diff.std(ddof=1)
        ci = stats.t.interval(0.95, len(diff) - 1, loc=diff.mean(), scale=stats.sem(diff))
        res.table(L.t("Paired t-test"), pd.DataFrame([{"N": len(diff), L.t("Mean Diff"): diff.mean(), "t": t, "DF": len(diff) - 1, "p-value": p, L.t("Lower 95%"): ci[0], L.t("Upper 95%"): ci[1], L.t("Cohen's d"): dz}]))
        w = stats.wilcoxon(diff)
        res.table(L.t("Wilcoxon signed-rank"), pd.DataFrame([{L.t("Statistic"): w.statistic, "p-value": w.pvalue}]))
        fig, ax = plt.subplots(figsize=(5, 4))
        for a, b in zip(d[col], d[params["column2"]]):
            ax.plot([0, 1], [a, b], color="#AAAAAA", lw=0.8)
        ax.plot([0, 1], [d[col].mean(), d[params["column2"]].mean()], color=PALETTE[3], lw=3)
        ax.set_xticks([0, 1]); ax.set_xticklabels([col, params["column2"]])
        res.fig(L.t("Paired t-test"), save_fig(fig, outdir, "paired"))
        res.summary = {"t": t, "p": p}
    else:
        g = params["group"]
        lv = sorted(df[g].dropna().unique(), key=str)
        if len(lv) != 2:
            raise ValueError(f"Group column '{g}' must have exactly 2 levels, found {len(lv)}: {lv}")
        a = df.loc[df[g] == lv[0], col].dropna(); b = df.loc[df[g] == lv[1], col].dropna()
        rows = []
        for name, ev in (("Welch (unequal var)", False), ("Pooled (equal var)", True)):
            t, p = stats.ttest_ind(a, b, equal_var=ev, alternative=alt)
            dfree = (len(a) + len(b) - 2) if ev else (a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b)) ** 2 / (
                (a.var(ddof=1) / len(a)) ** 2 / (len(a) - 1) + (b.var(ddof=1) / len(b)) ** 2 / (len(b) - 1))
            rows.append({L.t("Method"): name, L.t("Mean Diff"): a.mean() - b.mean(), "t": t, "DF": dfree, "p-value": p})
        sp = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
        d_eff = (a.mean() - b.mean()) / sp
        res.table(L.t("Two-sample t-test"), pd.DataFrame(rows))
        res.table(L.t("Means and Std Deviations"), pd.DataFrame([
            {L.t("Level"): str(lv[0]), "N": len(a), L.t("Mean"): a.mean(), L.t("Std Dev"): a.std(ddof=1)},
            {L.t("Level"): str(lv[1]), "N": len(b), L.t("Mean"): b.mean(), L.t("Std Dev"): b.std(ddof=1)}]))
        lev = stats.levene(a, b)
        mw = stats.mannwhitneyu(a, b, alternative=alt)
        res.table(L.t("Levene test"), pd.DataFrame([{L.t("Statistic"): lev.statistic, "p-value": lev.pvalue}]))
        res.table(L.t("Mann-Whitney U"), pd.DataFrame([{L.t("Statistic"): mw.statistic, "p-value": mw.pvalue}]))
        res.table(L.t("Effect size"), pd.DataFrame([{L.t("Cohen's d"): d_eff}]))
        fig, ax = plt.subplots(figsize=(5.5, 4))
        ax.boxplot([a, b], tick_labels=[str(lv[0]), str(lv[1])])
        for i, x in enumerate((a, b), 1):
            ax.plot(np.random.default_rng(0).normal(i, 0.04, len(x)), x, "o", ms=3, alpha=.5, color=PALETTE[i - 1])
        ax.set_title(f"{col} {L.t('by')} {g}")
        res.fig(L.t("Two-sample t-test"), save_fig(fig, outdir, "ttest2"))
        res.summary = {"welch_p": rows[0]["p-value"], "pooled_p": rows[1]["p-value"], "cohens_d": d_eff}
    return res


def correlation(df, params, L: Lang, outdir):
    """params: columns (list), method (pearson|spearman|kendall)."""
    cols = params.get("columns") or numeric_cols(df)
    method = params.get("method", "pearson")
    d = df[cols].dropna()
    r = d.corr(method=method)
    pm = pd.DataFrame(np.nan, index=cols, columns=cols)
    for i in cols:
        for j in cols:
            if i != j:
                f = {"pearson": stats.pearsonr, "spearman": stats.spearmanr, "kendall": stats.kendalltau}[method]
                pm.loc[i, j] = f(d[i], d[j])[1]
    res = Result(L.t("Correlation matrix"))
    res.table(f"{L.t('Correlation matrix')} ({method})", r.round(4), index=True)
    res.table("p-value", pm.round(6), index=True)
    fig, ax = plt.subplots(figsize=(1.0 + 0.9 * len(cols), 0.8 + 0.8 * len(cols)))
    im = ax.imshow(r.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, rotation=45, ha="right")
    ax.set_yticks(range(len(cols))); ax.set_yticklabels(cols); ax.grid(False)
    for i in range(len(cols)):
        for j in range(len(cols)):
            ax.text(j, i, f"{r.values[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, shrink=.8)
    res.fig(L.t("Correlation matrix"), save_fig(fig, outdir, "corr_heat"))
    if len(cols) <= 8:
        n = len(cols)
        fig, axs = plt.subplots(n, n, figsize=(2 * n, 2 * n))
        axs = np.atleast_2d(axs)
        for i, a in enumerate(cols):
            for j, b in enumerate(cols):
                ax = axs[i, j]
                if i == j:
                    ax.hist(d[a], bins="auto", color=PALETTE[0], alpha=.8)
                else:
                    ax.plot(d[b], d[a], "o", ms=2, alpha=.5, color=PALETTE[0])
                ax.grid(False)
                if j == 0: ax.set_ylabel(a, fontsize=7)
                if i == n - 1: ax.set_xlabel(b, fontsize=7)
                ax.tick_params(labelsize=6)
        res.fig(L.t("Scatterplot Matrix"), save_fig(fig, outdir, "scatter_matrix"))
    res.summary = {"max_abs_offdiag": float(np.nanmax(np.abs(r.values - np.eye(len(cols)))))}
    return res


def crosstab(df, params, L: Lang, outdir):
    """params: row, col."""
    a, b = params["row"], params["col"]
    ct = pd.crosstab(df[a], df[b])
    chi2, p, dof, exp = stats.chi2_contingency(ct)
    n = ct.values.sum()
    v = np.sqrt(chi2 / (n * (min(ct.shape) - 1)))
    res = Result(L.t("Chi-square test"))
    res.table(f"{L.t('Contingency table')} - {L.t('Observed')}", ct, index=True)
    res.table(f"{L.t('Contingency table')} - {L.t('Expected')}", pd.DataFrame(exp, index=ct.index, columns=ct.columns).round(3), index=True)
    res.table(L.t("Chi-square test"), pd.DataFrame([{"Chi2": chi2, "DF": dof, "p-value": p, L.t("Cramer's V"): v}]))
    if (exp < 5).any():
        res.note("Some expected counts < 5: chi-square approximation may be unreliable; consider Fisher exact test (2x2).")
    if ct.shape == (2, 2):
        o, fp = stats.fisher_exact(ct.values)
        res.table("Fisher exact", pd.DataFrame([{L.t("Odds Ratio"): o, "p-value": fp}]))
    fig, ax = plt.subplots(figsize=(6, 4))
    (ct.div(ct.sum(1), axis=0) * 100).plot(kind="bar", stacked=True, ax=ax, color=PALETTE[:ct.shape[1]])
    ax.set_ylabel("%"); ax.set_title(f"{a} x {b}")
    res.fig(L.t("Contingency table"), save_fig(fig, outdir, "crosstab"))
    res.summary = {"chi2": chi2, "p": p, "cramers_v": v}
    return res


def pca(df, params, L: Lang, outdir):
    """params: columns, n_components (default all), standardize (default True), color_by (optional)."""
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    cols = params.get("columns") or numeric_cols(df)
    d = df[cols + ([params["color_by"]] if params.get("color_by") else [])].dropna()
    X = d[cols].values
    if params.get("standardize", True):
        X = StandardScaler().fit_transform(X)
    k = int(params.get("n_components", len(cols)))
    p = PCA(n_components=k).fit(X)
    sc = p.transform(X)
    ev = p.explained_variance_
    res = Result(L.t("Principal Components"))
    res.table(L.t("Principal Components"), pd.DataFrame({L.t("Eigenvalue"): ev, L.t("Percent"): p.explained_variance_ratio_ * 100,
                                                         L.t("Cum Percent"): np.cumsum(p.explained_variance_ratio_) * 100},
                                                        index=[f"PC{i+1}" for i in range(k)]), index=True)
    load = pd.DataFrame(p.components_.T * np.sqrt(ev), index=cols, columns=[f"PC{i+1}" for i in range(k)])
    res.table(L.t("Loadings"), load.round(4), index=True)
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.plot(range(1, k + 1), ev, "o-", color=PALETTE[0]); ax.axhline(1, color="#999", ls="--")
    ax.set_title(L.fig("Scree plot")); ax.set_xlabel("PC"); ax.set_ylabel(L.fig("Eigenvalue"))
    res.fig(L.t("Scree plot"), save_fig(fig, outdir, "scree"))
    if k >= 2:
        fig, ax = plt.subplots(figsize=(6, 5))
        cb = params.get("color_by")
        if cb:
            for i, g in enumerate(d[cb].unique()):
                m = (d[cb] == g).values
                ax.scatter(sc[m, 0], sc[m, 1], s=18, color=PALETTE[i % 8], label=str(g), alpha=.8)
            ax.legend(fontsize=8)
        else:
            ax.scatter(sc[:, 0], sc[:, 1], s=18, color=PALETTE[0], alpha=.8)
        sf = np.abs(sc[:, :2]).max() / np.abs(load.iloc[:, :2].values).max() * 0.8
        for name, (a, b) in load.iloc[:, :2].iterrows():
            ax.arrow(0, 0, a * sf, b * sf, color=PALETTE[3], head_width=0.05 * sf / 3, alpha=.8)
            ax.text(a * sf * 1.08, b * sf * 1.08, name, color=PALETTE[3], fontsize=8)
        ax.set_xlabel(f"PC1 ({p.explained_variance_ratio_[0]*100:.1f}%)"); ax.set_ylabel(f"PC2 ({p.explained_variance_ratio_[1]*100:.1f}%)")
        ax.set_title(L.fig("Score plot"))
        res.fig(L.t("Score plot"), save_fig(fig, outdir, "pca_scores"))
    res.summary = {"pc1_pct": float(p.explained_variance_ratio_[0] * 100)}
    return res


def cluster(df, params, L: Lang, outdir):
    """params: columns, k (default: chosen by silhouette 2..8), method (kmeans|ward)."""
    from sklearn.cluster import KMeans, AgglomerativeClustering
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import silhouette_score
    cols = params.get("columns") or numeric_cols(df)
    d = df[cols].dropna()
    X = StandardScaler().fit_transform(d.values)
    method = params.get("method", "kmeans")
    k = params.get("k")
    sil = {}
    for kk in range(2, min(9, len(d))):
        lab = (KMeans(kk, n_init=10, random_state=0).fit_predict(X) if method == "kmeans"
               else AgglomerativeClustering(kk, linkage="ward").fit_predict(X))
        sil[kk] = silhouette_score(X, lab)
    k = int(k) if k else max(sil, key=sil.get)
    lab = (KMeans(k, n_init=10, random_state=0).fit_predict(X) if method == "kmeans"
           else AgglomerativeClustering(k, linkage="ward").fit_predict(X))
    out = d.copy(); out[L.t("Cluster")] = lab + 1
    res = Result(L.t("Clusters"))
    res.table("Silhouette by k", pd.DataFrame({"k": list(sil), "silhouette": list(sil.values())}))
    res.table(L.t("Cluster sizes"), out[L.t("Cluster")].value_counts().sort_index().rename("N").reset_index())
    res.table(f"{L.t('Mean')} / {L.t('Cluster')}", out.groupby(L.t("Cluster")).mean().reset_index())
    pc = np.linalg.svd(X - X.mean(0), full_matrices=False)
    s2 = (X - X.mean(0)) @ pc[2].T[:, :2]
    fig, ax = plt.subplots(figsize=(6, 5))
    for i in range(k):
        ax.scatter(s2[lab == i, 0], s2[lab == i, 1], s=18, color=PALETTE[i % 8], label=f"{i+1}")
    ax.legend(title=L.fig("Cluster")); ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    res.fig(L.t("Clusters"), save_fig(fig, outdir, "clusters"))
    if method != "kmeans" or len(d) <= 200:
        from scipy.cluster.hierarchy import linkage, dendrogram
        fig, ax = plt.subplots(figsize=(8, 4))
        dendrogram(linkage(X, "ward"), ax=ax, no_labels=len(d) > 40)
        res.fig(L.t("Dendrogram"), save_fig(fig, outdir, "dendrogram"))
    res.summary = {"k": k, "silhouette": sil.get(k)}
    out.to_csv(f"{outdir}/cluster_assignments.csv", index=False, encoding="utf-8-sig")
    return res
