"""ss_stats2.py - Additional classical tests & agreement analyses (JMP Fit Y by X / Distribution / Multivariate; SPSS Nonparametric, Compare Means, Reliability, GLM Repeated Measures, MANOVA).

Analyses: nonparametric, proportion, variance_test, equivalence, rm_anova, manova, agreement, icc, cronbach,
bland_altman, outliers, tolerance, transform, bootstrap, multiple_testing, tabulate, partial_corr.
Pure numpy/scipy.
"""
from __future__ import annotations
import itertools, math
import numpy as np, pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from ss_common import HE, Result, Lang, save_fig, PALETTE, numeric_cols, get_param

HE.update({
    "Nonparametric tests": "מבחנים א-פרמטריים", "Test of proportions": "מבחן פרופורציות",
    "Tests for equal variances": "מבחני שוויון שונויות", "Equivalence test (TOST)": "מבחן שקילות (TOST)",
    "Repeated measures ANOVA": "ANOVA למדידות חוזרות", "MANOVA": "MANOVA", "Agreement analysis": "ניתוח הסכמה",
    "Intraclass correlation": "מתאם תוך-מחלקתי (ICC)", "Cronbach's alpha": "אלפא של קרונבך",
    "Bland-Altman agreement": "הסכמה בלנד-אלטמן", "Outlier detection": "זיהוי חריגים",
    "Tolerance interval": "תחום סובלנות", "Normalizing transformation": "התמרת נרמול",
    "Bootstrap confidence interval": "רווח סמך בוטסטראפ", "Multiple-testing adjustment": "תיקון לבדיקות מרובות",
    "Tabulate": "טבלת סיכום", "Partial correlation": "מתאם חלקי", "Group": "קבוצה", "Test": "מבחן",
    "Estimate": "אומדן", "Lower 95%": "גבול תחתון 95%", "Upper 95%": "גבול עליון 95%",
    "Decision": "החלטה", "Pairwise comparisons": "השוואות זוגיות", "Difference": "הפרש",
    "Mean difference": "הפרש ממוצעים", "Limits of agreement": "גבולות הסכמה", "Outlier": "חריג",
    "Method": "שיטה", "Adjusted p": "p מתוקן", "Raw p": "p גולמי",
})


def _fp(p):
    return float(p) if p is not None else np.nan


def _holm(p):
    p = np.asarray(p, float); m = len(p); o = np.argsort(p)
    adj = np.empty(m); run = 0.0
    for i, k in enumerate(o):
        run = max(run, (m - i) * p[k]); adj[k] = min(1.0, run)
    return adj


def _bh(p, by=False):
    p = np.asarray(p, float); m = len(p); o = np.argsort(p)[::-1]
    c = np.sum(1 / np.arange(1, m + 1)) if by else 1.0
    adj = np.empty(m); run = 1.0
    for rank, k in zip(range(m, 0, -1), o):
        run = min(run, p[k] * m * c / rank); adj[k] = run
    return adj


def _hochberg(p):
    p = np.asarray(p, float); m = len(p); o = np.argsort(p)[::-1]
    adj = np.empty(m); run = 1.0
    for j, k in enumerate(o):  # j=0 is largest
        run = min(run, (j + 1) * p[k]); adj[k] = run
    return adj


def adjust_p(p, method):
    p = np.asarray(p, float); m = len(p)
    if method == "bonferroni": return np.minimum(1, p * m)
    if method == "sidak": return 1 - (1 - p) ** m
    if method == "holm": return _holm(p)
    if method == "hochberg": return _hochberg(p)
    if method == "bh": return _bh(p)
    if method == "by": return _bh(p, by=True)
    raise ValueError(method)


def _strip(ax, groups, labels):
    ax.boxplot(groups, tick_labels=labels, showfliers=False)
    for i, g in enumerate(groups, 1):
        ax.plot(np.random.default_rng(1).normal(i, 0.05, len(g)), g, "o", ms=3, alpha=.55, color=PALETTE[0])


# ============================================================================
# Nonparametric
# ============================================================================
def nonparametric(df, params, L: Lang, outdir):
    """mode: mannwhitney|wilcoxon|kruskal|friedman|sign|mcnemar|median|runs.
    column, group (mannwhitney/kruskal/median), column2 (wilcoxon paired), columns (friedman), mu0 (wilcoxon/sign/runs center)."""
    mode = params.get("mode", "kruskal")
    res = Result(L.t("Nonparametric tests"))
    col = params.get("column")
    if mode in ("mannwhitney", "kruskal", "median"):
        g = params["group"]
        d = df[[col, g]].dropna()
        lv = sorted(d[g].unique(), key=str)
        groups = [d.loc[d[g] == l, col].values for l in lv]
        fig, ax = plt.subplots(figsize=(5.5, 3.8)); _strip(ax, groups, [str(l) for l in lv]); ax.set_ylabel(col)
        ax.set_title(L.fig("Nonparametric tests"))
        res.fig(f"{col} by {g}", save_fig(fig, outdir, f"np_{mode}_{col}"))
        if mode == "mannwhitney":
            if len(lv) != 2: raise ValueError("mannwhitney needs exactly 2 groups")
            u = stats.mannwhitneyu(groups[0], groups[1], alternative=params.get("alternative", "two-sided"))
            n1, n2 = map(len, groups)
            rb = 1 - 2 * u.statistic / (n1 * n2)  # rank-biserial for group0 vs group1 (U of group 0)
            diffs = np.subtract.outer(groups[0], groups[1]).ravel()
            res.table("Mann-Whitney U", pd.DataFrame([{"Group 1": lv[0], "Group 2": lv[1], "n1": n1, "n2": n2, "U (group 1)": u.statistic,
                      "p-value": u.pvalue, "Rank-biserial r": -rb, "Hodges-Lehmann shift": float(np.median(diffs))}]))
            res.summary.update({"U": float(u.statistic), "p": float(u.pvalue)})
        elif mode == "kruskal":
            h = stats.kruskal(*groups); N = len(d)
            eps2 = h.statistic / ((N ** 2 - 1) / (N + 1))
            res.table("Kruskal-Wallis", pd.DataFrame([{"H": h.statistic, "DF": len(lv) - 1, "p-value": h.pvalue, "Epsilon squared": eps2, "N": N}]))
            res.summary.update({"H": float(h.statistic), "p": float(h.pvalue)})
            ranks = stats.rankdata(d[col].values); d = d.assign(_r=ranks)
            rbar = d.groupby(g)["_r"].mean(); ns = d.groupby(g).size()
            _, cnt = np.unique(d[col].values, return_counts=True)
            T = np.sum(cnt ** 3 - cnt)
            var = N * (N + 1) / 12 - T / (12 * (N - 1))
            pr = []
            for a, b in itertools.combinations(lv, 2):
                z = (rbar[a] - rbar[b]) / math.sqrt(var * (1 / ns[a] + 1 / ns[b]))
                pr.append((a, b, z, 2 * stats.norm.sf(abs(z))))
            if pr:
                pa = _holm([x[3] for x in pr])
                res.table("Dunn post-hoc (Holm)", pd.DataFrame([{"Group A": a, "Group B": b, "z": z, "Raw p": p, "Adjusted p": q} for (a, b, z, p), q in zip(pr, pa)]))
        else:
            m = stats.median_test(*groups)
            res.table("Mood's median test", pd.DataFrame([{"Chi-square": m[0], "p-value": m[1], "Grand median": m[2]}]))
            res.summary.update({"chi2": float(m[0]), "p": float(m[1])})
    elif mode == "wilcoxon":
        if params.get("column2"):
            d = df[[col, params["column2"]]].dropna(); x = (d[col] - d[params["column2"]]).values
        else:
            x = df[col].dropna().values - float(params.get("mu0", 0))
        w = stats.wilcoxon(x, alternative=params.get("alternative", "two-sided"))
        res.table("Wilcoxon signed-rank", pd.DataFrame([{"n (nonzero)": int(np.sum(x != 0)), "W": w.statistic, "p-value": w.pvalue, "Median difference": float(np.median(x))}]))
        fig, ax = plt.subplots(figsize=(5.2, 3.6)); ax.hist(x, bins="auto", color=PALETTE[0], edgecolor="white"); ax.axvline(0, color=PALETTE[3])
        ax.set_title(L.fig("Difference")); res.fig("differences", save_fig(fig, outdir, "np_wilcoxon"))
        res.summary.update({"W": float(w.statistic), "p": float(w.pvalue)})
    elif mode == "sign":
        x = df[col].dropna().values - float(params.get("mu0", 0)); x = x[x != 0]
        bt = stats.binomtest(int((x > 0).sum()), len(x), 0.5)
        res.table("Sign test", pd.DataFrame([{"n (nonzero)": len(x), "Positive": int((x > 0).sum()), "Negative": int((x < 0).sum()), "p-value (exact)": bt.pvalue}]))
        res.summary["p"] = float(bt.pvalue)
    elif mode == "friedman":
        cols = params["columns"]; d = df[cols].dropna()
        f = stats.friedmanchisquare(*[d[c].values for c in cols]); n, k = d.shape
        W = f.statistic / (n * (k - 1))
        res.table("Friedman", pd.DataFrame([{"Chi-square": f.statistic, "DF": k - 1, "p-value": f.pvalue, "Kendall W": W, "n": n}]))
        pr = [(a, b, stats.wilcoxon(d[a], d[b]).pvalue) for a, b in itertools.combinations(cols, 2)]
        pa = _holm([x[2] for x in pr])
        res.table("Pairwise Wilcoxon (Holm)", pd.DataFrame([{"A": a, "B": b, "Raw p": p, "Adjusted p": q} for (a, b, p), q in zip(pr, pa)]))
        fig, ax = plt.subplots(figsize=(6, 3.8)); _strip(ax, [d[c].values for c in cols], cols); ax.set_title(L.fig("Nonparametric tests"))
        res.fig("friedman", save_fig(fig, outdir, "np_friedman"))
        res.summary.update({"chi2": float(f.statistic), "p": float(f.pvalue), "W": float(W)})
    elif mode == "mcnemar":
        c2 = params["column2"]; d = df[[col, c2]].dropna()
        lv = sorted(set(d[col]) | set(d[c2]), key=str)
        tab = pd.crosstab(d[col], d[c2]).reindex(index=lv, columns=lv, fill_value=0)
        b, c = int(tab.iloc[0, 1]), int(tab.iloc[1, 0])
        exact = stats.binomtest(b, b + c, 0.5).pvalue if b + c else np.nan
        chi = (abs(b - c) - 1) ** 2 / (b + c) if b + c else np.nan
        res.table("2x2 table", tab, index=True)
        res.table("McNemar", pd.DataFrame([{"b (A=1,B=2)": b, "c (A=2,B=1)": c, "Chi-square (cc)": chi, "p (chi-square)": stats.chi2.sf(chi, 1) if b + c else np.nan, "p (exact)": exact}]))
        res.summary.update({"p_exact": _fp(exact)})
    elif mode == "runs":
        x = df[col].dropna().values; c0 = float(params.get("mu0", np.median(x)))
        s = x[x != c0] > c0; n1, n2 = int(s.sum()), int((~s).sum()); runs = 1 + int(np.sum(s[1:] != s[:-1]))
        mu = 2 * n1 * n2 / (n1 + n2) + 1; var = 2 * n1 * n2 * (2 * n1 * n2 - n1 - n2) / ((n1 + n2) ** 2 * (n1 + n2 - 1))
        z = (runs - mu) / math.sqrt(var)
        res.table("Runs test", pd.DataFrame([{"Cut value": c0, "n above": n1, "n below": n2, "Runs": runs, "Expected": mu, "z": z, "p-value": 2 * stats.norm.sf(abs(z))}]))
        fig, ax = plt.subplots(figsize=(7, 3.2)); ax.plot(x, "o-", ms=3); ax.axhline(c0, color=PALETTE[3]); ax.set_title(L.fig("Nonparametric tests"))
        res.fig("runs", save_fig(fig, outdir, "np_runs")); res.summary.update({"runs": runs, "z": float(z)})
    else:
        raise ValueError("Unknown mode " + mode)
    return res


# ============================================================================
# Proportions
# ============================================================================
def _wilson(x, n, a=0.05):
    z = stats.norm.isf(a / 2); p = x / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n); h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return c - h, c + h


def _cp(x, n, a=0.05):
    lo = 0.0 if x == 0 else stats.beta.ppf(a / 2, x, n - x + 1)
    hi = 1.0 if x == n else stats.beta.ppf(1 - a / 2, x + 1, n - x)
    return lo, hi


def proportion(df, params, L: Lang, outdir):
    """One-sample: (column+success) or (x,n); p0. Two-sample: (column+group+success) or (x,n,x2,n2)."""
    res = Result(L.t("Test of proportions"))
    a = 1 - float(params.get("conf", 0.95))
    if "x" in params:
        x, n = int(params["x"]), int(params["n"]); x2 = params.get("x2"); n2 = params.get("n2")
        labels = ["Sample 1", "Sample 2"]
    else:
        col = params["column"]; succ = params["success"]
        if params.get("group"):
            d = df[[col, params["group"]]].dropna(); lv = sorted(d[params["group"]].unique(), key=str)
            if len(lv) != 2: raise ValueError("group must have 2 levels")
            sel = [d[d[params["group"]] == l][col].astype(str) == str(succ) for l in lv]
            x, n, x2, n2 = int(sel[0].sum()), len(sel[0]), int(sel[1].sum()), len(sel[1]); labels = [str(l) for l in lv]
        else:
            s = df[col].dropna().astype(str) == str(succ); x, n, x2, n2 = int(s.sum()), len(s), None, None; labels = [col]
    rows = []
    for lab, xx, nn in [(labels[0], x, n)] + ([(labels[1], x2, n2)] if x2 is not None else []):
        p = xx / nn
        rows.append({"Sample": lab, "x": xx, "n": nn, "Proportion": p, "Wilson low": _wilson(xx, nn, a)[0], "Wilson high": _wilson(xx, nn, a)[1],
                     "Exact low": _cp(xx, nn, a)[0], "Exact high": _cp(xx, nn, a)[1]})
    res.table("Proportions & confidence intervals", pd.DataFrame(rows))
    if x2 is None:
        p0 = float(params.get("p0", 0.5))
        bt = stats.binomtest(x, n, p0, alternative=params.get("alternative", "two-sided"))
        z = (x / n - p0) / math.sqrt(p0 * (1 - p0) / n)
        res.table("One-sample test", pd.DataFrame([{"p0": p0, "p (exact binomial)": bt.pvalue, "z (normal approx.)": z, "p (z)": 2 * stats.norm.sf(abs(z))}]))
        res.summary.update({"p_hat": x / n, "p_exact": float(bt.pvalue)})
    else:
        p1, p2 = x / n, x2 / n2; pp = (x + x2) / (n + n2)
        z = (p1 - p2) / math.sqrt(pp * (1 - pp) * (1 / n + 1 / n2))
        fe = stats.fisher_exact([[x, n - x], [x2, n2 - x2]])
        l1, u1 = _wilson(x, n, a); l2, u2 = _wilson(x2, n2, a)
        d = p1 - p2   # Newcombe hybrid score
        lo = d - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2); hi = d + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)
        rr = p1 / p2 if p2 > 0 else np.nan
        orr = fe[0]
        res.table("Two-sample test", pd.DataFrame([{"Difference": d, "Lower 95%": lo, "Upper 95%": hi, "z (pooled)": z, "p (z)": 2 * stats.norm.sf(abs(z)),
                  "p (Fisher exact)": fe[1], "Risk ratio": rr, "Odds ratio": orr}]))
        res.summary.update({"diff": d, "p_z": float(2 * stats.norm.sf(abs(z))), "p_fisher": float(fe[1])})
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    for i, r in enumerate(rows):
        ax.errorbar(r["Proportion"], i, xerr=[[r["Proportion"] - r["Exact low"]], [r["Exact high"] - r["Proportion"]]], fmt="o", color=PALETTE[0], capsize=4)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r["Sample"] for r in rows]); ax.set_xlim(0, 1); ax.set_title(L.fig("Test of proportions"))
    res.fig("proportions", save_fig(fig, outdir, "proportions"))
    return res


# ============================================================================
# Equal variances
# ============================================================================
def variance_test(df, params, L: Lang, outdir):
    """params: column, group. Levene, Brown-Forsythe, Bartlett, Fligner-Killeen (+ F test and ratio CI for 2 groups)."""
    col, g = params["column"], params["group"]
    d = df[[col, g]].dropna(); lv = sorted(d[g].unique(), key=str); groups = [d.loc[d[g] == l, col].values for l in lv]
    res = Result(L.t("Tests for equal variances"))
    res.table("Group statistics", pd.DataFrame([{"Group": l, "N": len(x), "Mean": x.mean(), "Std Dev": x.std(ddof=1)} for l, x in zip(lv, groups)]))
    lev = stats.levene(*groups, center="mean"); bf = stats.levene(*groups, center="median"); bar = stats.bartlett(*groups); fl = stats.fligner(*groups)
    rows = [{"Test": "Levene (mean)", "Statistic": lev.statistic, "p-value": lev.pvalue}, {"Test": "Brown-Forsythe (median)", "Statistic": bf.statistic, "p-value": bf.pvalue},
            {"Test": "Bartlett (assumes normality)", "Statistic": bar.statistic, "p-value": bar.pvalue}, {"Test": "Fligner-Killeen", "Statistic": fl.statistic, "p-value": fl.pvalue}]
    if len(lv) == 2:
        f = groups[0].var(ddof=1) / groups[1].var(ddof=1); d1, d2 = len(groups[0]) - 1, len(groups[1]) - 1
        p = 2 * min(stats.f.cdf(f, d1, d2), stats.f.sf(f, d1, d2))
        rows.append({"Test": "F test (2 variances)", "Statistic": f, "p-value": p})
        res.summary["ratio_ci"] = [float(f / stats.f.ppf(.975, d1, d2)), float(f / stats.f.ppf(.025, d1, d2))]
    res.table("Tests", pd.DataFrame(rows)); res.summary.update({"levene_p": float(lev.pvalue), "bartlett_p": float(bar.pvalue)})
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    sd = np.array([x.std(ddof=1) for x in groups]); df_ = np.array([len(x) - 1 for x in groups])
    lo = sd * np.sqrt(df_ / stats.chi2.ppf(.975, df_)); hi = sd * np.sqrt(df_ / stats.chi2.ppf(.025, df_))
    ax.errorbar(range(len(lv)), sd, yerr=[sd - lo, hi - sd], fmt="o", capsize=4, color=PALETTE[0]); ax.set_xticks(range(len(lv))); ax.set_xticklabels([str(l) for l in lv])
    ax.set_ylabel(L.fig("Std Dev")); ax.set_title(L.fig("Tests for equal variances")); res.fig("std dev 95% CI", save_fig(fig, outdir, "var_test"))
    return res


# ============================================================================
# Equivalence TOST
# ============================================================================
def equivalence(df, params, L: Lang, outdir):
    """mode one|two|paired; column; group | column2; low/high margins (difference) ; alpha; mu0 for one."""
    mode = params.get("mode", "two"); a = float(params.get("alpha", 0.05)); col = params["column"]
    low, high = float(params["low"]), float(params["high"])
    if mode == "one":
        x = df[col].dropna().values; diff = x.mean() - float(params.get("mu0", 0)); se = x.std(ddof=1) / math.sqrt(len(x)); dof = len(x) - 1
    elif mode == "paired":
        d = df[[col, params["column2"]]].dropna(); x = (d[col] - d[params["column2"]]).values; diff = x.mean(); se = x.std(ddof=1) / math.sqrt(len(x)); dof = len(x) - 1
    else:
        g = params["group"]; d = df[[col, g]].dropna(); lv = sorted(d[g].unique(), key=str)
        x1, x2 = d.loc[d[g] == lv[0], col].values, d.loc[d[g] == lv[1], col].values
        v1, v2 = x1.var(ddof=1) / len(x1), x2.var(ddof=1) / len(x2); se = math.sqrt(v1 + v2); diff = x1.mean() - x2.mean()
        dof = (v1 + v2) ** 2 / (v1 ** 2 / (len(x1) - 1) + v2 ** 2 / (len(x2) - 1))
    t1 = (diff - low) / se; t2 = (diff - high) / se
    p1 = stats.t.sf(t1, dof); p2 = stats.t.cdf(t2, dof); p = max(p1, p2)
    tc = stats.t.ppf(1 - a, dof); ci = (diff - tc * se, diff + tc * se)
    res = Result(L.t("Equivalence test (TOST)"))
    res.table("TOST", pd.DataFrame([{"Difference": diff, "Std Error": se, "DF": dof, "Lower margin": low, "Upper margin": high, "p (lower test)": p1, "p (upper test)": p2,
                                     "TOST p": p, f"{int((1-2*a)*100)}% CI low": ci[0], f"{int((1-2*a)*100)}% CI high": ci[1],
                                     "Decision": "Equivalent" if p < a else "Not shown equivalent"}]))
    fig, ax = plt.subplots(figsize=(6, 2.8)); ax.axvspan(low, high, color="#D9EAD3", alpha=.7)
    ax.errorbar(diff, 0, xerr=[[diff - ci[0]], [ci[1] - diff]], fmt="o", capsize=5, color=PALETTE[0]); ax.axvline(0, color="grey", lw=.8)
    ax.set_yticks([]); ax.set_title(L.fig("Equivalence test (TOST)")); res.fig("TOST", save_fig(fig, outdir, "tost"))
    res.summary.update({"tost_p": float(p), "diff": float(diff), "equivalent": bool(p < a)})
    return res


# ============================================================================
# Repeated measures ANOVA
# ============================================================================
def rm_anova(df, params, L: Lang, outdir):
    """columns: list of repeated-measure columns (wide format; one row per subject)."""
    cols = params["columns"]; d = df[cols].dropna(); Y = d.values; n, k = Y.shape
    gm = Y.mean(); ss_c = n * ((Y.mean(0) - gm) ** 2).sum(); ss_s = k * ((Y.mean(1) - gm) ** 2).sum(); ss_t = ((Y - gm) ** 2).sum(); ss_e = ss_t - ss_c - ss_s
    df_c, df_e = k - 1, (n - 1) * (k - 1); F = (ss_c / df_c) / (ss_e / df_e); p = stats.f.sf(F, df_c, df_e)
    S = np.cov(Y.T)
    # orthonormal contrasts
    Cm = np.linalg.svd(np.eye(k) - 1.0 / k)[0][:, :k - 1].T
    Sc = Cm @ S @ Cm.T
    if k > 2:
        eg = np.trace(Sc) ** 2 / ((k - 1) * np.trace(Sc @ Sc)); eh = min(1.0, (n * (k - 1) * eg - 2) / ((k - 1) * (n - 1 - (k - 1) * eg)))
        pp = k - 1; W = np.linalg.det(Sc) / (np.trace(Sc) / pp) ** pp
        dd = 1 - (2 * pp * pp + pp + 2) / (6 * pp * (n - 1)); chi = -(n - 1) * dd * math.log(W); mdf = pp * (pp + 1) / 2 - 1
        mp = stats.chi2.sf(chi, mdf)
    else:
        eg = eh = 1.0; W = chi = np.nan; mdf = 0; mp = np.nan
    res = Result(L.t("Repeated measures ANOVA"))
    res.table("Within-subjects effects", pd.DataFrame([
        {"Source": "Condition", "Correction": "Sphericity assumed", "DF": df_c, "Sum of Squares": ss_c, "Mean Square": ss_c / df_c, "F Ratio": F, "Prob > F": p},
        {"Source": "Condition", "Correction": "Greenhouse-Geisser", "DF": df_c * eg, "Sum of Squares": ss_c, "Mean Square": ss_c / (df_c * eg), "F Ratio": F, "Prob > F": stats.f.sf(F, df_c * eg, df_e * eg)},
        {"Source": "Condition", "Correction": "Huynh-Feldt", "DF": df_c * eh, "Sum of Squares": ss_c, "Mean Square": ss_c / (df_c * eh), "F Ratio": F, "Prob > F": stats.f.sf(F, df_c * eh, df_e * eh)},
        {"Source": "Error", "Correction": "", "DF": df_e, "Sum of Squares": ss_e, "Mean Square": ss_e / df_e, "F Ratio": np.nan, "Prob > F": np.nan}]))
    res.table("Sphericity", pd.DataFrame([{"Mauchly W": W, "Chi-square": chi, "DF": mdf, "p-value": mp, "Epsilon (GG)": eg, "Epsilon (HF)": eh, "Partial eta squared": ss_c / (ss_c + ss_e)}]))
    res.table("Condition means", pd.DataFrame({"Condition": cols, "N": n, "Mean": Y.mean(0), "Std Dev": Y.std(0, ddof=1)}))
    pr = [(a, b, stats.ttest_rel(d[a], d[b])) for a, b in itertools.combinations(cols, 2)]
    pa = _holm([t.pvalue for _, _, t in pr])
    res.table("Pairwise comparisons", pd.DataFrame([{"A": a, "B": b, "Mean difference": (d[a] - d[b]).mean(), "t": t.statistic, "Raw p": t.pvalue, "Adjusted p": q} for (a, b, t), q in zip(pr, pa)]))
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    for r in Y: ax[0].plot(range(k), r, color="#BBBBBB", lw=.7)
    ax[0].errorbar(range(k), Y.mean(0), yerr=stats.sem(Y) * stats.t.ppf(.975, n - 1), fmt="o-", color=PALETTE[0], lw=2, capsize=4)
    ax[0].set_xticks(range(k)); ax[0].set_xticklabels(cols); ax[0].set_title(L.fig("Repeated measures ANOVA"))
    _strip(ax[1], [Y[:, i] for i in range(k)], cols); res.fig("profile plot", save_fig(fig, outdir, "rm_anova"))
    res.summary.update({"F": float(F), "p": float(p), "eps_gg": float(eg), "p_gg": float(stats.f.sf(F, df_c * eg, df_e * eg))})
    return res


# ============================================================================
# MANOVA
# ============================================================================
def manova(df, params, L: Lang, outdir):
    """responses: list; factor: one grouping column.  Wilks, Pillai, Hotelling-Lawley, Roy + univariate ANOVAs."""
    ys, g = params["responses"], params["factor"]
    d = df[ys + [g]].dropna(); lv = sorted(d[g].unique(), key=str); p = len(ys); q = len(lv) - 1; N = len(d); v = N - len(lv)
    Y = d[ys].values; gm = Y.mean(0); E = np.zeros((p, p)); H = np.zeros((p, p))
    for l in lv:
        Yg = Y[(d[g] == l).values]; m = Yg.mean(0); E += (Yg - m).T @ (Yg - m); H += len(Yg) * np.outer(m - gm, m - gm)
    lam = np.sort(np.real(np.linalg.eigvals(np.linalg.solve(E, H))))[::-1]; lam = np.clip(lam, 0, None)
    Wl = float(np.prod(1 / (1 + lam))); V = float(np.sum(lam / (1 + lam))); U = float(lam.sum()); R = float(lam[0])
    s = min(p, q); m_ = (abs(p - q) - 1) / 2; n_ = (v - p - 1) / 2
    den = p * p + q * q - 5; t = math.sqrt((p * p * q * q - 4) / den) if den > 0 else 1.0
    mm = v - (p - q + 1) / 2; d1w = p * q; d2w = mm * t - (p * q - 2) / 2
    Fw = (1 - Wl ** (1 / t)) / Wl ** (1 / t) * d2w / d1w
    d1p, d2p = s * (2 * m_ + s + 1), s * (2 * n_ + s + 1); Fp = (2 * n_ + s + 1) / (2 * m_ + s + 1) * V / (s - V)
    d1h, d2h = s * (2 * m_ + s + 1), 2 * (s * n_ + 1); Fh = 2 * (s * n_ + 1) * U / (s * s * (2 * m_ + s + 1))
    dm = max(p, q); d2r = v - dm + q; Fr = R * d2r / dm
    res = Result(L.t("MANOVA"))
    res.table("Multivariate tests", pd.DataFrame([
        {"Test": "Wilks' Lambda", "Value": Wl, "F": Fw, "Num DF": d1w, "Den DF": d2w, "Prob > F": stats.f.sf(Fw, d1w, d2w)},
        {"Test": "Pillai's Trace", "Value": V, "F": Fp, "Num DF": d1p, "Den DF": d2p, "Prob > F": stats.f.sf(Fp, d1p, d2p)},
        {"Test": "Hotelling-Lawley Trace", "Value": U, "F": Fh, "Num DF": d1h, "Den DF": d2h, "Prob > F": stats.f.sf(Fh, d1h, d2h)},
        {"Test": "Roy's Largest Root (upper bound)", "Value": R, "F": Fr, "Num DF": dm, "Den DF": d2r, "Prob > F": stats.f.sf(Fr, dm, d2r)}]))
    un = []
    for c in ys:
        r = stats.f_oneway(*[d.loc[d[g] == l, c] for l in lv]); un.append({"Response": c, "F": r.statistic, "Prob > F": r.pvalue})
    res.table("Univariate ANOVAs", pd.DataFrame(un))
    res.table("Group means", d.groupby(g)[ys].mean().reset_index())
    # canonical variates
    w, vec = np.linalg.eig(np.linalg.solve(E / v, H)); o = np.argsort(-np.real(w)); vec = np.real(vec[:, o])
    Z = (Y - gm) @ vec[:, :min(2, vec.shape[1])]
    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    for i, l in enumerate(lv):
        mk = (d[g] == l).values; ax.scatter(Z[mk, 0], Z[mk, 1] if Z.shape[1] > 1 else np.zeros(mk.sum()), s=18, color=PALETTE[i % 8], label=str(l), alpha=.8)
    ax.legend(fontsize=7); ax.set_xlabel("Canonical 1"); ax.set_ylabel("Canonical 2"); ax.set_title(L.fig("MANOVA")); res.fig("canonical plot", save_fig(fig, outdir, "manova"))
    res.summary.update({"wilks": Wl, "wilks_F": float(Fw), "wilks_p": float(stats.f.sf(Fw, d1w, d2w)), "pillai": V})
    res.note("Roy's largest root F is an upper bound on the true p-value; Wilks/Pillai F are the standard exact/approximate F forms.")
    return res


# ============================================================================
# Agreement: kappa, Fleiss, attribute MSA
# ============================================================================
def _kappa2(a, b, weights=None):
    lv = sorted(set(a) | set(b), key=str); k = len(lv); idx = {l: i for i, l in enumerate(lv)}
    M = np.zeros((k, k))
    for x, y in zip(a, b): M[idx[x], idx[y]] += 1
    n = M.sum(); P = M / n; r, c = P.sum(1), P.sum(0)
    if weights == "linear": Wt = 1 - np.abs(np.subtract.outer(range(k), range(k))) / (k - 1)
    elif weights == "quadratic": Wt = 1 - (np.subtract.outer(range(k), range(k)) / (k - 1)) ** 2
    else: Wt = np.eye(k)
    po = (Wt * P).sum(); pe = (Wt * np.outer(r, c)).sum(); kap = (po - pe) / (1 - pe)
    if weights is None:  # Fleiss, Cohen & Everitt large-sample SE
        po_ = np.trace(P); pe_ = (r * c).sum()
        A = sum(P[i, i] * (1 - (r[i] + c[i]) * (1 - kap)) ** 2 for i in range(k))
        B = (1 - kap) ** 2 * sum(P[i, j] * (c[i] + r[j]) ** 2 for i in range(k) for j in range(k) if i != j)
        C = (kap - pe_ * (1 - kap)) ** 2
        se = math.sqrt((A + B - C) / n) / (1 - pe_)
    else:
        se = math.sqrt(po * (1 - po) / n) / (1 - pe)
    return kap, se, po, M, lv


def _fleiss(R):  # R: n subjects x k raters (categories)
    n, m = R.shape; lv = sorted(set(R.ravel()), key=str); idx = {l: i for i, l in enumerate(lv)}
    C = np.zeros((n, len(lv)))
    for i in range(n):
        for x in R[i]: C[i, idx[x]] += 1
    pj = C.sum(0) / (n * m); Pi = ((C ** 2).sum(1) - m) / (m * (m - 1)); Pb = Pi.mean(); Pe = (pj ** 2).sum()
    return (Pb - Pe) / (1 - Pe), Pb, Pe


def agreement(df, params, L: Lang, outdir):
    """mode kappa: rater1, rater2 [weights linear|quadratic].  mode fleiss: raters (list).
    mode attribute: appraiser, sample, rating, [standard], [positive level]."""
    mode = params.get("mode", "kappa"); res = Result(L.t("Agreement analysis"))
    if mode == "kappa":
        d = df[[params["rater1"], params["rater2"]]].dropna()
        kap, se, po, M, lv = _kappa2(d.iloc[:, 0].astype(str).tolist(), d.iloc[:, 1].astype(str).tolist(), params.get("weights"))
        res.table("Cross-classification", pd.DataFrame(M.astype(int), index=lv, columns=lv), index=True)
        res.table("Cohen's kappa", pd.DataFrame([{"Kappa": kap, "Std Error": se, "Lower 95%": kap - 1.96 * se, "Upper 95%": kap + 1.96 * se, "Observed agreement": po, "N": int(M.sum()), "Weights": params.get("weights") or "none"}]))
        res.summary.update({"kappa": float(kap), "po": float(po)})
        fig, ax = plt.subplots(figsize=(4.6, 4)); im = ax.imshow(M, cmap="Blues"); ax.set_xticks(range(len(lv))); ax.set_xticklabels(lv); ax.set_yticks(range(len(lv))); ax.set_yticklabels(lv)
        for i in range(len(lv)):
            for j in range(len(lv)): ax.text(j, i, int(M[i, j]), ha="center", va="center")
        ax.grid(False); ax.set_title(L.fig("Agreement analysis")); res.fig("agreement matrix", save_fig(fig, outdir, "kappa"))
    elif mode == "fleiss":
        d = df[params["raters"]].dropna().astype(str).values; k, Pb, Pe = _fleiss(d)
        res.table("Fleiss' kappa", pd.DataFrame([{"Kappa": k, "Observed agreement": Pb, "Expected agreement": Pe, "Subjects": d.shape[0], "Raters": d.shape[1]}])); res.summary["kappa"] = float(k)
    else:
        ap, sm, rt = params["appraiser"], params["sample"], params["rating"]; std = params.get("standard")
        cols = [ap, sm, rt] + ([std] if std else []); d = df[cols].dropna().copy(); d[rt] = d[rt].astype(str)
        if std: d[std] = d[std].astype(str)
        samples = sorted(d[sm].unique(), key=str); apps = sorted(d[ap].unique(), key=str)
        ci = lambda x, n: _cp(x, n)
        rows_w, rows_s = [], []
        for a in apps:
            da = d[d[ap] == a]; ok = 0; oks = 0; tot = 0
            for s in samples:
                r = da[da[sm] == s][rt]
                if len(r) == 0: continue
                tot += 1; ok += int(r.nunique() == 1)
                if std: oks += int((r == da[da[sm] == s][std].iloc[0]).all())
            lo, hi = ci(ok, tot); rows_w.append({"Appraiser": a, "Samples": tot, "Matched (all trials agree)": ok, "Percent": 100 * ok / tot, "95% CI low": 100 * lo, "95% CI high": 100 * hi})
            if std:
                lo, hi = ci(oks, tot); rows_s.append({"Appraiser": a, "Samples": tot, "Matched to standard": oks, "Percent": 100 * oks / tot, "95% CI low": 100 * lo, "95% CI high": 100 * hi})
        res.table("Within appraiser", pd.DataFrame(rows_w))
        if std: res.table("Each appraiser vs standard", pd.DataFrame(rows_s))
        allag = 0; allstd = 0
        for s in samples:
            r = d[d[sm] == s]; allag += int(r[rt].nunique() == 1)
            if std: allstd += int((r[rt] == r[std].iloc[0]).all())
        lo, hi = ci(allag, len(samples))
        res.table("Between appraisers", pd.DataFrame([{"Samples": len(samples), "All appraisers agree": allag, "Percent": 100 * allag / len(samples), "95% CI low": 100 * lo, "95% CI high": 100 * hi}]))
        res.summary["between_pct"] = 100 * allag / len(samples)
        if std:
            lo, hi = ci(allstd, len(samples))
            res.table("All appraisers vs standard", pd.DataFrame([{"Samples": len(samples), "All match standard": allstd, "Percent": 100 * allstd / len(samples), "95% CI low": 100 * lo, "95% CI high": 100 * hi}]))
            ks = []
            for a in apps:
                da = d[d[ap] == a]; kap, se, po, _, _ = _kappa2(da[rt].tolist(), da[std].tolist()); ks.append({"Appraiser": a, "Kappa vs standard": kap, "Std Error": se})
            res.table("Kappa vs standard", pd.DataFrame(ks))
            pos = params.get("positive")
            if pos is not None:
                pos = str(pos); rows = []
                for a in apps:
                    da = d[d[ap] == a]; P = da[da[std] == pos]; N_ = da[da[std] != pos]
                    miss = (P[rt] != pos).mean() if len(P) else np.nan; fa = (N_[rt] == pos).mean() if len(N_) else np.nan
                    rows.append({"Appraiser": a, "Miss rate (false negative)": miss, "False-alarm rate (false positive)": fa, "Effectiveness": (da[rt] == da[std]).mean()})
                res.table("Miss / false-alarm rates", pd.DataFrame(rows))
            res.summary["all_vs_std_pct"] = 100 * allstd / len(samples)
        fig, ax = plt.subplots(figsize=(6, 3.6)); x = np.arange(len(apps)); ax.bar(x - .2, [r["Percent"] for r in rows_w], .4, label="within", color=PALETTE[0])
        if std: ax.bar(x + .2, [r["Percent"] for r in rows_s], .4, label="vs standard", color=PALETTE[1])
        ax.set_xticks(x); ax.set_xticklabels(apps); ax.set_ylim(0, 105); ax.set_ylabel("%"); ax.legend(); ax.set_title(L.fig("Agreement analysis")); res.fig("attribute agreement", save_fig(fig, outdir, "attr_agreement"))
        res.note("Agreement thresholds (e.g. kappa > 0.9 / >= 0.75) are common conventions (AIAG MSA 4th ed. suggests kappa > 0.75 acceptable), not universal standards.")
    return res


def _icc_table(Y):
    n, k = Y.shape; gm = Y.mean(); ssr = k * ((Y.mean(1) - gm) ** 2).sum(); ssc = n * ((Y.mean(0) - gm) ** 2).sum(); sst = ((Y - gm) ** 2).sum()
    sse = sst - ssr - ssc; msr, msc, mse = ssr / (n - 1), ssc / (k - 1), sse / ((n - 1) * (k - 1)); msw = (ssc + sse) / (n * (k - 1))
    out = {"ICC(1,1)": (msr - msw) / (msr + (k - 1) * msw), "ICC(2,1)": (msr - mse) / (msr + (k - 1) * mse + k * (msc - mse) / n),
           "ICC(3,1)": (msr - mse) / (msr + (k - 1) * mse), "ICC(1,k)": (msr - msw) / msr,
           "ICC(2,k)": (msr - mse) / (msr + (msc - mse) / n), "ICC(3,k)": (msr - mse) / msr}
    a = 0.05; ci = {}
    F1 = msr / msw; d1, d2 = n - 1, n * (k - 1)
    fl, fu = F1 / stats.f.ppf(1 - a / 2, d1, d2), F1 * stats.f.ppf(1 - a / 2, d2, d1)
    ci["ICC(1,1)"] = ((fl - 1) / (fl + k - 1), (fu - 1) / (fu + k - 1)); ci["ICC(1,k)"] = (1 - 1 / fl, 1 - 1 / fu)
    F3 = msr / mse; d2 = (n - 1) * (k - 1)
    fl, fu = F3 / stats.f.ppf(1 - a / 2, n - 1, d2), F3 * stats.f.ppf(1 - a / 2, d2, n - 1)
    ci["ICC(3,1)"] = ((fl - 1) / (fl + k - 1), (fu - 1) / (fu + k - 1)); ci["ICC(3,k)"] = (1 - 1 / fl, 1 - 1 / fu)
    # ICC(2,1): McGraw & Wong (1996) Satterthwaite approximation
    r = out["ICC(2,1)"]
    a_ = k * r / (n * (1 - r)); b_ = 1 + k * r * (n - 1) / (n * (1 - r))
    v = (a_ * msc + b_ * mse) ** 2 / ((a_ * msc) ** 2 / (k - 1) + (b_ * mse) ** 2 / ((n - 1) * (k - 1)))
    Fs = stats.f.ppf(1 - a / 2, n - 1, v); Fl = stats.f.ppf(1 - a / 2, v, n - 1)
    ci["ICC(2,1)"] = ((n * (msr - Fs * mse)) / (Fs * (k * msc + (k * n - k - n) * mse) + n * msr), (n * (Fl * msr - mse)) / (k * msc + (k * n - k - n) * mse + n * Fl * msr))
    ci["ICC(2,k)"] = (np.nan, np.nan)
    return out, ci, (msr, msc, mse, msw)


def icc(df, params, L: Lang, outdir):
    """columns: raters/measurements (wide: one row per subject/target)."""
    cols = params["columns"]; Y = df[cols].dropna().values; out, ci, ms = _icc_table(Y)
    res = Result(L.t("Intraclass correlation"))
    res.table("ICC (Shrout & Fleiss)", pd.DataFrame([{"Type": k, "ICC": v, "Lower 95%": ci[k][0], "Upper 95%": ci[k][1]} for k, v in out.items()]))
    res.table("Mean squares", pd.DataFrame([{"MS rows (targets)": ms[0], "MS columns (raters)": ms[1], "MS error": ms[2], "MS within": ms[3], "n": Y.shape[0], "k": Y.shape[1]}]))
    res.summary.update({k: float(v) for k, v in out.items()})
    fig, ax = plt.subplots(figsize=(6, 3.8))
    for r in Y: ax.plot(range(Y.shape[1]), r, "o-", color="#AAAAAA", ms=3, lw=.7)
    ax.set_xticks(range(Y.shape[1])); ax.set_xticklabels(cols); ax.set_title(L.fig("Intraclass correlation")); res.fig("targets across raters", save_fig(fig, outdir, "icc"))
    res.note("ICC(1)=one-way random; ICC(2)=two-way random (absolute agreement); ICC(3)=two-way mixed (consistency). (k) = mean of k raters. ICC(2,k) CI not computed.")
    return res


def cronbach(df, params, L: Lang, outdir):
    """params: columns (item columns; default all numeric). Alpha, standardized alpha, corrected item-total r, alpha if item deleted."""
    cols = params.get("columns") or numeric_cols(df); d = df[cols].dropna(); k = len(cols)
    def alpha(x):
        v = x.var(ddof=1).sum(); t = x.sum(1).var(ddof=1); return k_(x) / (k_(x) - 1) * (1 - v / t)
    k_ = lambda x: x.shape[1]
    a = alpha(d); R = d.corr().values; rbar = (R.sum() - k) / (k * (k - 1)); astd = k * rbar / (1 + (k - 1) * rbar)
    rows = []
    for c in cols:
        rest = d.drop(columns=c); rows.append({"Item": c, "Mean": d[c].mean(), "Std Dev": d[c].std(), "Corrected item-total r": np.corrcoef(d[c], rest.sum(1))[0, 1],
                                              "Alpha if item deleted": alpha(rest) if k > 2 else np.nan})
    res = Result(L.t("Cronbach's alpha"))
    res.table("Reliability", pd.DataFrame([{"Cronbach alpha": a, "Standardized alpha": astd, "Items": k, "N": len(d), "Mean inter-item r": rbar}]))
    res.table("Item statistics", pd.DataFrame(rows)); res.summary.update({"alpha": float(a), "alpha_std": float(astd)})
    fig, ax = plt.subplots(figsize=(6, 3.6)); ax.bar(cols, [r["Alpha if item deleted"] for r in rows], color=PALETTE[0]); ax.axhline(a, color=PALETTE[3]); ax.set_title(L.fig("Cronbach's alpha"))
    res.fig("alpha if deleted", save_fig(fig, outdir, "cronbach"))
    res.note("Cutoffs such as alpha >= 0.7 are conventions, not standards.")
    return res


def bland_altman(df, params, L: Lang, outdir):
    """params: column (method A), column2 (method B). Bias, 95% limits of agreement, proportional-bias slope."""
    a, b = params["column"], params["column2"]; d = df[[a, b]].dropna(); x, y = d[a].values, d[b].values
    diff = x - y; mean = (x + y) / 2; n = len(d); md, sd = diff.mean(), diff.std(ddof=1)
    tc = stats.t.ppf(.975, n - 1); se_loa = sd * math.sqrt(1 / n + 1.96 ** 2 / (2 * (n - 1)))
    lo, hi = md - 1.96 * sd, md + 1.96 * sd
    sl = stats.linregress(mean, diff)
    res = Result(L.t("Bland-Altman agreement"))
    res.table("Agreement", pd.DataFrame([{"N": n, "Mean difference (bias)": md, "Bias low 95%": md - tc * sd / math.sqrt(n), "Bias high 95%": md + tc * sd / math.sqrt(n),
                                          "SD of differences": sd, "Lower LoA": lo, "Upper LoA": hi, "LoA CI half-width (approx)": tc * se_loa,
                                          "Slope diff~mean": sl.slope, "Slope p-value (proportional bias)": sl.pvalue, "Pearson r": np.corrcoef(x, y)[0, 1]}]))
    fig, ax = plt.subplots(figsize=(6.4, 4.2)); ax.scatter(mean, diff, s=18, color=PALETTE[0])
    for v_, st in [(md, "-"), (lo, "--"), (hi, "--")]: ax.axhline(v_, color=PALETTE[3], ls=st)
    ax.axhline(0, color="grey", lw=.6); ax.set_xlabel("(A+B)/2"); ax.set_ylabel("A-B"); ax.set_title(L.fig("Bland-Altman agreement")); res.fig("Bland-Altman", save_fig(fig, outdir, "bland_altman"))
    res.summary.update({"bias": float(md), "loa": [float(lo), float(hi)]})
    return res


# ============================================================================
# Outliers
# ============================================================================
def _grubbs_crit(n, a=0.05):
    t = stats.t.ppf(1 - a / (2 * n), n - 2); return (n - 1) / math.sqrt(n) * math.sqrt(t * t / (n - 2 + t * t))


def _esd(x, r, a=0.05):
    x = list(map(float, x)); n = len(x); idx = list(range(n)); out = []
    for i in range(1, r + 1):
        arr = np.array([x[j] for j in idx]); z = np.abs(arr - arr.mean()) / arr.std(ddof=1); j = int(np.argmax(z)); R = z[j]
        p = 1 - a / (2 * (n - i + 1)); t = stats.t.ppf(p, n - i - 1); lam = (n - i) * t / math.sqrt((n - i - 1 + t * t) * (n - i + 1))
        out.append((i, idx[j], R, lam)); idx.pop(j)
    last = max([o[0] for o in out if o[2] > o[3]] or [0])
    return out, last


def outliers(df, params, L: Lang, outdir):
    """column (univariate) or columns (multivariate Mahalanobis).  max_outliers for generalized ESD."""
    res = Result(L.t("Outlier detection"))
    if params.get("columns"):
        cols = params["columns"]; d = df[cols].dropna(); X = d.values; mu = X.mean(0); S = np.cov(X.T); D2 = np.einsum("ij,jk,ik->i", X - mu, np.linalg.inv(S), X - mu)
        cut = stats.chi2.ppf(1 - float(params.get("alpha", 0.001)), len(cols)); flag = D2 > cut
        res.table("Mahalanobis distance", pd.DataFrame({"Row": d.index, "D2": D2, "p (chi-square)": stats.chi2.sf(D2, len(cols)), "Outlier": flag}).sort_values("D2", ascending=False).head(25))
        fig, ax = plt.subplots(figsize=(7, 3.6)); ax.plot(D2, "o", ms=3); ax.axhline(cut, color=PALETTE[3]); ax.set_title(L.fig("Outlier detection")); res.fig("Mahalanobis", save_fig(fig, outdir, "outliers_mv"))
        res.summary["n_outliers"] = int(flag.sum()); return res
    col = params["column"]; x = df[col].dropna(); xv = x.values; n = len(xv); a = float(params.get("alpha", 0.05))
    G = np.abs(xv - xv.mean()) / xv.std(ddof=1); gi = int(np.argmax(G)); gc = _grubbs_crit(n, a)
    res.table("Grubbs test (single outlier, two-sided)", pd.DataFrame([{"Max |z|": G[gi], "Critical G": gc, "Value": xv[gi], "Row": x.index[gi], "Outlier": bool(G[gi] > gc), "p-value (approx.)": min(1, n * 2 * stats.t.sf(math.sqrt(n * (n - 2) * G[gi] ** 2 / ((n - 1) ** 2 - n * G[gi] ** 2)), n - 2)) if (n - 1) ** 2 > n * G[gi] ** 2 else 0.0}]))
    r = int(params.get("max_outliers", max(1, min(10, n // 10)))); out, last = _esd(xv, r, a)
    res.table("Generalized ESD (Rosner)", pd.DataFrame([{"i": i, "Index (position)": j, "R_i": R_, "lambda_i": lam, "Significant": R_ > lam} for i, j, R_, lam in out]).assign(**{"Outliers found": last}))
    q1, q3 = np.percentile(xv, [25, 75]); iqr = q3 - q1; med = np.median(xv); mad = stats.median_abs_deviation(xv); mz = 0.6745 * (xv - med) / mad if mad > 0 else np.zeros(n)
    inner = (xv < q1 - 1.5 * iqr) | (xv > q3 + 1.5 * iqr); outer = (xv < q1 - 3 * iqr) | (xv > q3 + 3 * iqr); mzf = np.abs(mz) > 3.5
    res.table("Rule-based flags", pd.DataFrame([{"Method": "Tukey fences 1.5 IQR", "Count": int(inner.sum())}, {"Method": "Tukey fences 3 IQR (extreme)", "Count": int(outer.sum())}, {"Method": "Modified z > 3.5 (MAD)", "Count": int(mzf.sum())}]))
    flagged = x[inner | mzf]
    if len(flagged): res.table("Flagged values (Tukey 1.5 IQR or modified z)", pd.DataFrame({"Row": flagged.index, col: flagged.values}))
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.6)); ax[0].boxplot(xv, vert=False); ax[0].set_title(L.fig("Box plot")); ax[1].plot(xv, "o", ms=3); ax[1].axhline(q1 - 1.5 * iqr, color=PALETTE[3], ls="--"); ax[1].axhline(q3 + 1.5 * iqr, color=PALETTE[3], ls="--")
    res.fig("outliers", save_fig(fig, outdir, f"outliers_{col}")); res.summary.update({"grubbs_G": float(G[gi]), "grubbs_crit": float(gc), "esd_outliers": int(last)})
    res.note("Removing a flagged point requires an engineering reason (measurement error, special cause); statistical flagging alone is not sufficient.")
    return res


# ============================================================================
# Tolerance intervals
# ============================================================================
def tol_factor(n, p, conf, two_sided=True):
    if not two_sided:
        zp = stats.norm.ppf(p); return stats.nct.ppf(conf, n - 1, zp * math.sqrt(n)) / math.sqrt(n)
    nu = n - 1; z = stats.norm.ppf((1 + p) / 2); k = z * math.sqrt(nu * (1 + 1 / n) / stats.chi2.ppf(1 - conf, nu))
    return k  # Howe (1969) approximation


def tolerance(df, params, L: Lang, outdir):
    """params: column, coverage (default 0.99), conf (default 0.95). Normal two-/one-sided tolerance factors, distribution-free [min,max], confidence & prediction intervals."""
    col = params["column"]; x = df[col].dropna().values; n = len(x); p = float(params.get("coverage", 0.99)); c = float(params.get("conf", 0.95))
    m, s = x.mean(), x.std(ddof=1); k2 = tol_factor(n, p, c, True); k1 = tol_factor(n, p, c, False)
    xs = np.sort(x); conf_np = 1 - (n * p ** (n - 1) - (n - 1) * p ** n)
    nmin = next(i for i in range(2, 100000) if 1 - (i * p ** (i - 1) - (i - 1) * p ** i) >= c)
    res = Result(L.t("Tolerance interval"))
    res.table("Normal-theory tolerance intervals", pd.DataFrame([{"Coverage": p, "Confidence": c, "n": n, "Mean": m, "Std Dev": s, "k (two-sided, Howe)": k2, "Two-sided low": m - k2 * s, "Two-sided high": m + k2 * s,
                                                                  "k (one-sided, exact)": k1, "Upper bound (1-sided)": m + k1 * s, "Lower bound (1-sided)": m - k1 * s}]))
    res.table("Distribution-free [min, max]", pd.DataFrame([{"Interval": f"[{xs[0]:.6g}, {xs[-1]:.6g}]", "Confidence achieved for coverage": conf_np, "Min n for requested conf.": nmin}]))
    tc = stats.t.ppf(.975, n - 1)
    res.table("Confidence & prediction intervals (95%)", pd.DataFrame([{"Mean CI low": m - tc * s / math.sqrt(n), "Mean CI high": m + tc * s / math.sqrt(n), "Prediction low (1 new obs)": m - tc * s * math.sqrt(1 + 1 / n), "Prediction high (1 new obs)": m + tc * s * math.sqrt(1 + 1 / n)}]))
    fig, ax = plt.subplots(figsize=(7, 3.6)); ax.hist(x, bins="auto", color=PALETTE[0], alpha=.75, edgecolor="white")
    for v_ in (m - k2 * s, m + k2 * s): ax.axvline(v_, color=PALETTE[3], ls="--")
    ax.set_title(L.fig("Tolerance interval")); res.fig("tolerance", save_fig(fig, outdir, f"tolerance_{col}"))
    res.summary.update({"k2": float(k2), "k1": float(k1), "low": float(m - k2 * s), "high": float(m + k2 * s)})
    res.note("Two-sided factor uses Howe's approximation (typically within ~0.5% of exact tables); assumes normality - check Q-Q plot.")
    return res


# ============================================================================
# Normalizing transformations
# ============================================================================
def transform(df, params, L: Lang, outdir):
    """column; method boxcox|yeojohnson|both."""
    col = params["column"]; x = df[col].dropna().values.astype(float); res = Result(L.t("Normalizing transformation")); rows = []; tr = {}
    sw = lambda v: stats.shapiro(v)
    base = sw(x); rows.append({"Method": "Original", "Lambda": np.nan, "Shapiro-Wilk W": base[0], "p-value": base[1], "Skewness": stats.skew(x)})
    shift = 0.0
    if params.get("method", "both") in ("boxcox", "both"):
        if x.min() <= 0:
            shift = -x.min() + 1e-6 + float(params.get("shift", 0)); res.note(f"Box-Cox needs positive data: shifted by {shift:.6g}.")
        y, lam, ci = stats.boxcox(x + shift, alpha=0.05); tr["Box-Cox"] = y; s_ = sw(y)
        rows.append({"Method": "Box-Cox", "Lambda": lam, "Shapiro-Wilk W": s_[0], "p-value": s_[1], "Skewness": stats.skew(y), "Lambda low 95%": ci[0], "Lambda high 95%": ci[1]}); res.summary["boxcox_lambda"] = float(lam)
    if params.get("method", "both") in ("yeojohnson", "both"):
        y, lam = stats.yeojohnson(x); tr["Yeo-Johnson"] = y; s_ = sw(y); rows.append({"Method": "Yeo-Johnson", "Lambda": lam, "Shapiro-Wilk W": s_[0], "p-value": s_[1], "Skewness": stats.skew(y)}); res.summary["yj_lambda"] = float(lam)
    res.table("Transformations", pd.DataFrame(rows))
    names = ["Original"] + list(tr); vals = [x] + list(tr.values())
    fig, axs = plt.subplots(2, len(vals), figsize=(3.6 * len(vals), 6.2), squeeze=False)
    for j, (nm, v) in enumerate(zip(names, vals)):
        axs[0, j].hist(v, bins="auto", color=PALETTE[j % 8], edgecolor="white"); axs[0, j].set_title(nm)
        (osm, osr), (sl, ic, r) = stats.probplot(v, dist="norm"); axs[1, j].plot(osm, osr, "o", ms=3); axs[1, j].plot(osm, sl * np.asarray(osm) + ic, color=PALETTE[3])
    res.fig("before / after", save_fig(fig, outdir, f"transform_{col}"))
    res.table("Transformed data (first 50 rows)", pd.DataFrame({col: x, **{k: v for k, v in tr.items()}}).head(50))
    return res


# ============================================================================
# Bootstrap
# ============================================================================
def bootstrap(df, params, L: Lang, outdir):
    """column; statistic mean|median|std|var|trimmed_mean|corr|cpk; column2 (corr); lsl/usl (cpk); n_resamples; seed."""
    st = params.get("statistic", "mean"); col = params["column"]; nres = int(params.get("n_resamples", 5000)); seed = int(params.get("seed", 1))
    if st == "corr":
        d = df[[col, params["column2"]]].dropna(); data = (d[col].values, d[params["column2"]].values)
        f = lambda a, b, axis=-1: np.array([np.corrcoef(a[i], b[i])[0, 1] for i in range(a.shape[0])]) if a.ndim > 1 else np.corrcoef(a, b)[0, 1]
        paired = True; est = np.corrcoef(*data)[0, 1]
    else:
        x = df[col].dropna().values; data = (x,); paired = False
        lsl, usl = params.get("lsl"), params.get("usl")
        fs = {"mean": lambda a, axis=-1: np.mean(a, axis=axis), "median": lambda a, axis=-1: np.median(a, axis=axis), "std": lambda a, axis=-1: np.std(a, axis=axis, ddof=1),
              "var": lambda a, axis=-1: np.var(a, axis=axis, ddof=1), "trimmed_mean": lambda a, axis=-1: stats.trim_mean(a, 0.1, axis=axis)}
        if st == "cpk":
            def cpk(a, axis=-1):
                m_, s_ = np.mean(a, axis=axis), np.std(a, axis=axis, ddof=1); return np.minimum(usl - m_, m_ - lsl) / (3 * s_)
            fs["cpk"] = cpk
        f = fs[st]; est = float(f(x))
    kw = dict(n_resamples=nres, random_state=seed, confidence_level=float(params.get("conf", 0.95)))
    if paired:
        # manual percentile + BCa-free for paired corr (resample pairs)
        rng = np.random.default_rng(seed); n = len(data[0]); bs = np.array([np.corrcoef(data[0][i], data[1][i])[0, 1] for i in (rng.integers(0, n, n) for _ in range(nres))])
        a_ = (1 - kw["confidence_level"]) / 2; ci_p = tuple(np.quantile(bs, [a_, 1 - a_])); ci_b = (np.nan, np.nan); se = bs.std(ddof=1)
    else:
        bp = stats.bootstrap(data, f, method="percentile", **kw); bb = stats.bootstrap(data, f, method="BCa", **kw)
        ci_p = (bp.confidence_interval.low, bp.confidence_interval.high); ci_b = (bb.confidence_interval.low, bb.confidence_interval.high); se = bp.standard_error; bs = bp.bootstrap_distribution
    res = Result(L.t("Bootstrap confidence interval"))
    res.table("Bootstrap", pd.DataFrame([{"Statistic": st, "Estimate": est, "Boot SE": se, "Percentile low": ci_p[0], "Percentile high": ci_p[1], "BCa low": ci_b[0], "BCa high": ci_b[1], "Resamples": nres}]))
    fig, ax = plt.subplots(figsize=(6, 3.6)); ax.hist(bs, bins=50, color=PALETTE[0], edgecolor="white"); ax.axvline(est, color=PALETTE[3])
    for v_ in ci_p: ax.axvline(v_, color="k", ls="--")
    ax.set_title(L.fig("Bootstrap confidence interval")); res.fig("bootstrap distribution", save_fig(fig, outdir, f"bootstrap_{st}"))
    res.summary.update({"estimate": float(est), "ci_percentile": [float(ci_p[0]), float(ci_p[1])]})
    return res


# ============================================================================
# Multiple testing
# ============================================================================
def multiple_testing(df, params, L: Lang, outdir):
    """column with raw p-values (or params['pvalues'] list); optional label column."""
    if params.get("pvalues") is not None: p = np.array(params["pvalues"], float); lab = [f"H{i+1}" for i in range(len(p))]
    else:
        d = df[[params["column"]] + ([params["label"]] if params.get("label") else [])].dropna(); p = d[params["column"]].values.astype(float); lab = d[params["label"]].astype(str).tolist() if params.get("label") else [f"H{i+1}" for i in range(len(p))]
    a = float(params.get("alpha", 0.05)); tab = pd.DataFrame({"Hypothesis": lab, "Raw p": p})
    for m in ["bonferroni", "sidak", "holm", "hochberg", "bh", "by"]: tab[m] = adjust_p(p, m)
    res = Result(L.t("Multiple-testing adjustment")); res.table("Adjusted p-values", tab)
    res.table("Rejections at alpha", pd.DataFrame([{"Method": m, "Rejected": int((tab[m] < a).sum()), "alpha": a} for m in ["Raw p", "bonferroni", "sidak", "holm", "hochberg", "bh", "by"]]))
    res.note("Bonferroni/Sidak/Holm/Hochberg control FWER; BH/BY control FDR (Hochberg assumes independence or positive dependence; BH likewise; BY valid under any dependence).")
    o = np.argsort(p); fig, ax = plt.subplots(figsize=(6, 3.8)); ax.plot(np.arange(1, len(p) + 1), p[o], "o-", ms=3, label="raw p"); ax.plot(np.arange(1, len(p) + 1), a * np.arange(1, len(p) + 1) / len(p), "--", label="BH line")
    ax.axhline(a, color="grey", lw=.7); ax.legend(); ax.set_xlabel("rank"); ax.set_title(L.fig("Multiple-testing adjustment")); res.fig("p-value plot", save_fig(fig, outdir, "multitest"))
    res.summary.update({m: int((tab[m] < a).sum()) for m in ["bonferroni", "holm", "bh"]})
    return res


# ============================================================================
# Tabulate (JMP Tabulate / SPSS Descriptives, Frequencies, Means)
# ============================================================================
def tabulate(df, params, L: Lang, outdir):
    """columns (numeric; default all), by (list of grouping columns), stats list from N,Mean,Std Dev,Min,Q1,Median,Q3,Max,Sum,CV%,Missing; categorical (list) for frequency tables."""
    cols = params.get("columns") or numeric_cols(df); by = params.get("by") or []; by = [by] if isinstance(by, str) else by
    sts = params.get("stats") or ["N", "Mean", "Std Dev", "Min", "Median", "Max"]
    fn = {"N": lambda s: s.count(), "Mean": lambda s: s.mean(), "Std Dev": lambda s: s.std(), "Min": lambda s: s.min(), "Q1": lambda s: s.quantile(.25), "Median": lambda s: s.median(),
          "Q3": lambda s: s.quantile(.75), "Max": lambda s: s.max(), "Sum": lambda s: s.sum(), "CV%": lambda s: 100 * s.std() / s.mean() if s.mean() else np.nan, "Missing": lambda s: s.isna().sum()}
    res = Result(L.t("Tabulate")); rows = []
    groups = df.groupby(by, dropna=False) if by else [((), df)]
    for key, d in groups:
        key = key if isinstance(key, tuple) else (key,)
        for c in cols:
            rows.append({**dict(zip(by, key)), "Variable": c, **{s: fn[s](d[c]) for s in sts}})
    t = pd.DataFrame(rows); res.table("Summary by group" if by else "Summary", t)
    for c in params.get("categorical") or []:
        vc = df[c].value_counts(dropna=False); res.table(f"Frequencies - {c}", pd.DataFrame({c: vc.index.astype(str), "Count": vc.values, "Percent": 100 * vc.values / vc.sum(), "Cumulative %": 100 * vc.cumsum().values / vc.sum()}))
    if by and "Mean" in sts:
        c0 = cols[0]; fig, ax = plt.subplots(figsize=(7, 3.8)); sub = t[t["Variable"] == c0]; lab = sub[by].astype(str).agg(" / ".join, axis=1)
        err = sub["Std Dev"] / np.sqrt(sub["N"]) * 1.96 if "Std Dev" in sts and "N" in sts else None
        ax.bar(lab, sub["Mean"], yerr=err, capsize=3, color=PALETTE[0]); ax.set_ylabel(c0); plt.setp(ax.get_xticklabels(), rotation=30, ha="right"); ax.set_title(L.fig("Tabulate")); res.fig("means", save_fig(fig, outdir, "tabulate"))
    return res


# ============================================================================
# Partial correlation
# ============================================================================
def partial_corr(df, params, L: Lang, outdir):
    """x, y, controls (list)."""
    x, y, z = params["x"], params["y"], params["controls"]; d = df[[x, y] + z].dropna(); n = len(d)
    Z = np.column_stack([np.ones(n)] + [d[c].values for c in z])
    rx = d[x].values - Z @ np.linalg.lstsq(Z, d[x].values, rcond=None)[0]; ry = d[y].values - Z @ np.linalg.lstsq(Z, d[y].values, rcond=None)[0]
    r = np.corrcoef(rx, ry)[0, 1]; dof = n - 2 - len(z); t = r * math.sqrt(dof / (1 - r * r)); p = 2 * stats.t.sf(abs(t), dof)
    zf = np.arctanh(r); se = 1 / math.sqrt(n - 3 - len(z)); ci = (math.tanh(zf - 1.96 * se), math.tanh(zf + 1.96 * se))
    r0 = np.corrcoef(d[x], d[y])[0, 1]
    sp = float(np.corrcoef(rx, d[y].values)[0, 1])
    res = Result(L.t("Partial correlation"))
    res.table("Partial correlation", pd.DataFrame([{"x": x, "y": y, "Controls": ", ".join(z), "Zero-order r": r0, "Partial r": r, "DF": dof, "t": t, "p-value": p, "Lower 95%": ci[0], "Upper 95%": ci[1], "Semi-partial r (x|controls vs y)": sp}]))
    fig, ax = plt.subplots(figsize=(5.2, 4)); ax.scatter(rx, ry, s=16); ax.set_xlabel(f"{x} | controls"); ax.set_ylabel(f"{y} | controls"); ax.set_title(L.fig("Partial correlation")); res.fig("partial residual plot", save_fig(fig, outdir, "partial_corr"))
    res.summary.update({"partial_r": float(r), "p": float(p)})
    return res
