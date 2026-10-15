"""ss_graph.py - Graph Builder (JMP) / Chart Builder (SPSS) and automatic Fit-Y-by-X dispatcher.
graph: scatter, line, bar, box, violin, hist, density, heatmap, bubble, pairs, area, pie, with optional fit lines, grouping colour, facets (by).
auto_xy: chooses the analysis from the modeling types of Y and X exactly like JMP's Fit Y by X.
"""
from __future__ import annotations
import math
import numpy as np, pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from ss_common import HE, Result, Lang, save_fig, PALETTE, numeric_cols

HE.update({"Graph": "גרף", "Fit Y by X": "התאמת Y לפי X"})


def lowess(x, y, frac=0.6, grid=None):
    x = np.asarray(x, float); y = np.asarray(y, float); n = len(x); r = max(3, int(math.ceil(frac * n))); g = np.sort(x) if grid is None else grid; out = []
    for x0 in g:
        dist = np.abs(x - x0); h = np.sort(dist)[min(r, n) - 1]; h = h if h > 0 else 1e-12; w = np.clip(1 - (dist / h) ** 3, 0, None) ** 3
        A = np.column_stack([np.ones(n), x - x0]); W = w; AtW = A.T * W
        try: beta = np.linalg.solve(AtW @ A + 1e-12 * np.eye(2), AtW @ y); out.append(beta[0])
        except np.linalg.LinAlgError: out.append(np.nan)
    return g, np.array(out)


def _fit_line(ax, x, y, fit, color, ci=True, label=None):
    m = np.isfinite(x) & np.isfinite(y); x, y = x[m], y[m]
    if len(x) < 4: return
    xs = np.linspace(x.min(), x.max(), 200)
    if fit in ("linear", "quadratic", "cubic"):
        deg = {"linear": 1, "quadratic": 2, "cubic": 3}[fit]; X = np.vander(x, deg + 1); b = np.linalg.lstsq(X, y, rcond=None)[0]; Xs = np.vander(xs, deg + 1); ys = Xs @ b
        ax.plot(xs, ys, color=color, lw=2, label=label)
        if ci:
            dfe = len(x) - (deg + 1); s2 = ((y - X @ b) ** 2).sum() / dfe; XtXi = np.linalg.pinv(X.T @ X); se = np.sqrt(s2 * np.einsum("ij,jk,ik->i", Xs, XtXi, Xs)); tc = stats.t.ppf(.975, dfe); ax.fill_between(xs, ys - tc * se, ys + tc * se, color=color, alpha=.15)
    elif fit == "lowess": gx, gy = lowess(x, y, grid=xs); ax.plot(gx, gy, color=color, lw=2, label=label)


def _cat_order(s): return sorted(s.dropna().unique(), key=lambda v: (str(type(v)), v) if not isinstance(v, str) else (str(type(v)), v))


def graph(df, params, L: Lang, outdir):
    """type: scatter|line|bar|box|violin|hist|density|heatmap|bubble|pairs|area|pie.  x, y, color (group col), size (bubble), by (facet col), fit (none|linear|quadratic|cubic|lowess), ci (bool),
    agg (bar: mean|median|sum|count), error (bar: sd|se|ci|none), bins, title, columns (pairs/heatmap), logx, logy."""
    typ = params.get("type", "scatter"); x, y, col, size, by = params.get("x"), params.get("y"), params.get("color"), params.get("size"), params.get("by"); fit = params.get("fit", "none"); ci = params.get("ci", True)
    res = Result(L.t("Graph")); title = params.get("title") or f"{typ}: " + " vs ".join([v for v in (y, x) if v]); facets = [(None, df)] if not by else [(g, d) for g, d in df.groupby(by, sort=True)]; nf = len(facets)
    if typ == "pairs":
        cols = params.get("columns") or numeric_cols(df)[:6]; k = len(cols); fig, axs = plt.subplots(k, k, figsize=(2.2 * k, 2.2 * k), squeeze=False)
        for i, a in enumerate(cols):
            for j, b in enumerate(cols):
                ax = axs[i, j]; ax.grid(False)
                if i == j: ax.hist(df[a].dropna(), bins="auto", color=PALETTE[0], edgecolor="white")
                else:
                    if col:
                        for gi, (gname, gd) in enumerate(df.groupby(col)): ax.plot(gd[b], gd[a], ".", ms=3, color=PALETTE[gi % 8], alpha=.7)
                    else: ax.plot(df[b], df[a], ".", ms=3, color=PALETTE[0], alpha=.6)
                    if fit != "none": _fit_line(ax, df[b].values.astype(float), df[a].values.astype(float), fit, PALETTE[3], ci)
                if j == 0: ax.set_ylabel(a, fontsize=7)
                if i == k - 1: ax.set_xlabel(b, fontsize=7)
                ax.tick_params(labelsize=6)
        res.fig(title, save_fig(fig, outdir, "graph_pairs")); return res
    if typ == "heatmap":
        if params.get("columns") or (x is None and y is None):
            cols = params.get("columns") or numeric_cols(df); M = df[cols].corr(method=params.get("method", "pearson")); fig, ax = plt.subplots(figsize=(1 + .7 * len(cols), 1 + .6 * len(cols))); im = ax.imshow(M.values, cmap="RdBu_r", vmin=-1, vmax=1); ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, rotation=45, ha="right"); ax.set_yticks(range(len(cols))); ax.set_yticklabels(cols); ax.grid(False)
            for i in range(len(cols)):
                for j in range(len(cols)): ax.text(j, i, f"{M.values[i,j]:.2f}", ha="center", va="center", fontsize=7)
            fig.colorbar(im); ax.set_title(title); res.table("Correlation matrix", M.round(4), index=True)
        else:
            z = params.get("z"); P = df.pivot_table(index=y, columns=x, values=z, aggfunc=params.get("agg", "mean")) if z else pd.crosstab(df[y], df[x]); fig, ax = plt.subplots(figsize=(1 + .6 * P.shape[1], 1 + .5 * P.shape[0])); im = ax.imshow(P.values, cmap="viridis", aspect="auto"); ax.set_xticks(range(P.shape[1])); ax.set_xticklabels(P.columns.astype(str), rotation=45, ha="right"); ax.set_yticks(range(P.shape[0])); ax.set_yticklabels(P.index.astype(str)); ax.grid(False); fig.colorbar(im); ax.set_title(title)
        res.fig(title, save_fig(fig, outdir, "graph_heatmap")); return res
    if typ == "pie":
        vc = df[x].value_counts(); fig, ax = plt.subplots(figsize=(5, 5)); ax.pie(vc.values, labels=vc.index.astype(str), autopct="%1.1f%%", colors=PALETTE[:len(vc)] * 3, startangle=90); ax.set_title(title); res.fig(title, save_fig(fig, outdir, "graph_pie")); res.table("Counts", pd.DataFrame({x: vc.index.astype(str), "Count": vc.values})); return res
    cols_n = min(nf, 3); rows_n = math.ceil(nf / cols_n); fig, axs = plt.subplots(rows_n, cols_n, figsize=(5.2 * cols_n, 3.8 * rows_n), squeeze=False)
    for fi, (gname, d) in enumerate(facets):
        ax = axs[fi // cols_n, fi % cols_n]
        groups = [(None, d)] if not col else [(g, dd) for g, dd in d.groupby(col, sort=True)]
        if typ in ("scatter", "bubble", "line", "area"):
            for gi, (g, dd) in enumerate(groups):
                dd = dd.dropna(subset=[x, y]); colr = PALETTE[gi % 8]
                if typ == "line" or typ == "area": dd = dd.sort_values(x)
                if typ == "scatter": ax.plot(dd[x], dd[y], "o", ms=4, color=colr, alpha=.75, label=None if g is None else str(g))
                elif typ == "bubble": s_ = dd[size].values.astype(float); ax.scatter(dd[x], dd[y], s=30 + 400 * (s_ - s_.min()) / (np.ptp(s_) or 1), color=colr, alpha=.55, edgecolor="white", label=None if g is None else str(g))
                elif typ == "line": ax.plot(dd[x], dd[y], "-o", ms=3, color=colr, label=None if g is None else str(g))
                else: ax.fill_between(dd[x], dd[y], color=colr, alpha=.35, label=None if g is None else str(g))
                if fit != "none" and typ in ("scatter", "bubble", "line"): _fit_line(ax, dd[x].values.astype(float), dd[y].values.astype(float), fit, colr if col else PALETTE[3], ci)
            if params.get("logx"): ax.set_xscale("log")
            if params.get("logy"): ax.set_yscale("log")
            ax.set_xlabel(x); ax.set_ylabel(y)
        elif typ in ("box", "violin"):
            cats = _cat_order(d[x]) if x else [None]; w = .8 / max(len(groups), 1)
            for gi, (g, dd) in enumerate(groups):
                data = [dd.loc[dd[x] == c, y].dropna().values if x else dd[y].dropna().values for c in cats]; pos = np.arange(len(cats)) + (gi - (len(groups) - 1) / 2) * w
                keep = [i for i, v in enumerate(data) if len(v) > 0]
                if typ == "box":
                    bp = ax.boxplot([data[i] for i in keep], positions=pos[keep], widths=w * .85, patch_artist=True, showfliers=True, flierprops={"markersize": 3})
                    for p_ in bp["boxes"]: p_.set_facecolor(PALETTE[gi % 8]); p_.set_alpha(.55)
                else:
                    vp = ax.violinplot([data[i] for i in keep], positions=pos[keep], widths=w * .9, showmedians=True)
                    for b_ in vp["bodies"]: b_.set_facecolor(PALETTE[gi % 8]); b_.set_alpha(.55)
            ax.set_xticks(range(len(cats))); ax.set_xticklabels([str(c) for c in cats]); ax.set_ylabel(y); ax.set_xlabel(x or "")
            if col: ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=PALETTE[i % 8], alpha=.55) for i in range(len(groups))], labels=[str(g) for g, _ in groups], fontsize=7)
        elif typ in ("hist", "density"):
            v = x or y
            for gi, (g, dd) in enumerate(groups):
                s_ = dd[v].dropna().values.astype(float)
                if typ == "hist": ax.hist(s_, bins=params.get("bins", "auto"), alpha=.6 if col else .85, color=PALETTE[gi % 8], edgecolor="white", label=None if g is None else str(g))
                else: xs = np.linspace(s_.min(), s_.max(), 200); ax.plot(xs, stats.gaussian_kde(s_)(xs), color=PALETTE[gi % 8], lw=2, label=None if g is None else str(g)); ax.fill_between(xs, stats.gaussian_kde(s_)(xs), color=PALETTE[gi % 8], alpha=.2)
            ax.set_xlabel(v)
        elif typ == "bar":
            agg = params.get("agg", "mean"); err = params.get("error", "ci" if agg == "mean" else "none"); cats = _cat_order(d[x]); w = .8 / max(len(groups), 1)
            for gi, (g, dd) in enumerate(groups):
                vals, errs = [], []
                for c in cats:
                    s_ = dd.loc[dd[x] == c, y].dropna() if y else dd.loc[dd[x] == c, x]
                    v = {"mean": s_.mean, "median": s_.median, "sum": s_.sum, "count": s_.count}[agg]() if y or agg == "count" else len(s_); vals.append(v)
                    e = 0
                    if agg == "mean" and len(s_) > 1:
                        e = {"sd": s_.std(), "se": s_.sem(), "ci": s_.sem() * stats.t.ppf(.975, len(s_) - 1), "none": 0}[err]
                    errs.append(e)
                pos = np.arange(len(cats)) + (gi - (len(groups) - 1) / 2) * w; ax.bar(pos, vals, w * .9, yerr=errs if any(errs) else None, capsize=3, color=PALETTE[gi % 8], label=None if g is None else str(g))
                res.tables.append((f"Bar values{'' if g is None else ' - ' + str(g)}{'' if gname is None else ' [' + str(gname) + ']'}", pd.DataFrame({x: [str(c) for c in cats], f"{agg}({y or 'rows'})": vals, "Error bar": errs})))
            ax.set_xticks(range(len(cats))); ax.set_xticklabels([str(c) for c in cats], rotation=20 if len(cats) > 6 else 0); ax.set_ylabel(f"{agg}({y})" if y else "count"); ax.set_xlabel(x)
        else: raise ValueError("unknown graph type")
        if col and typ not in ("box", "violin"): ax.legend(fontsize=7, title=col)
        ax.set_title(title if gname is None else f"{by} = {gname}", fontsize=10)
    for k in range(nf, rows_n * cols_n): axs[k // cols_n, k % cols_n].axis("off")
    fig.tight_layout(); res.fig(title, save_fig(fig, outdir, f"graph_{typ}")); return res


def auto_xy(df, params, L: Lang, outdir):
    """Fit Y by X: y, x; picks one-way ANOVA (num~cat), regression (num~num), logistic/multinomial (cat~num), contingency (cat~cat). Optional 'by' not supported."""
    from ss_common import Result
    import ss_linear, ss_desc, ss_models2
    y, x = params["y"], params["x"]; is_num = lambda c: pd.api.types.is_numeric_dtype(df[c]) and df[c].nunique() > int(params.get("max_levels", 8)) or (pd.api.types.is_float_dtype(df[c]))
    ym, xm = ("continuous" if is_num(y) else "categorical"), ("continuous" if is_num(x) else "categorical")
    if ym == "continuous" and xm == "categorical": r = ss_linear.oneway(df, {"response": y, "factor": x, "posthoc": params.get("posthoc", "tukey")}, L, outdir); how = "One-way ANOVA (Y continuous, X categorical)"
    elif ym == "continuous": r = ss_linear.regression(df, {"response": y, "predictors": [x], "poly": int(params.get("poly", 1))}, L, outdir); how = "Bivariate regression (Y continuous, X continuous)"
    elif xm == "continuous":
        lv = df[y].dropna().astype(str).nunique()
        if lv == 2: r = ss_linear.glm(df, {"response": y, "predictors": [x], "family": "binomial"}, L, outdir); how = "Logistic regression (Y binary, X continuous)"
        else: r = ss_models2.multinomial(df, {"response": y, "predictors": [x]}, L, outdir); how = "Multinomial logistic (Y nominal, X continuous)"
    else: r = ss_desc.crosstab(df, {"row": y, "col": x}, L, outdir); how = "Contingency analysis (Y categorical, X categorical)"
    r.notes.insert(0, f"Fit Y by X chose: {how}.")
    r.summary["method"] = how; return r
