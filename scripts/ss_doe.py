"""ss_doe.py - Design of Experiments: design generation (full/fractional factorial, CCD, Box-Behnken) and analysis
(effects, Lenth PSE, Pareto, half-normal, main/interaction plots, ANOVA, curvature, optimisation).  (JMP: DOE > Custom/Classical + Fit Model + Prediction Profiler)"""
from __future__ import annotations
import itertools, string
import numpy as np, pandas as pd
from scipy import stats, optimize
import matplotlib.pyplot as plt
from ss_common import Result, Lang, save_fig, PALETTE
from ss_linear import OLS, build_terms, _tr_cols, _diag_plots


def _alias_structure(k, gens):
    """gens: list like ['D=ABC'] -> defining relation words and alias table for main effects and 2FIs. Returns (words, resolution, aliases dict)."""
    letters = string.ascii_uppercase[:k]
    words = []
    for g in gens:
        lhs, rhs = g.replace(" ", "").split("=")
        words.append(frozenset(lhs + rhs))
    group = set(words)
    changed = True
    while changed:
        changed = False
        for a, b in itertools.combinations(list(group), 2):
            c = a ^ b
            if c and c not in group:
                group.add(c); changed = True
    res = min(len(w) for w in group) if group else None
    effects = [frozenset(c) for r in (1, 2) for c in itertools.combinations(letters, r)]
    al = {}
    for ef in effects:
        al["".join(sorted(ef))] = ["".join(sorted(ef ^ w)) for w in group if (ef ^ w) and (ef ^ w) != ef and len(ef ^ w) <= 3]
    return group, res, al


def make_design(df, params, L: Lang, outdir):
    """params: type (full|fractional|ccd|boxbehnken), factors (dict name -> [low, high]) , generators (for fractional, e.g. ['D=ABC']), center_points (int), replicates (int), randomize (bool), seed, alpha (ccd: 'face'|'rotatable')."""
    typ = params.get("type", "full"); fac = params["factors"]; names = list(fac); k = len(names)
    reps = int(params.get("replicates", 1)); cp = int(params.get("center_points", 0)); rng = np.random.default_rng(int(params.get("seed", 1)))
    res = Result("Design of Experiments - " + typ)
    gens = params.get("generators", [])
    if typ in ("full", "fractional"):
        base_k = k - len(gens) if typ == "fractional" else k
        base = np.array(list(itertools.product([-1, 1], repeat=base_k)))[:, ::-1] if base_k else np.zeros((1, 0))
        base = np.array(list(itertools.product([-1, 1], repeat=base_k)))
        M = base.copy()
        letters = string.ascii_uppercase[:k]
        if typ == "fractional":
            for g in gens:
                lhs, rhs = g.replace(" ", "").split("=")
                col = np.prod([M[:, letters.index(c)] for c in rhs], axis=0)
                M = np.column_stack([M, col])
            # reorder columns to factor letter order (generated col index = letters.index(lhs))
            order = list(range(base_k)) + [letters.index(g.replace(" ", "").split("=")[0]) for g in gens]
            Mo = np.zeros_like(M)
            for j, tgt in enumerate(order): Mo[:, tgt] = M[:, j]
            M = Mo
            grp, resolution, al = _alias_structure(k, gens)
            res.table("Defining relation / aliases (main effects & 2FI)", pd.DataFrame([{"Effect": a, "Aliased with": ", ".join(b) if b else "-"} for a, b in al.items()]))
            res.note(f"Resolution {['', '', '', 'III', 'IV', 'V', 'VI', 'VII'][min(resolution, 7)] if resolution else '?'} (shortest word in defining relation = {resolution}). Res III: main effects confounded with 2FI; Res IV: 2FI confounded with each other; Res V: clear 2FI.")
            res.summary["resolution"] = resolution
    elif typ == "ccd":
        f_ = np.array(list(itertools.product([-1, 1], repeat=k)))
        a = 1.0 if params.get("alpha", "face") == "face" else (2 ** k) ** .25
        ax_ = np.vstack([np.eye(k) * a, -np.eye(k) * a])
        M = np.vstack([f_, ax_]); cp = cp or max(3, k)
    elif typ == "boxbehnken":
        if k < 3: raise ValueError("Box-Behnken needs >= 3 factors")
        rows = []
        for i, j in itertools.combinations(range(k), 2):
            for a, b in itertools.product([-1, 1], repeat=2):
                r_ = np.zeros(k); r_[i], r_[j] = a, b; rows.append(r_)
        M = np.array(rows); cp = cp or 3
    else:
        raise ValueError("type must be full|fractional|ccd|boxbehnken")
    M = np.vstack([np.tile(M, (reps, 1)), np.zeros((cp, k))]) if cp else np.tile(M, (reps, 1))
    idx = np.arange(len(M)); std_order = idx + 1
    if params.get("randomize", True): idx = rng.permutation(len(M))
    M = M[idx]; std_order = std_order[idx]
    out = pd.DataFrame(M, columns=[f"{n}_coded" for n in names])
    for j, n in enumerate(names):
        lo, hi = fac[n]; out[n] = (lo + hi) / 2 + out[f"{n}_coded"] * (hi - lo) / 2
    out.insert(0, "Std order", std_order); out.insert(0, "Run", np.arange(1, len(M) + 1)); out["Response"] = np.nan
    path = f"{outdir}/design_{typ}.csv"; out.to_csv(path, index=False, encoding="utf-8-sig")
    res.table(L.t("Design matrix"), out)
    res.note(f"{len(out)} runs. Randomized run order is protection against time trends - do not execute in standard order. Fill the 'Response' column and analyze with analysis 'doe_analyze'.")
    res.summary.update({"runs": len(out), "design_csv": path})
    return res


def _lenth(eff):
    c = np.abs(eff); s0 = 1.5 * np.median(c); pse = 1.5 * np.median(c[c < 2.5 * s0]) if (c < 2.5 * s0).any() else s0
    d = len(c) / 3; me = stats.t.ppf(.975, d) * pse; gam = (1 + .95 ** (1 / len(c))) / 2; sme = stats.t.ppf(gam, d) * pse
    return pse, me, sme


def doe_analyze(df, params, L: Lang, outdir):
    """params: response, factors (list of columns - actual or coded values, 2-level or 3-level), degree (interaction order, default 2), quadratic (bool, for CCD/BB),
    goal (max|min|target), target (value)."""
    y = params["response"]; fac = list(params["factors"]); deg = int(params.get("degree", 2)); quad = bool(params.get("quadratic", False))
    d = df[[y] + fac].dropna().copy()
    info = {}
    for f in fac:
        lo, hi = d[f].min(), d[f].max(); ctr = (lo + hi) / 2; half = (hi - lo) / 2
        info[f] = (ctr, half); d[f + "_c"] = (d[f] - ctr) / half
    cf = [f + "_c" for f in fac]
    cols_extra = []
    if quad:
        for f in cf: d[f + "^2"] = d[f] ** 2; cols_extra.append(f + "^2")
    terms = build_terms(cf, deg) + [(c,) for c in cols_extra]
    m = OLS(d, y, [], cf + cols_extra, terms=terms)
    res = Result(f"DOE analysis: {y}")
    res.table(L.t("Summary of Fit"), _tr_cols(pd.DataFrame([{L.t("R-squared"): m.r2, L.t("Adj R-squared"): m.adj_r2, L.t("RMSE"): np.sqrt(m.mse) if m.dfe > 0 else np.nan, L.t("Observations"): m.n}]), L))
    nm_map = {c: c.replace("_c", "") for c in cf + cols_extra}
    def pretty(s):
        for a, b in nm_map.items(): s = s.replace(a, b)
        return s
    pt = m.params_table(); pt["Term"] = pt["Term"].map(pretty)
    eff = pt[pt["Term"] != "Intercept"].copy(); eff[L.t("Effect")] = 2 * eff["Estimate"]
    res.table(L.t("Parameter Estimates") + " (coded -1/+1; Effect = 2 x Coefficient)", _tr_cols(eff.rename(columns={"Estimate": "Coefficient"}), L))
    if m.dfe > 0:
        a_ = m.anova_table(); res.table(L.t("Analysis of Variance"), _tr_cols(a_, L))
        t3 = m.ss3(); t3["Source"] = t3["Source"].map(pretty); res.table(L.t("Effect Tests"), _tr_cols(t3, L))
    # center-point curvature test (if center points exist in 2-level design)
    is_center = (np.abs(d[cf].values).sum(1) == 0)
    if is_center.any() and not quad:
        yc = d.loc[is_center, y]; yf = d.loc[~is_center, y]
        if len(yc) > 1:
            se = np.sqrt(yc.var(ddof=1) * (1 / len(yc) + 1 / len(yf))); t_ = (yf.mean() - yc.mean()) / se
            res.table("Curvature test (factorial mean vs center mean)", pd.DataFrame([{"Diff": yf.mean() - yc.mean(), "t": t_, "p-value": 2 * stats.t.sf(abs(t_), len(yc) - 1)}]))
    # effects pareto / half-normal / Lenth
    ef_main = eff[~eff["Term"].str.contains(r"\^2")].copy()
    ev = ef_main[L.t("Effect")].values; labels = ef_main["Term"].values
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.6))
    ax = axs[0]
    if m.dfe > 0:
        tv = np.abs(ef_main["Estimate"].values / ef_main["Std Error"].values); order = np.argsort(tv)
        ax.barh(range(len(tv)), tv[order], color=[PALETTE[3] if v > stats.t.ppf(.975, m.dfe) else PALETTE[0] for v in tv[order]])
        ax.axvline(stats.t.ppf(.975, m.dfe), color="k", ls="--"); ax.set_xlabel("|t ratio|"); ax.set_yticks(range(len(tv))); ax.set_yticklabels(labels[order], fontsize=8)
        sig = [labels[i] for i in np.where(tv > stats.t.ppf(.975, m.dfe))[0]]
    else:
        pse, me, sme = _lenth(ev); order = np.argsort(np.abs(ev))
        ax.barh(range(len(ev)), np.abs(ev)[order], color=[PALETTE[3] if abs(v) > me else PALETTE[0] for v in np.abs(ev)[order]])
        ax.axvline(me, color="k", ls="--", label=f"ME={me:.3g}"); ax.axvline(sme, color="k", ls=":", label=f"SME={sme:.3g}"); ax.legend(fontsize=8)
        ax.set_yticks(range(len(ev))); ax.set_yticklabels(labels[order], fontsize=8); ax.set_xlabel("|Effect|")
        sig = [labels[i] for i in np.where(np.abs(ev) > me)[0]]
        res.table(L.t("Lenth PSE"), pd.DataFrame([{"PSE": pse, "ME (95%)": me, "SME (simultaneous)": sme}]))
        res.note("Saturated/unreplicated design: significance judged by Lenth's method (ME / SME), not by p-values.")
    ax.set_title(L.fig("Pareto of Effects"))
    ax = axs[1]; ae = np.sort(np.abs(ev)); q = stats.norm.ppf(.5 + .5 * (np.arange(1, len(ae) + 1) - .5) / len(ae))
    ax.plot(q, ae, "o", color=PALETTE[0]);
    for xq, yq, lb in zip(q, ae, labels[np.argsort(np.abs(ev))]): ax.annotate(lb, (xq, yq), fontsize=7, xytext=(3, 3), textcoords="offset points")
    sl = (ae[:max(2, len(ae) // 2)] @ q[:max(2, len(ae) // 2)]) / (q[:max(2, len(ae) // 2)] @ q[:max(2, len(ae) // 2)]); ax.plot(q, sl * q, color=PALETTE[3])
    ax.set_title(L.fig("Half-normal plot")); ax.set_xlabel("half-normal quantile"); ax.set_ylabel("|Effect|")
    fig.tight_layout(); res.fig(L.t("Pareto of Effects"), save_fig(fig, outdir, "doe_pareto"))
    # main effects & interaction plots
    k = len(fac)
    fig, axs = plt.subplots(1, k, figsize=(3.6 * k, 3.6), squeeze=False)
    for i, f in enumerate(fac):
        g = d.groupby(f)[y].mean(); axs[0, i].plot(g.index, g.values, "o-", color=PALETTE[0], lw=2); axs[0, i].axhline(d[y].mean(), color="#999", ls="--"); axs[0, i].set_title(f)
    fig.suptitle(L.fig("Main Effects")); fig.tight_layout(); res.fig(L.t("Main Effects"), save_fig(fig, outdir, "doe_main"))
    prs = list(itertools.combinations(range(k), 2))[:6]
    if prs:
        fig, axs = plt.subplots(1, len(prs), figsize=(4.2 * len(prs), 3.8), squeeze=False)
        for n_, (i, j) in enumerate(prs):
            a, b = fac[i], fac[j]; lo, hi = d[b].min(), d[b].max()
            for c, v in enumerate((lo, hi)):
                xs = np.sort(d[a].unique()); ys = []
                for xv in xs:
                    tmp = d[[f + "_c" for f in fac]].mean().to_frame().T * 0
                    tmp = pd.DataFrame([{f + "_c": 0.0 for f in fac}]); tmp[a + "_c"] = (xv - info[a][0]) / info[a][1]; tmp[b + "_c"] = (v - info[b][0]) / info[b][1]
                    for ce in cols_extra: tmp[ce] = tmp[ce.replace("^2", "")] ** 2
                    ys.append(float(m.predict_df(tmp)[0]))
                axs[0, n_].plot(xs, ys, "o-", color=PALETTE[c], label=f"{b}={v:g}")
            axs[0, n_].set_title(f"{a} x {b}"); axs[0, n_].legend(fontsize=7)
        fig.suptitle(L.fig("Interaction plot")); fig.tight_layout(); res.fig(L.t("Interaction plot"), save_fig(fig, outdir, "doe_interaction"))
    # optimisation in coded space
    goal = params.get("goal"); best = None
    def pred(z):
        tmp = pd.DataFrame([dict(zip(cf, z))])
        for ce in cols_extra: tmp[ce] = tmp[ce.replace("^2", "")] ** 2
        return float(m.predict_df(tmp)[0])
    if goal:
        obj = (lambda z: -pred(z)) if goal == "max" else ((lambda z: pred(z)) if goal == "min" else (lambda z: (pred(z) - float(params["target"])) ** 2))
        bestv = None
        for s in range(20):
            x0 = np.random.default_rng(s).uniform(-1, 1, k)
            r = optimize.minimize(obj, x0, bounds=[(-1, 1)] * k, method="L-BFGS-B")
            if bestv is None or r.fun < bestv.fun: bestv = r
        zs = bestv.x
        best = {f: info[f][0] + zs[i] * info[f][1] for i, f in enumerate(fac)}
        res.table("Optimum (within tested range)", pd.DataFrame([{**best, "Predicted " + y: pred(zs)}]))
        res.summary["optimum"] = {**best, "predicted": pred(zs)}
        res.note("Optimum is a model prediction inside the experimental region; confirm with 3-5 confirmation runs before adopting it.")
    if k == 2 or (k >= 2 and quad):
        a, b = fac[:2]; g = np.linspace(-1, 1, 40); Z = np.zeros((40, 40))
        for ii, u in enumerate(g):
            for jj, v in enumerate(g):
                z = np.zeros(k); z[0], z[1] = u, v; Z[jj, ii] = pred(z)
        fig, ax = plt.subplots(figsize=(6, 5)); cs = ax.contourf(info[a][0] + g * info[a][1], info[b][0] + g * info[b][1], Z, 15, cmap="viridis"); fig.colorbar(cs); ax.set_xlabel(a); ax.set_ylabel(b); ax.set_title(f"Contour: {y}")
        res.fig("Contour", save_fig(fig, outdir, "doe_contour"))
    if m.dfe > 0:
        pth, _ = _diag_plots(m, L, outdir, "doe"); res.fig(L.t("Residuals"), pth)
    res.note("Significant effects: " + (", ".join(sig) if sig else "none at 5%") + ".")
    res.summary.update({"significant": sig, "R2": m.r2, "effects": {l: float(v) for l, v in zip(labels, ev)}})
    return res
