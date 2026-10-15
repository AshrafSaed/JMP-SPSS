"""ss_doe2.py - Advanced DOE: Plackett-Burman, Definitive Screening (DSD), D-optimal custom designs, Taguchi orthogonal arrays,
mixture designs, Latin squares, design augmentation (foldover/axial/center), power for factorials, Taguchi S/N analysis,
Scheffe mixture analysis, multi-response Desirability optimisation with a Prediction-Profiler plot.
JMP: DOE > Screening / Custom / Mixture / Taguchi, Prediction Profiler (desirability).
"""
from __future__ import annotations
import itertools, math, string
import numpy as np, pandas as pd
from scipy import stats, optimize
import matplotlib.pyplot as plt
from ss_common import HE, Result, Lang, save_fig, PALETTE
from ss_linear import OLS, build_terms

HE.update({"Design of Experiments": "תכנון ניסויים", "Desirability optimisation": "אופטימיזציית Desirability", "Taguchi S/N analysis": "ניתוח Taguchi (S/N)",
           "Mixture analysis": "ניתוח תערובות", "Power for factorial designs": "עוצמה לתכנונים פקטוריאליים", "Design augmentation": "הרחבת תכנון",
           "Prediction profiler": "פרופיילר חיזוי", "Overall desirability": "Desirability כולל"})

# ----------------------------------------------------------------------------
# Constructions
# ----------------------------------------------------------------------------
PB_GEN = {12: "++-+++---+-", 20: "++--++++-+-+----++-", 24: "+++++-+-++--++--+-+----"}


def hadamard_sylvester(n):
    H = np.array([[1]])
    while H.shape[0] < n: H = np.block([[H, H], [H, -H]])
    return H


def plackett_burman(N):
    if N in PB_GEN:
        g = np.array([1 if c == "+" else -1 for c in PB_GEN[N]]); rows = [np.roll(g, i) for i in range(N - 1)]; M = np.vstack(rows + [-np.ones(N - 1)]); return M
    if N & (N - 1) == 0:
        H = hadamard_sylvester(N); return H[:, 1:]
    raise ValueError("Plackett-Burman supports N = 8, 12, 16, 20, 24, 32")


def _legendre(a, q):
    a %= q
    if a == 0: return 0
    return 1 if pow(a, (q - 1) // 2, q) == 1 else -1


def _isprime(n): return n > 1 and all(n % i for i in range(2, int(n ** .5) + 1))


def conference_matrix(order):
    q = order - 1
    if not _isprime(q): raise ValueError("no Paley conference matrix")
    Q = np.array([[_legendre(i - j, q) for j in range(q)] for i in range(q)]); sgn = 1 if q % 4 == 1 else -1
    C = np.zeros((order, order)); C[0, 1:] = 1; C[1:, 0] = sgn; C[1:, 1:] = Q; return C


def dsd_matrix(m):
    for order in range(max(m, 4), 60):
        if _isprime(order - 1) and (order - 1) % 2 == 1:
            C = conference_matrix(order)
            if np.allclose(C @ C.T, (order - 1) * np.eye(order)):
                Cm = C[:, :m]; return np.vstack([Cm, -Cm, np.zeros((1, m))]), order
    raise ValueError("DSD size not supported")


def gf_array(base, p):
    """Orthogonal array from GF(p) linear combinations: runs p^base, columns (p^base-1)/(p-1)."""
    pts = np.array(list(itertools.product(range(p), repeat=base))); cols = []
    for coef in itertools.product(range(p), repeat=base):
        if not any(coef): continue
        first = next(c for c in coef if c);
        if first != 1: continue
        cols.append((pts @ np.array(coef)) % p)
    return np.column_stack(cols)


TAGUCHI = {"L4": (2, 2), "L8": (2, 3), "L16": (2, 4), "L32": (2, 5), "L9": (3, 2), "L27": (3, 3)}


def d_optimal(cand, n, X_fn, rng, starts=15, max_iter=200):
    best = None
    for s in range(starts):
        idx = list(rng.choice(len(cand), n, replace=len(cand) < n)); X = X_fn(cand[idx]);
        for _ in range(max_iter):
            improved = False; M = X.T @ X
            try: Mi = np.linalg.inv(M)
            except np.linalg.LinAlgError: Mi = np.linalg.pinv(M)
            for i in range(n):
                xi = X[i]; Xc = X_fn(cand); vi = xi @ Mi @ xi; gains = np.einsum("ij,jk,ik->i", Xc, Mi, Xc); cross = Xc @ Mi @ xi
                delta = (1 + gains) * (1 - vi) + cross ** 2   # det ratio for swapping row i by candidate j
                j = int(np.argmax(delta))
                if delta[j] > 1 + 1e-9: idx[i] = j; X = X_fn(cand[idx]); M = X.T @ X; Mi = np.linalg.pinv(M); improved = True
            if not improved: break
        ld = np.linalg.slogdet(X.T @ X)[1]
        if best is None or ld > best[0]: best = (ld, list(idx))
    return best


def _model_fn(k, model):
    def f(Z):
        cols = [np.ones(len(Z))] + [Z[:, i] for i in range(k)]
        if model in ("interaction", "quadratic"): cols += [Z[:, i] * Z[:, j] for i, j in itertools.combinations(range(k), 2)]
        if model == "quadratic": cols += [Z[:, i] ** 2 for i in range(k)]
        return np.column_stack(cols)
    return f


def _decode(M, names, fac):
    out = pd.DataFrame(M, columns=[f"{n}_coded" for n in names])
    for n in names: lo, hi = fac[n]; out[n] = (lo + hi) / 2 + out[f"{n}_coded"] * (hi - lo) / 2
    return out


def _design_fig(out, cols, outdir, name, block=None, title="Design points"):
    """Scatter-matrix of the design (first 4 numeric columns) so coverage/orthogonality can be inspected visually."""
    cols = [c for c in cols if pd.api.types.is_numeric_dtype(out[c])][:4]; k = len(cols)
    if k < 2:
        fig, ax = plt.subplots(figsize=(4, 3)); ax.plot(out[cols[0]] if cols else range(len(out)), "o"); return save_fig(fig, outdir, name)
    fig, axs = plt.subplots(k - 1, k - 1, figsize=(2.6 * (k - 1), 2.6 * (k - 1)), squeeze=False); rng = np.random.default_rng(0)
    for i in range(k - 1):
        for j in range(k - 1):
            ax = axs[i, j]
            if j > i: ax.axis("off"); continue
            xa, ya = out[cols[j]].values.astype(float), out[cols[i + 1]].values.astype(float); jit = 0.02 * (np.ptp(xa) or 1)
            ax.scatter(xa + rng.normal(0, jit, len(xa)), ya + rng.normal(0, jit, len(ya)), s=22, c=None if block is None else [PALETTE[int(b) % 8] for b in block], alpha=.75, color=PALETTE[0] if block is None else None)
            ax.set_xlabel(cols[j], fontsize=8); ax.set_ylabel(cols[i + 1], fontsize=8)
    fig.suptitle(title); fig.tight_layout(); return save_fig(fig, outdir, name)


def make_design2(df, params, L: Lang, outdir):
    """type: plackett_burman | dsd | d_optimal | taguchi | mixture | latin_square.
    factors {name:[low,high]} (not for mixture/latin/taguchi labels); extra: runs (PB N / d_optimal n), model (main|interaction|quadratic), array (taguchi), components + degree/centroid (mixture), size (latin_square), seed, randomize."""
    typ = params.get("type", "dsd"); rng = np.random.default_rng(int(params.get("seed", 1))); res = Result("DOE - " + typ); fac = params.get("factors", {}); names = list(fac)
    if typ == "plackett_burman":
        k = len(names); N = int(params.get("runs") or next(n for n in (8, 12, 16, 20, 24, 32) if n - 1 >= k)); M = plackett_burman(N)[:, :k]
        out = _decode(M, names, fac); res.note(f"Plackett-Burman N={N}: main effects only (Resolution III); 2-factor interactions are partially aliased with main effects. Use for screening; follow with a larger design for the active few."); res.summary["runs"] = N
    elif typ == "dsd":
        k = len(names); M, order = dsd_matrix(k); extra = int(params.get("extra_center", 0)); M = np.vstack([M, np.zeros((extra, k))]) if extra else M; out = _decode(M, names, {n: fac[n] for n in names})
        res.table("DSD properties", pd.DataFrame([{"Factors": k, "Conference matrix order used": order, "Runs": len(M), "Min DSD runs (2k+1)": 2 * k + 1, "Main-effects orthogonal": bool(np.allclose((M.T @ M) - np.diag(np.diag(M.T @ M)), 0)), "Max |corr| main vs 2FI": float(max(abs(np.corrcoef(M[:, i], M[:, j] * M[:, l])[0, 1]) if M[:, j].std() * M[:, l].std() > 0 and M[:, i].std() > 0 else 0 for i in range(k) for j, l in itertools.combinations(range(k), 2) if i not in (j, l)) if k >= 3 else 0)}]))
        res.note("Definitive Screening Design (Jones & Nachtsheim 2011 construction via Paley conference matrices): 3 levels per factor, main effects orthogonal to each other and to all quadratic and two-factor-interaction terms. Runs may exceed 2k+1 when no prime-order conference matrix of order k exists; extra columns are dropped. Fit with doe_analyze quadratic=true / stepwise.")
        res.summary.update({"runs": len(M), "order": order})
    elif typ == "d_optimal":
        k = len(names); n = int(params["runs"]); model = params.get("model", "interaction"); lv = [-1, 0, 1] if model == "quadratic" else [-1, 1]; cand = np.array(list(itertools.product(lv, repeat=k)), float); fn = _model_fn(k, model)
        if fn(cand[:1]).shape[1] > n: raise ValueError(f"runs ({n}) < number of model terms ({fn(cand[:1]).shape[1]})")
        ld, idx = d_optimal(cand, n, fn, rng); M = cand[idx]; X = fn(M); p = X.shape[1]; deff = 100 * math.exp(ld / p) / n; Mi = np.linalg.pinv(X.T @ X); aeff = 100 * p / (n * np.trace(Mi))
        out = _decode(M, names, fac); res.table("Design efficiency", pd.DataFrame([{"Runs": n, "Model": model, "Parameters": p, "D-efficiency %": deff, "A-efficiency %": aeff, "Avg prediction variance (x sigma^2)": float(np.mean(np.einsum('ij,jk,ik->i', fn(cand), Mi, fn(cand))))}]))
        res.summary.update({"runs": n, "d_eff": float(deff)}); res.note("Fedorov exchange with 15 random starts on the coded candidate grid; efficiencies are relative to an orthogonal design of the same size (100% = optimal).")
    elif typ == "taguchi":
        arr = params.get("array") or None; k = len(names); need = k
        if arr is None: arr = next(a for a, (p, b) in TAGUCHI.items() if (p ** b - 1) // (p - 1) >= need and all(True for _ in [0]) and p == int(params.get("levels", 2)))
        p, b = TAGUCHI[arr]; A = gf_array(b, p);
        if k > A.shape[1]: raise ValueError(f"{arr} has only {A.shape[1]} columns")
        out = pd.DataFrame({n: [fac[n][int(v)] if isinstance(fac[n], (list, tuple)) and len(fac[n]) == p else int(v) + 1 for v in A[:, i]] for i, n in enumerate(names)}); res.table("Array", pd.DataFrame([{"Array": arr, "Levels": p, "Runs": A.shape[0], "Columns available": A.shape[1], "Factors assigned": k}]))
        res.note(f"{arr}: GF({p}) orthogonal array. Factors are assigned to the first {k} columns; interactions are confounded with other columns - use only if interactions are negligible or assign by the Taguchi linear graph. Give factors as {{name:[level1,level2,...]}}.")
        res.summary.update({"runs": int(A.shape[0]), "array": arr})
    elif typ == "mixture":
        comps = params["components"]; q = len(comps); deg = int(params.get("degree", 2)); lo = np.array(params.get("lower", [0] * q), float); pts = []
        if params.get("centroid", False):
            for r in range(1, q + 1):
                for c in itertools.combinations(range(q), r): v = np.zeros(q); v[list(c)] = 1 / r; pts.append(v)
        else:
            for c in itertools.product(range(deg + 1), repeat=q):
                if sum(c) == deg: pts.append(np.array(c) / deg)
        P = np.array(pts); P = np.vstack([P, np.ones(q) / q]) if params.get("add_centroid", True) else P; P = np.unique(np.round(P, 10), axis=0); real = lo + P * (1 - lo.sum())
        out = pd.DataFrame(real, columns=comps); res.note("Proportions sum to 1. With lower bounds the design is built in pseudo-components and rescaled. Analyse with 'mixture_analyze'."); res.summary["runs"] = len(out)
    elif typ == "latin_square":
        n = int(params.get("size", 4)); base = (np.add.outer(range(n), range(n)) % n); r = rng.permutation(n); c = rng.permutation(n); s = rng.permutation(n); LS = s[base[np.ix_(r, c)]]
        out = pd.DataFrame([{"Row (blocking 1)": i + 1, "Column (blocking 2)": j + 1, "Treatment": string.ascii_uppercase[LS[i, j]]} for i in range(n) for j in range(n)]); res.note("Randomized Latin square: each treatment appears once per row and column. Assumes no row x column x treatment interactions."); res.summary["runs"] = n * n
    else: raise ValueError("type must be plackett_burman|dsd|d_optimal|taguchi|mixture|latin_square")
    if typ not in ("latin_square", "taguchi", "mixture") and params.get("randomize", True):
        idx = rng.permutation(len(out)); out = out.iloc[idx].reset_index(drop=True)
    if typ == "mixture" and params.get("randomize", True): out = out.sample(frac=1, random_state=int(params.get("seed", 1))).reset_index(drop=True)
    out.insert(0, "Run", np.arange(1, len(out) + 1)); out["Response"] = np.nan; path = f"{outdir}/design_{typ}.csv"; out.to_csv(path, index=False, encoding="utf-8-sig"); res.table("Design matrix", out); res.summary["design_csv"] = path
    if typ == "latin_square":
        n_ = int(out["Row (blocking 1)"].max()); grid = np.array([[ord(t) - 65 for t in out[out["Row (blocking 1)"] == r_]["Treatment"]] for r_ in range(1, n_ + 1)]); fig, ax = plt.subplots(figsize=(3.2 + n_ * .3, 3.2 + n_ * .3)); ax.imshow(grid, cmap="Pastel1"); ax.grid(False)
        for r_ in range(n_):
            for c_ in range(n_): ax.text(c_, r_, string.ascii_uppercase[grid[r_, c_]], ha="center", va="center", fontsize=14)
        ax.set_xticks(range(n_)); ax.set_xticklabels(range(1, n_ + 1)); ax.set_yticks(range(n_)); ax.set_yticklabels(range(1, n_ + 1)); ax.set_title("Latin square layout"); res.fig("Latin square", save_fig(fig, outdir, "design_latin"))
    elif typ == "mixture" and len(out.columns) >= 5:
        cm = [c for c in out.columns if c not in ("Run", "Response")][:3]; P_ = out[cm].values; fig, ax = plt.subplots(figsize=(5, 4.4)); xx = P_[:, 1] + P_[:, 2] / 2; yy = P_[:, 2] * math.sqrt(3) / 2; ax.plot([0, 1, .5, 0], [0, 0, math.sqrt(3) / 2, 0], "k-"); ax.plot(xx, yy, "o", color=PALETTE[0]); ax.set_aspect("equal"); ax.axis("off")
        ax.text(-.04, -.04, cm[0]); ax.text(1.0, -.04, cm[1]); ax.text(.5, .9, cm[2]); ax.set_title("Mixture design points (ternary)"); res.fig("Mixture design", save_fig(fig, outdir, "design_mixture"))
    else:
        cc_ = [c for c in out.columns if c.endswith("_coded")] or [c for c in out.columns if c not in ("Run", "Response")]; res.fig("Design points", _design_fig(out, cc_, outdir, f"design_{typ}_pairs", title=f"{typ}: design points"))
    return res


# ----------------------------------------------------------------------------
# Augmentation
# ----------------------------------------------------------------------------
def doe_augment(df, params, L: Lang, outdir):
    """df: a design with *_coded columns (from make_design). operation foldover (all signs, or 'flip': one factor) | axial (alpha face|rotatable|number) | center (n)."""
    op = params.get("operation", "foldover"); cc = [c for c in df.columns if c.endswith("_coded")]; names = [c[:-6] for c in cc]; M = df[cc].values.astype(float); res = Result(L.t("Design augmentation"))
    if op == "foldover":
        Mf = M.copy();
        if params.get("flip"): Mf[:, names.index(params["flip"])] *= -1
        else: Mf = -Mf
        new = Mf; note = "Full foldover de-aliases all main effects from two-factor interactions (Res III -> Res IV)." if not params.get("flip") else f"Single-factor foldover on {params['flip']}: de-aliases that factor's main effect and its two-factor interactions."
    elif op == "axial":
        k = len(names); a = params.get("alpha", "face"); a = 1.0 if a == "face" else ((2 ** k) ** .25 if a == "rotatable" else float(a)); new = np.vstack([np.eye(k) * a, -np.eye(k) * a]); note = f"Axial points at +/-{a:.4g} added (CCD)."
    else: new = np.zeros((int(params.get("n", 3)), len(names))); note = "Center points added (curvature check / pure error)."
    fac = params.get("factors")
    if fac is None:
        fac = {}
        for n in names:
            if n in df.columns:
                lo, hi = df[n].min(), df[n].max(); fac[n] = [lo, hi]
    add = _decode(new, names, fac); allr = pd.concat([pd.DataFrame(M, columns=[f"{n}_coded" for n in names]).pipe(lambda t: pd.concat([t, _decode(M, names, fac)[names]], axis=1)), add], ignore_index=True)
    allr.insert(0, "Block", [1] * len(M) + [2] * len(new)); allr.insert(0, "Run", np.arange(1, len(allr) + 1)); allr["Response"] = np.nan
    path = f"{outdir}/design_augmented.csv"; allr.to_csv(path, index=False, encoding="utf-8-sig"); res.table("Augmented design (Block 2 = new runs)", allr); res.note(note + " Include 'Block' as an effect when analysing (run in a different time period).")
    res.fig("Augmented design (colour = block)", _design_fig(allr, [c for c in allr.columns if c.endswith("_coded")], outdir, "design_augmented_pairs", block=allr["Block"].values - 1, title="Augmented design"))
    res.summary.update({"runs_total": len(allr), "runs_added": len(new), "design_csv": path}); return res


# ----------------------------------------------------------------------------
# Power
# ----------------------------------------------------------------------------
def doe_power(df, params, L: Lang, outdir):
    """No data. sigma (error SD), effect (list; difference between high and low means), runs (list), terms (model parameters incl. intercept), alpha."""
    sg = float(params["sigma"]); effs = params.get("effect", [sg, 2 * sg]); runs = params.get("runs", [8, 12, 16, 24, 32]); p = int(params.get("terms", 4)); a = float(params.get("alpha", .05)); rows = []
    for N in runs:
        dfe = N - p
        if dfe < 1: continue
        for eff in effs:
            se = 2 * sg / math.sqrt(N); nc = eff / se; tc = stats.t.ppf(1 - a / 2, dfe); pw = stats.nct.sf(tc, dfe, nc) + stats.nct.cdf(-tc, dfe, nc); rows.append({"Runs N": N, "Error df": dfe, "Effect": eff, "Effect/sigma": eff / sg, "SE of effect": se, "Power": pw})
    res = Result(L.t("Power for factorial designs")); t = pd.DataFrame(rows); res.table("Power of two-level orthogonal design", t)
    need = []
    for eff in effs:
        for N in range(p + 1, 2000):
            se = 2 * sg / math.sqrt(N); tc = stats.t.ppf(1 - a / 2, N - p); pw = stats.nct.sf(tc, N - p, eff / se) + stats.nct.cdf(-tc, N - p, eff / se)
            if pw >= .8: need.append({"Effect": eff, "Effect/sigma": eff / sg, "Runs for 80% power": N}); break
    res.table("Runs needed for 80% power (continuous N; round up to a design size)", pd.DataFrame(need))
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    for eff in effs: s_ = t[t["Effect"] == eff]; ax.plot(s_["Runs N"], s_["Power"], "o-", label=f"effect={eff:g}")
    ax.axhline(.8, color="grey", ls=":"); ax.set_xlabel("runs"); ax.set_ylabel("power"); ax.legend(fontsize=7); ax.set_title(L.fig("Power for factorial designs")); res.fig("DOE power", save_fig(fig, outdir, "doe_power"))
    res.summary["runs_for_80"] = need[0]["Runs for 80% power"] if need else None; res.note("Effect = (mean at +1) - (mean at -1); orthogonal two-level design, effect SE = 2 sigma/sqrt(N). Power for the t-test of one effect with error df = N - terms (needs replication or pure-error df)."); return res


# ----------------------------------------------------------------------------
# Taguchi S/N analysis
# ----------------------------------------------------------------------------
def taguchi_analyze(df, params, L: Lang, outdir):
    """factors (list of columns); responses (list of replicate columns) ; sn larger|smaller|nominal ; ANOVA of S/N."""
    fs, rs = params["factors"], params["responses"]; sn = params.get("sn", "larger"); d = df[fs + rs].dropna().copy(); Y = d[rs].values.astype(float)
    if sn == "larger": d["SN"] = -10 * np.log10(np.mean(1 / Y ** 2, axis=1))
    elif sn == "smaller": d["SN"] = -10 * np.log10(np.mean(Y ** 2, axis=1))
    else: d["SN"] = 10 * np.log10(Y.mean(1) ** 2 / Y.var(1, ddof=1))
    d["Mean"] = Y.mean(1); res = Result(L.t("Taguchi S/N analysis")); rows = []
    for f in fs:
        g = d.groupby(f)["SN"].mean(); rows.append({"Factor": f, **{f"S/N @ {l}": v for l, v in g.items()}, "Delta": g.max() - g.min(), "Best level": g.idxmax()})
    t = pd.DataFrame(rows); t["Rank"] = t["Delta"].rank(ascending=False).astype(int); res.table(f"Response table for S/N ({sn}-the-better; higher S/N is better)", t)
    m = OLS(d.assign(**{f: d[f].astype(str) for f in fs}), "SN", factors=fs);
    if m.dfe > 0: res.table("ANOVA on S/N", m.ss3())
    fig, axs = plt.subplots(1, len(fs), figsize=(3.4 * len(fs), 3.5), squeeze=False)
    for i, f in enumerate(fs): g = d.groupby(f)["SN"].mean(); axs[0, i].plot(range(len(g)), g.values, "o-"); axs[0, i].set_xticks(range(len(g))); axs[0, i].set_xticklabels(g.index.astype(str)); axs[0, i].axhline(d["SN"].mean(), color="grey", ls=":"); axs[0, i].set_title(f)
    fig.suptitle(L.fig("Taguchi S/N analysis")); fig.tight_layout(); res.fig("S/N main effects", save_fig(fig, outdir, "taguchi_sn"))
    best = {r["Factor"]: r["Best level"] for r in rows}; pred = d["SN"].mean() + sum(d.groupby(f)["SN"].mean()[best[f]] - d["SN"].mean() for f in fs); res.table("Predicted optimum (additive model)", pd.DataFrame([{**best, "Predicted S/N": pred}]))
    res.summary.update({"best_levels": {k: str(v) for k, v in best.items()}, "pred_sn": float(pred)}); res.note("Additive (no-interaction) prediction: confirm with a verification run. S/N formulas: larger = -10log10(mean(1/y^2)); smaller = -10log10(mean(y^2)); nominal = 10log10(ybar^2/s^2)."); return res


# ----------------------------------------------------------------------------
# Mixture (Scheffe) analysis
# ----------------------------------------------------------------------------
def _scheffe_X(P, model):
    q = P.shape[1]; cols = [P[:, i] for i in range(q)]
    if model in ("quadratic", "special_cubic"): cols += [P[:, i] * P[:, j] for i, j in itertools.combinations(range(q), 2)]
    if model == "special_cubic": cols += [P[:, i] * P[:, j] * P[:, k] for i, j, k in itertools.combinations(range(q), 3)]
    return np.column_stack(cols)


def mixture_analyze(df, params, L: Lang, outdir):
    """components (list); response; model linear|quadratic|special_cubic; goal max|min|target (+target)."""
    comps, y = params["components"], params["response"]; model = params.get("model", "quadratic"); d = df[comps + [y]].dropna(); P = d[comps].values.astype(float); P = P / P.sum(1, keepdims=True); yv = d[y].values.astype(float); X = _scheffe_X(P, model); n, p = X.shape
    b = np.linalg.lstsq(X, yv, rcond=None)[0]; r = yv - X @ b; dfe = n - p; mse = r @ r / dfe if dfe > 0 else np.nan; cov = mse * np.linalg.pinv(X.T @ X); se = np.sqrt(np.diag(cov)); ssm = ((X @ b - yv.mean()) ** 2).sum(); sst = ((yv - yv.mean()) ** 2).sum()
    names = list(comps) + [f"{comps[i]}*{comps[j]}" for i, j in itertools.combinations(range(len(comps)), 2)] * (model != "linear") + ([f"{comps[i]}*{comps[j]}*{comps[k]}" for i, j, k in itertools.combinations(range(len(comps)), 3)] if model == "special_cubic" else [])
    names = names[:p]; res = Result(L.t("Mixture analysis"))
    res.table("Scheffe model coefficients (no intercept)", pd.DataFrame({"Term": names, "Coefficient": b, "Std Error": se, "t": b / se if dfe > 0 else np.nan, "p-value": 2 * stats.t.sf(abs(b / se), dfe) if dfe > 0 else np.nan}))
    F = (ssm / (p - 1)) / mse if dfe > 0 else np.nan; res.table("Fit", pd.DataFrame([{"Model": model, "n": n, "Terms": p, "R2": 1 - (r @ r) / sst, "Adj R2": 1 - (r @ r / dfe) / (sst / (n - 1)) if dfe > 0 else np.nan, "RMSE": math.sqrt(mse) if dfe > 0 else np.nan, "F (model)": F, "p": stats.f.sf(F, p - 1, dfe) if dfe > 0 else np.nan}]))
    goal = params.get("goal", "max"); q = len(comps); lo = np.array(params.get("lower", [0] * q), float); hi = np.array(params.get("upper", [1] * q), float)
    f = lambda z: float((_scheffe_X(z[None, :], model) @ b)[0]); obj = (lambda z: -f(z)) if goal == "max" else ((lambda z: f(z)) if goal == "min" else (lambda z: (f(z) - float(params["target"])) ** 2)); bestv = None
    for s in range(30):
        x0 = np.random.default_rng(s).dirichlet(np.ones(q)); r_ = optimize.minimize(obj, x0, method="SLSQP", bounds=list(zip(lo, hi)), constraints=[{"type": "eq", "fun": lambda z: z.sum() - 1}])
        if r_.success and (bestv is None or r_.fun < bestv.fun): bestv = r_
    if bestv is not None: res.table("Optimum blend (within bounds)", pd.DataFrame([{**dict(zip(comps, bestv.x)), "Predicted " + y: f(bestv.x)}])); res.summary["optimum"] = {c: float(v) for c, v in zip(comps, bestv.x)}
    if q == 3:
        g = np.array([(a, b_, 1 - a - b_) for a in np.linspace(0, 1, 60) for b_ in np.linspace(0, 1, 60) if a + b_ <= 1]); zz = _scheffe_X(g, model) @ b; xx = g[:, 1] + g[:, 2] / 2; yy = g[:, 2] * math.sqrt(3) / 2
        fig, ax = plt.subplots(figsize=(5.8, 5)); cs = ax.tricontourf(xx, yy, zz, 14, cmap="viridis"); fig.colorbar(cs); xp = P[:, 1] + P[:, 2] / 2; yp = P[:, 2] * math.sqrt(3) / 2; ax.plot(xp, yp, "k.", ms=4)
        ax.text(-.04, -.04, comps[0]); ax.text(1.0, -.04, comps[1]); ax.text(.5, .9, comps[2]); ax.set_aspect("equal"); ax.axis("off"); ax.set_title(L.fig("Mixture analysis")); res.fig("ternary contour", save_fig(fig, outdir, "mixture_ternary"))
    else:
        fig, ax = plt.subplots(figsize=(6, 3.8)); ax.scatter(X @ b, yv); ax.plot([yv.min(), yv.max()], [yv.min(), yv.max()], color=PALETTE[3]); ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); res.fig("fit", save_fig(fig, outdir, "mixture_fit"))
    res.summary.update({"r2": float(1 - (r @ r) / sst)}); res.note("Scheffe polynomials have no intercept; coefficients of linear terms are the predicted response at the pure components. Use component bounds if the pure blends are infeasible."); return res


# ----------------------------------------------------------------------------
# Desirability optimisation + prediction profiler
# ----------------------------------------------------------------------------
def _d_single(v, g):
    goal = g["goal"]; w = float(g.get("weight", 1))
    if goal == "max": lo, hi = g["low"], g["high"]; d = np.clip((v - lo) / (hi - lo), 0, 1) ** w
    elif goal == "min": lo, hi = g["low"], g["high"]; d = np.clip((hi - v) / (hi - lo), 0, 1) ** w
    else:
        lo, t, hi = g["low"], g["target"], g["high"]; d = np.where(v <= t, np.clip((v - lo) / (t - lo), 0, 1), np.clip((hi - v) / (hi - t), 0, 1)) ** w
    return d


def doe_optimize(df, params, L: Lang, outdir):
    """factors (list of columns), responses {name: {goal: max|min|target, low, high, [target], [weight], [importance]}}, degree (1 or 2; default 2), quadratic (bool)."""
    fac, rsp = list(params["factors"]), params["responses"]; deg = int(params.get("degree", 2)); quad = bool(params.get("quadratic", False)); d = df[fac + list(rsp)].dropna().copy(); info = {}
    for f in fac: lo, hi = d[f].min(), d[f].max(); info[f] = ((lo + hi) / 2, (hi - lo) / 2); d[f + "_c"] = (d[f] - info[f][0]) / info[f][1]
    cf = [f + "_c" for f in fac]; extra = []
    if quad:
        for c in cf: d[c + "^2"] = d[c] ** 2; extra.append(c + "^2")
    terms = build_terms(cf, deg) + [(c,) for c in extra]; models = {r: OLS(d, r, [], cf + extra, terms=terms) for r in rsp}; k = len(fac)
    def pred_all(z):
        tmp = pd.DataFrame([dict(zip(cf, z))])
        for e_ in extra: tmp[e_] = tmp[e_.replace("^2", "")] ** 2
        return {r: float(m.predict_df(tmp)[0]) for r, m in models.items()}
    imp = {r: float(g.get("importance", 1)) for r, g in rsp.items()}
    def overall(z):
        pr = pred_all(z); ds = {r: float(_d_single(pr[r], g)) for r, g in rsp.items()}; return float(np.prod([max(ds[r], 1e-12) ** imp[r] for r in rsp]) ** (1 / sum(imp.values()))), ds, pr
    best = None
    for s in range(40):
        x0 = np.random.default_rng(s).uniform(-1, 1, k); r_ = optimize.minimize(lambda z: -overall(z)[0], x0, bounds=[(-1, 1)] * k, method="L-BFGS-B")
        if best is None or r_.fun < best.fun: best = r_
    z = best.x; D, ds, pr = overall(z); res = Result(L.t("Desirability optimisation"))
    res.table("Optimal factor settings", pd.DataFrame([{**{f: info[f][0] + z[i] * info[f][1] for i, f in enumerate(fac)}, "Overall desirability": D}]))
    res.table("Predicted responses at optimum", pd.DataFrame([{"Response": r, "Predicted": pr[r], "Desirability": ds[r], "Goal": rsp[r]["goal"], "Low": rsp[r].get("low"), "High": rsp[r].get("high"), "Target": rsp[r].get("target"), "Model R2": models[r].r2} for r in rsp]))
    res.table("Model quality", pd.DataFrame([{"Response": r, "R2": m.r2, "Adj R2": m.adj_r2, "RMSE": math.sqrt(m.mse) if m.dfe > 0 else np.nan, "Error df": m.dfe} for r, m in models.items()]))
    nr = len(rsp) + 1; fig, axs = plt.subplots(nr, k, figsize=(3.2 * k, 2.2 * nr), squeeze=False); g = np.linspace(-1, 1, 41)
    for j, f in enumerate(fac):
        dd = []
        for i, r in enumerate(rsp):
            ys = []
            for v in g: zz = z.copy(); zz[j] = v; ys.append(pred_all(zz)[r])
            axs[i, j].plot(info[f][0] + g * info[f][1], ys, color=PALETTE[0]); axs[i, j].axvline(info[f][0] + z[j] * info[f][1], color=PALETTE[3], ls="--"); axs[i, j].axhline(pr[r], color="grey", lw=.5)
            if j == 0: axs[i, j].set_ylabel(r, fontsize=8)
        ov = []
        for v in g: zz = z.copy(); zz[j] = v; ov.append(overall(zz)[0])
        axs[-1, j].plot(info[f][0] + g * info[f][1], ov, color=PALETTE[2]); axs[-1, j].axvline(info[f][0] + z[j] * info[f][1], color=PALETTE[3], ls="--"); axs[-1, j].set_xlabel(f); axs[-1, j].set_ylim(0, 1.02)
        if j == 0: axs[-1, j].set_ylabel("Desirability", fontsize=8)
    fig.suptitle(L.fig("Prediction profiler")); fig.tight_layout(); res.fig("Prediction profiler", save_fig(fig, outdir, "doe_profiler"))
    res.summary.update({"optimum": {f: float(info[f][0] + z[i] * info[f][1]) for i, f in enumerate(fac)}, "overall_desirability": D, "predicted": pr})
    res.note("Derringer-Suich desirability; the optimum is a model prediction inside the experimental region - run 3-5 confirmation runs. Weak models (low R2) make the optimum unreliable."); return res
