"""Verification suite 3: the limitation fixes (Satterthwaite df, exact-ML SARIMA, ADF/KPSS, Grambsch-Therneau PH test, multi-factor MANOVA,
GLMM, REML Gage R&R, split-plot design + analysis, reference coding).  Each check compares to an independent computation or simulation truth.
Usage: python3 -I tests/run_tests3.py <outdir>"""
import json, math, os, subprocess, sys, tempfile
import numpy as np, pandas as pd
from scipy import stats, linalg, integrate
HERE = os.path.dirname(os.path.abspath(__file__)); S = os.path.join(HERE, "..", "scripts"); sys.path.insert(0, S)
OUT = sys.argv[1] if len(sys.argv) > 1 else tempfile.mkdtemp(); os.makedirs(OUT, exist_ok=True)
from ss_common import Lang
import ss_models2 as M2, ss_ts2 as TS, ss_rel2 as R2, ss_stats2 as S2, ss_ext as EX, ss_doe2 as D2, ss_spc as C1, ss_linear
L = Lang("en"); rng = np.random.default_rng(21); ok, bad = [], []


def check(name, cond, detail=""):
    (ok if cond else bad).append(name); print(("PASS " if cond else "FAIL ") + name + (f"  [{detail}]" if detail else ""))


def T(res, key):
    for n, t in res.tables:
        if key.lower() in n.lower(): return t
    raise KeyError(key + " in " + str([n for n, _ in res.tables]))


def O(name): d = os.path.join(OUT, name); os.makedirs(d, exist_ok=True); return d


# ============================ Satterthwaite df ============================
a_, n_ = 8, 5; g = np.repeat(range(a_), n_); y = 10 + rng.normal(0, 2, a_)[g] + rng.normal(0, 1, a_ * n_); d1 = pd.DataFrame({"y": y, "g": g.astype(str)})
r = M2.mixed(d1, {"response": "y", "random": ["g"]}, L, O("sat1")); ft = T(r, "Fixed effects (t")
check("Satterthwaite: balanced one-way random model, intercept df == a-1", abs(ft["DF"][0] - (a_ - 1)) < 1e-3, f"df={ft['DF'][0]:.4f}")
gm = d1.groupby("g").y.mean(); msb = n_ * ((gm - y.mean()) ** 2).sum() / (a_ - 1)
check("Satterthwaite t == classical (ybar / sqrt(MSB/an))", abs(ft["t"][0] - y.mean() / math.sqrt(msb / (a_ * n_))) < 1e-3, f"{ft['t'][0]:.4f}")
# balanced split plot: A (whole plot, 2 levels), B (sub plot, 3 levels), 8 whole plots per A level
W = 8; rows = []
for a in "xy":
    for w in range(W):
        wpe = rng.normal(0, 1.5)
        for b in "pqr": rows.append({"A": a, "B": b, "WP": f"{a}{w}", "y": 5 + (a == "x") * 1.2 + {"p": 0, "q": .8, "r": 1.5}[b] + wpe + rng.normal(0, .7)})
sp = pd.DataFrame(rows); N = len(sp)
# classical split-plot ANOVA
gmn = sp.y.mean(); mA = sp.groupby("A").y.mean(); wpm = sp.groupby("WP").y.mean(); nb = 3
ssA = W * nb * ((mA - gmn) ** 2).sum(); wp_a = sp.groupby("WP").A.first(); ssWP = nb * ((wpm - wp_a.map(mA)) ** 2).sum(); dfWP = 2 * (W - 1); F_A = (ssA / 1) / (ssWP / dfWP)
rs = EX.split_plot_analyze(sp, {"response": "y", "whole_plot": ["A"], "subplot": ["B"], "plot": "WP"}, L, O("sp")); es = T(rs, "Effects by error stratum"); ra = es[es.Effect == "A"].iloc[0]
check("Split-plot: whole-plot effect F and error df == classical split-plot ANOVA", abs(ra["F"] - F_A) / F_A < 0.02 and abs(ra["Den DF (Satterthwaite)"] - dfWP) < 0.5 and ra["Stratum"] == "whole plot", f"F={ra['F']:.3f} vs {F_A:.3f}; df={ra['Den DF (Satterthwaite)']:.2f} vs {dfWP}")
rb = es[es.Effect == "B"].iloc[0]; check("Split-plot: sub-plot effect tested on residual df a(W-1)(b-1) = 28", abs(rb["Den DF (Satterthwaite)"] - 2 * (W - 1) * 2) < 0.5 and rb["Stratum"] == "sub plot", f"df={rb['Den DF (Satterthwaite)']:.2f}")
# RCBD: fixed effect df == (b-1)(t-1), F == classical
blk = np.repeat(range(10), 3); trt = np.tile(list("ABC"), 10); yb = 5 + rng.normal(0, 1.5, 10)[blk] + np.tile([0, 1, 2.5], 10) + rng.normal(0, .5, 30); dfb = pd.DataFrame({"y": yb, "trt": trt, "blk": blk.astype(str)})
rb_ = M2.mixed(dfb, {"response": "y", "fixed": ["trt"], "random": ["blk"]}, L, O("rcbd")); fte = T(rb_, "Fixed-effect tests")
mt = dfb.groupby("trt").y.mean(); mb = dfb.groupby("blk").y.mean(); sst = 10 * ((mt - dfb.y.mean()) ** 2).sum(); sse = ((dfb.y - dfb.trt.map(mt) - dfb.blk.map(mb) + dfb.y.mean()) ** 2).sum(); Fc = (sst / 2) / (sse / 18)
check("RCBD mixed: treatment F == classical RCBD F, den df == 18", abs(fte["F"][0] - Fc) / Fc < 5e-3 and abs(fte.iloc[0, 2] - 18) < .05, f"F={fte['F'][0]:.4f} vs {Fc:.4f}; df={fte.iloc[0,2]:.3f}")
rw_ = M2.mixed(d1, {"response": "y", "random": ["g"], "df_method": "wald"}, L, O("wald")); check("df_method='wald' still available", "Wald" in rw_.tables[1][0])

# ============================ exact-ML SARIMA ============================
def acov(phi, th, nlag, K=3000):
    psi = np.zeros(K); psi[0] = 1
    for j in range(1, K): psi[j] = (th[j - 1] if j - 1 < len(th) else 0.0) + sum(ph * psi[j - i] for i, ph in enumerate(phi, 1) if j - i >= 0)
    return np.array([psi[:K - k] @ psi[k:] for k in range(nlag)])
okk = True
for phi, th in [([.6], []), ([], [.5]), ([.5], [.4]), ([.5, -.2], [.3]), ([], [.3, .2])]:
    z = rng.normal(size=60); nll, s2, _ = TS.arma_exact_nll(z, np.array(phi), np.array(th)); Sg = linalg.toeplitz(acov(phi, th, 60) * s2); ld = np.linalg.slogdet(Sg)[1]
    okk &= abs(-nll - (-0.5 * (60 * math.log(2 * math.pi) + ld + z @ np.linalg.solve(Sg, z)))) < 1e-6
check("Kalman exact ARMA log-likelihood == multivariate-normal Toeplitz likelihood (5 models)", okk)
z = rng.normal(size=80); ph = .7; nll, s2, _ = TS.arma_exact_nll(z, np.array([ph]), np.array([]))
ll_cf = -.5 * (80 * math.log(2 * math.pi * s2) + math.log(1 / (1 - ph ** 2)) + (1 - ph ** 2) * z[0] ** 2 / s2 + (np.sum((z[1:] - ph * z[:-1]) ** 2)) / s2)
check("AR(1) exact likelihood == closed form", abs(-nll - ll_cf) < 1e-8)
n = 800; e = rng.normal(size=n + 50); zz = np.zeros(n + 50)
for t in range(1, n + 50): zz[t] = .6 * zz[t - 1] + e[t] + .4 * e[t - 1]
f = TS.sarima_ml(zz[50:] + 3, 1, 0, 1); se = f["native_se"]; tr = np.array([.6, .4, 3.0])
check("Exact-ML ARMA(1,1) recovers phi=.6, theta=.4, mean=3 (within 3.5 SE)", np.all(np.abs(f["native"] - tr) < 3.5 * se), f"{np.round(f['native'],3)} se {np.round(se,3)}")
n = 360; m = 12; e = rng.normal(size=n + 60); w = np.zeros(n + 60)
for t in range(13, n + 60): w[t] = e[t] - .4 * e[t - 1] - .6 * e[t - 12] + .24 * e[t - 13]
yy = np.zeros(n + 60)
for t in range(13, n + 60): yy[t] = w[t] + yy[t - 1] + yy[t - 12] - yy[t - 13]
fa = TS.sarima_ml(yy[60:], 0, 1, 1, 0, 1, 1, 12); check("Seasonal 'airline' SARIMA(0,1,1)(0,1,1)12 recovers theta=-.4, Theta=-.6 (3.5 SE)", np.all(np.abs(fa["native"] - np.array([-.4, -.6])) < 3.5 * fa["native_se"]), f"{np.round(fa['native'],3)} se {np.round(fa['native_se'],3)}")
fc_, se_ = M2.arima_forecast(fa, 12); check("Seasonal forecast finite, widening intervals", np.all(np.isfinite(fc_)) and se_[-1] > se_[0])
a1 = np.zeros(400)
for t in range(1, 400): a1[t] = .7 * a1[t - 1] + rng.normal()
fm = TS.sarima_ml(a1, 1, 0, 0); fcs = M2.arima_fit(a1, 1, 0, 0); check("Exact ML and CSS agree for a long AR(1)", abs(fm["native"][0] - fcs["phi"][0]) < .02 and abs(fm["native"][0] - .7) < 3.5 * fm["native_se"][0])
rt = M2.timeseries(pd.DataFrame({"v": yy[60:]}), {"column": "v", "mode": "arima", "p": 0, "d": 1, "q": 1, "P": 0, "D": 1, "Q": 1, "period": 12, "horizon": 12}, L, O("sarima")); check("timeseries(mode=arima) uses exact ML + seasonal terms", "exact Gaussian" in rt.tables[0][0] and "SARIMA" in rt.tables[0][0])

# ============================ ADF / KPSS ============================
crit500 = np.percentile([TS._adf_stat(np.cumsum(rng.normal(size=500)), 2, "c")[0] for _ in range(4000)], [1, 5, 10]); check("ADF Monte-Carlo critical values (n=500, const) match published asymptotic -3.43/-2.86/-2.57", np.all(np.abs(crit500 - np.array([-3.43, -2.86, -2.57])) < .09), str(np.round(crit500, 3)))
crit_ct = np.percentile([TS._adf_stat(np.cumsum(rng.normal(size=500)), 2, "ct")[0] for _ in range(4000)], [1, 5, 10]); check("ADF critical values (const+trend) match -3.96/-3.41/-3.12", np.all(np.abs(crit_ct - np.array([-3.96, -3.41, -3.12])) < .1), str(np.round(crit_ct, 3)))
rej0 = np.mean([TS.adf_test(np.cumsum(rng.normal(size=150)), reps=300, seed=i)["p"] < .05 for i in range(120)]); check("ADF size under random walk ~5%", .01 <= rej0 <= .12, f"{rej0:.3f}")
def ar1(phi, n):
    x = np.zeros(n)
    for t in range(1, n): x[t] = phi * x[t - 1] + rng.normal()
    return x
pw = np.mean([TS.adf_test(ar1(.5, 200), reps=300, seed=i)["p"] < .05 for i in range(60)]); check("ADF power against AR(0.5) (n=200) > 95%", pw > .95, f"{pw:.2f}")
k_size = np.mean([TS.kpss_test(rng.normal(size=200))["stat"] > .463 for _ in range(600)]); check("KPSS size (level) ~5% at its 5% critical value", .02 <= k_size <= .09, f"{k_size:.3f}")
k_size2 = np.mean([TS.kpss_test(rng.normal(size=200) + .02 * np.arange(200), "ct")["stat"] > .146 for _ in range(600)]); check("KPSS size (trend) ~5%", .02 <= k_size2 <= .09, f"{k_size2:.3f}")
check("KPSS rejects a random walk", TS.kpss_test(np.cumsum(rng.normal(size=300)))["p"] <= .02)
rs_ = TS.stationarity(pd.DataFrame({"v": np.cumsum(rng.normal(size=250))}), {"column": "v", "mc_reps": 400}, L, O("stat")); check("stationarity verdict flags a random walk as non-stationary", "NON-STATIONARY" in rs_.summary["verdict"])
rs2 = TS.stationarity(pd.DataFrame({"v": ar1(.4, 250)}), {"column": "v", "mc_reps": 400}, L, O("stat2")); check("stationarity verdict flags AR(0.4) as stationary", "STATIONARY" in rs2.summary["verdict"] and "NON" not in rs2.summary["verdict"])

# ============================ Cox PH test ============================
def simcox(nonph, n=250):
    x = rng.integers(0, 2, n).astype(float)
    t = np.where(x == 1, rng.weibull(3.0, n), rng.weibull(1.0, n)) if nonph else rng.exponential(1, n) / np.exp(.7 * x)
    c = rng.exponential(2.5, n); return x[:, None], np.minimum(t, c), (t <= c).astype(int)
pph, pnon = [], []
for _ in range(150):
    X, t, e = simcox(False); b, cov, *_ = R2.cox_fit(X, t, e); pph.append(R2.schoenfeld_test(X, t, e, b, cov)["p_global"])
    X, t, e = simcox(True); b, cov, *_ = R2.cox_fit(X, t, e); pnon.append(R2.schoenfeld_test(X, t, e, b, cov)["p_global"])
check("Grambsch-Therneau test: size ~5% under proportional hazards", .01 <= np.mean(np.array(pph) < .05) <= .11, f"{np.mean(np.array(pph) < .05):.3f}")
check("Grambsch-Therneau test: power > 90% for crossing hazards", np.mean(np.array(pnon) < .05) > .9, f"{np.mean(np.array(pnon) < .05):.3f}")
check("PH p-values ~uniform under H0 (KS p > .01)", stats.kstest(pph, "uniform").pvalue > .01, f"KS p={stats.kstest(pph, 'uniform').pvalue:.3f}")
Xc, tc_, ec_ = simcox(True); dd = pd.DataFrame({"t": tc_, "e": ec_, "x": Xc[:, 0]}); rcx = R2.cox_ph(dd, {"time": "t", "event": "e", "covariates": ["x"]}, L, O("coxph")); check("cox_ph reports the PH test table and flags non-PH", "Grambsch" in rcx.tables[2][0] and rcx.summary["ph_global_p"] < .05)

# ============================ multi-factor MANOVA ============================
a, b, rep = 2, 3, 12; rows = []
for i in range(a):
    for j in range(b):
        for _ in range(rep): rows.append({"A": f"a{i}", "B": f"b{j}", "Y1": rng.normal() + (i == 0) * .8 + (j == 0) * .5, "Y2": rng.normal() + (j == 1) * .9})
dm = pd.DataFrame(rows); dm["Y2"] += .5 * dm.Y1
rm = S2.manova(dm, {"responses": ["Y1", "Y2"], "factors": ["A", "B"]}, L, O("manova2")); tm = T(rm, "Multivariate tests")
Y = dm[["Y1", "Y2"]].values; cellm = dm.groupby(["A", "B"])[["Y1", "Y2"]].transform("mean").values; E = (Y - cellm).T @ (Y - cellm); gmn = Y.mean(0); mAi = dm.groupby("A")[["Y1", "Y2"]].mean().values; mBj = dm.groupby("B")[["Y1", "Y2"]].mean().values
HA = rep * b * (mAi - gmn).T @ (mAi - gmn); HB = rep * a * (mBj - gmn).T @ (mBj - gmn)
wA = np.linalg.det(E) / np.linalg.det(E + HA); wB = np.linalg.det(E) / np.linalg.det(E + HB)
check("Multi-factor MANOVA Wilks (A, B) == hand-computed balanced-design values", abs(tm[(tm.Effect == "A") & tm.Test.str.startswith("Wilks")].Value.iloc[0] - wA) < 1e-9 and abs(tm[(tm.Effect == "B") & tm.Test.str.startswith("Wilks")].Value.iloc[0] - wB) < 1e-9)
r1 = S2.manova(dm, {"responses": ["Y1", "Y2"], "factor": "B"}, L, O("m1")); r2 = S2.manova(dm, {"responses": ["Y1", "Y2"], "factors": ["B"], "degree": 1}, L, O("m2"))
check("factors=[B] reproduces one-way MANOVA (Wilks, F)", abs(r1.tables[0][1].Value[0] - T(r2, "Multivariate")["Value"][0]) < 1e-9 and abs(r1.tables[0][1].F[0] - T(r2, "Multivariate")["F"][0]) < 1e-9)
dm2 = dm.copy(); dm2["x"] = rng.normal(size=len(dm2)); dm2["Y1"] += 2 * dm2.x; rc = S2.manova(dm2, {"responses": ["Y1", "Y2"], "factors": ["A"], "covariates": ["x"]}, L, O("mancova")); tc2 = T(rc, "Multivariate tests"); check("MANCOVA detects covariate effect", tc2[(tc2.Effect == "x") & tc2.Test.str.startswith("Wilks")]["Prob > F"].iloc[0] < 1e-6)

# ============================ GLMM ============================
Gn, nn_ = 40, 8; gg = np.repeat(range(Gn), nn_); uu = rng.normal(0, 1.2, Gn); xx = rng.normal(size=Gn * nn_); eta = -.4 + .9 * xx + uu[gg]; yb_ = (rng.random(Gn * nn_) < 1 / (1 + np.exp(-eta))).astype(float)
Xg = np.column_stack([np.ones(Gn * nn_), xx]); th = np.array([-.3, .8, math.log(1.1)]); nll15 = EX.glmm_nll(th, Xg, yb_, np.ones(len(yb_)), gg, Gn, "binomial", 25)
tot = 0.0
for k in range(Gn):
    mk = gg == k; fun = lambda u: np.exp(np.sum(yb_[mk] * (Xg[mk] @ th[:2] + u) - np.log1p(np.exp(Xg[mk] @ th[:2] + u)))) * stats.norm.pdf(u, 0, math.exp(th[2]))
    tot += math.log(integrate.quad(fun, -12, 12, epsabs=1e-13, epsrel=1e-12, limit=200)[0])
check("GLMM adaptive Gauss-Hermite marginal likelihood == numerical integration (quad)", abs(-nll15 - tot) < 1e-6, f"{-nll15:.8f} vs {tot:.8f}")
nll1 = EX.glmm_nll(th, Xg, yb_, np.ones(len(yb_)), gg, Gn, "binomial", 1); check("Laplace (nAGQ=1) within 0.5 log-lik units of AGQ for this design", abs(nll1 - nll15) < .5, f"diff={abs(nll1-nll15):.4f}")
Gn, nn_ = 100, 10; gg = np.repeat(range(Gn), nn_); uu = rng.normal(0, 1.0, Gn); xx = rng.normal(size=Gn * nn_); yb_ = (rng.random(Gn * nn_) < 1 / (1 + np.exp(-(-.5 + .8 * xx + uu[gg])))).astype(int)
dg = pd.DataFrame({"y": yb_, "x": xx, "g": gg.astype(str)}); rg = EX.glmm(dg, {"response": "y", "fixed": ["x"], "group": "g"}, L, O("glmm")); fx = T(rg, "Fixed effects"); ri = T(rg, "Random intercept")
check("GLMM binomial recovers beta=(-.5,.8), sigma_u=1 (3 SE)", abs(fx.Estimate[0] + .5) < 3 * fx["Std Error"][0] and abs(fx.Estimate[1] - .8) < 3 * fx["Std Error"][1] and abs(ri["Std Dev (sigma_u)"][0] - 1) < 3 * ri["Std Error (delta)"][0], f"{fx.Estimate.round(3).tolist()} sigma={ri['Std Dev (sigma_u)'][0]:.3f}")
xx0 = rng.normal(size=600); y0 = (rng.random(600) < 1 / (1 + np.exp(-(.3 + .9 * xx0)))).astype(int); d0 = pd.DataFrame({"y": y0, "x": xx0, "g": np.repeat(range(60), 10).astype(str)})
r0 = EX.glmm(d0, {"response": "y", "fixed": ["x"], "group": "g"}, L, O("glmm0")); from sklearn.linear_model import LogisticRegression
sk = LogisticRegression(C=1e8, max_iter=2000).fit(xx0[:, None], y0)
check("GLMM with no group effect (sigma->0) == ordinary logistic regression (sklearn)", abs(T(r0, "Fixed effects").Estimate[1] - sk.coef_[0, 0]) < .03 and T(r0, "Random intercept")["Std Dev (sigma_u)"][0] < .25, f"{T(r0,'Fixed effects').Estimate[1]:.4f} vs {sk.coef_[0,0]:.4f}; sigma={T(r0,'Random intercept')['Std Dev (sigma_u)'][0]:.3f}")
Gp, npp = 50, 12; gp = np.repeat(range(Gp), npp); up = rng.normal(0, .6, Gp); xp = rng.normal(size=Gp * npp); yp = rng.poisson(np.exp(.5 + .4 * xp + up[gp])); dp = pd.DataFrame({"y": yp, "x": xp, "g": gp.astype(str)})
rp = EX.glmm(dp, {"response": "y", "family": "poisson", "fixed": ["x"], "group": "g"}, L, O("glmmp")); fp = T(rp, "Fixed effects"); check("GLMM Poisson recovers beta=(.5,.4), sigma_u=.6 (3 SE)", abs(fp.Estimate[0] - .5) < 3 * fp["Std Error"][0] and abs(fp.Estimate[1] - .4) < 3 * fp["Std Error"][1] and abs(T(rp, "Random intercept")["Std Dev (sigma_u)"][0] - .6) < .2)

# ============================ Gage R&R REML ============================
P_, O_, R_ = 10, 3, 3; pe = rng.normal(0, 2.0, P_); oe = rng.normal(0, .5, O_); ie = rng.normal(0, .3, (P_, O_)); rows = []
for i in range(P_):
    for j in range(O_):
        for _ in range(R_): rows.append({"part": f"P{i}", "op": f"O{j}", "m": 50 + pe[i] + oe[j] + ie[i, j] + rng.normal(0, .5)})
dgr = pd.DataFrame(rows); ra_ = C1.gage_rr(dgr, {"part": "part", "operator": "op", "measurement": "m", "alpha_int": 1.0}, L, O("grr")); rr_ = EX.gage_rr_reml(dgr, {"part": "part", "operator": "op", "measurement": "m"}, L, O("grrreml"))
check("REML Gage R&R %StudyVar and ndc == ANOVA Gage R&R (balanced, interaction kept)", abs(ra_.summary["pct_study_var"] - rr_.summary["pct_study_var"]) < .05 and ra_.summary["ndc"] == rr_.summary["ndc"], f"{ra_.summary['pct_study_var']:.3f} vs {rr_.summary['pct_study_var']:.3f}")
du = dgr.sample(frac=.8, random_state=3); ru = EX.gage_rr_reml(du, {"part": "part", "operator": "op", "measurement": "m"}, L, O("grru")); check("REML Gage R&R works on unbalanced data and stays near the balanced result (+-6 pts)", abs(ru.summary["pct_study_var"] - rr_.summary["pct_study_var"]) < 6, f"{ru.summary['pct_study_var']:.2f} vs {rr_.summary['pct_study_var']:.2f}")
dgr["temp"] = rng.normal(20, 3, len(dgr)); dgr["m"] = dgr["m"] + .8 * (dgr.temp - 20); rc_ = EX.gage_rr_reml(dgr, {"part": "part", "operator": "op", "measurement": "m", "covariates": ["temp"]}, L, O("grrcov")); rn_ = EX.gage_rr_reml(dgr, {"part": "part", "operator": "op", "measurement": "m"}, L, O("grrnocov"))
fcv = T(rc_, "covariates"); check("Gage R&R with covariate recovers temp effect .8 and lowers repeatability", abs(fcv.Estimate[1] - .8) < 3 * fcv["Std Error"][1] and rc_.summary["var"]["repeatability"] < rn_.summary["var"]["repeatability"], f"temp={fcv.Estimate[1]:.3f}; repeat var {rc_.summary['var']['repeatability']:.3f} vs {rn_.summary['var']['repeatability']:.3f}")

# ============================ split-plot design ============================
ds = D2.make_design2(pd.DataFrame(), {"type": "split_plot", "whole_plot_factors": {"Temp": [150, 200], "Press": [10, 20]}, "subplot_factors": {"Add": [0, 1], "Time": [5, 10]}, "whole_plot_reps": 2, "seed": 4}, L, O("spd"))
dd_ = T(ds, "Design matrix"); check("Split-plot design: 8 whole plots x 4 sub-plot runs = 32", len(dd_) == 32 and dd_.WholePlot.nunique() == 8)
check("Split-plot design: whole-plot factors constant within a whole plot; sub-plot factors full 2^2 in each", all(dd_[dd_.WholePlot == w][["Temp", "Press"]].drop_duplicates().shape[0] == 1 and dd_[dd_.WholePlot == w][["Add", "Time"]].drop_duplicates().shape[0] == 4 for w in dd_.WholePlot.unique()))
check("Split-plot design: each whole-plot combination replicated twice", dd_.groupby("WholePlot")[["Temp", "Press"]].first().value_counts().eq(2).all())

# ============================ reference coding ============================
dr = pd.DataFrame({"g": np.repeat(list("abc"), 20), "x": rng.normal(size=60)}); dr["y"] = 1 + dr.x + (dr.g == "a") * 2 + (dr.g == "b") * .5 + rng.normal(size=60) * .3
re = ss_linear.glm_anova(dr, {"response": "y", "factors": ["g"], "covariates": ["x"], "degree": 1}, L, O("eff")); rf = ss_linear.glm_anova(dr, {"response": "y", "factors": ["g"], "covariates": ["x"], "degree": 1, "coding": "reference"}, L, O("ref"))
Xd = np.column_stack([np.ones(60), (dr.g == "a"), (dr.g == "b"), dr.x]).astype(float); bd = np.linalg.lstsq(Xd, dr.y.values, rcond=None)[0]
check("Reference coding (last level = reference) parameter estimates == dummy-variable least squares (SPSS)", np.allclose(T(rf, "Parameter").Estimate.values, bd, atol=1e-8))
check("Type III table identical under both codings (tests never change)", np.allclose(T(re, "Effect Tests").iloc[:, 2:].values.astype(float), T(rf, "Effect Tests").iloc[:, 2:].values.astype(float), equal_nan=True))
lg = pd.DataFrame({"g": np.repeat(list("abc"), 100), "x": rng.normal(size=300)}); lg["y"] = (rng.random(300) < 1 / (1 + np.exp(-(.2 + lg.x + (lg.g == "a") * 1.0)))).astype(int)
g_e = ss_linear.glm(lg, {"response": "y", "predictors": ["x"], "factors": ["g"]}, L, O("glme")); g_r = ss_linear.glm(lg, {"response": "y", "predictors": ["x"], "factors": ["g"], "coding": "reference"}, L, O("glmr"))
check("Logistic: both codings give identical log-likelihood; reference coefficients match dummy-coded sklearn", abs(T(g_e, "Model fit")["Log-Likelihood"][0] - T(g_r, "Model fit")["Log-Likelihood"][0]) < 1e-6 and abs(T(g_r, "Parameter").Estimate.values[3] - LogisticRegression(C=1e8, max_iter=5000).fit(np.column_stack([(lg.g == "a"), (lg.g == "b"), lg.x]).astype(float), lg.y).coef_[0, 2]) < .02)

# ============================ CLI smoke (EN + HE) ============================
R = os.path.join(S, "run.py"); P = lambda n, d_: (d_.to_csv(os.path.join(OUT, n + ".csv"), index=False, encoding="utf-8-sig"), os.path.join(OUT, n + ".csv"))[1]
cases = [("stationarity", P("ts", pd.DataFrame({"v": np.cumsum(rng.normal(size=200))})), {"column": "v", "mc_reps": 300}), ("timeseries", P("ts2", pd.DataFrame({"v": yy[60:260]})), {"column": "v", "mode": "arima", "p": 0, "d": 1, "q": 1, "P": 0, "D": 1, "Q": 1, "period": 12, "horizon": 6}),
         ("glmm", P("glmm", dg), {"response": "y", "fixed": ["x"], "group": "g"}), ("gage_rr_reml", P("grr", du), {"part": "part", "operator": "op", "measurement": "m"}), ("split_plot", P("sp", sp), {"response": "y", "whole_plot": ["A"], "subplot": ["B"], "plot": "WP"}),
         ("manova", P("mv", dm), {"responses": ["Y1", "Y2"], "factors": ["A", "B"]}), ("mixed", P("mx", dfb), {"response": "y", "fixed": ["trt"], "random": ["blk"]}), ("cox_ph", P("cx", dd), {"time": "t", "event": "e", "covariates": ["x"]}), ("anova", P("an", dr), {"response": "y", "factors": ["g"], "coding": "reference"}),
         ("doe_design2", None, {"type": "split_plot", "whole_plot_factors": {"A": [0, 1]}, "subplot_factors": {"B": [0, 1], "C": [0, 1]}, "whole_plot_reps": 3})]
for an, dp_, prm in cases:
    for lang in ("en", "he"):
        od = O(f"cli_{an}_{lang}"); cmd = [sys.executable, "-I", R, an, "--params", json.dumps(prm), "--out", od, "--lang", lang] + (["--data", dp_] if dp_ else [])
        pr = subprocess.run(cmd, capture_output=True, text=True, timeout=600); good = pr.returncode == 0 and os.path.exists(json.loads(pr.stdout.strip().splitlines()[-1])["docx"]) if pr.returncode == 0 else False
        check(f"CLI {an} [{lang}]", good, (pr.stderr[-300:] if not good else ""))
print(f"\n{len(ok)} passed, {len(bad)} failed"); print("OUT:", OUT)
if bad: print("FAILED:", bad); sys.exit(1)
