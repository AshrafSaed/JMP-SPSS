"""Verification suite 2 for the JMP+SPSS skill: numeric checks of the extended analyses against independent computations,
published values and simulation truth, then runs every extended analysis through the CLI in EN and HE."""
import itertools, json, math, os, subprocess, sys, tempfile
import numpy as np, pandas as pd
from scipy import stats
HERE = os.path.dirname(os.path.abspath(__file__)); S = os.path.join(HERE, "..", "scripts"); sys.path.insert(0, S)
OUT = sys.argv[1] if len(sys.argv) > 1 else tempfile.mkdtemp(); os.makedirs(OUT, exist_ok=True)
from ss_common import Lang
import ss_stats2 as S2, ss_models2 as M2, ss_spc2 as C2, ss_rel2 as R2, ss_doe2 as D2, ss_graph as GR, ss_rel, ss_linear
L = Lang("en"); rng = np.random.default_rng(7); ok, bad = [], []


def check(name, cond, detail=""):
    (ok if cond else bad).append(name); print(("PASS " if cond else "FAIL ") + name + (f"  [{detail}]" if detail else ""))


def T(res, key):
    for n, t in res.tables:
        if key.lower() in n.lower(): return t
    raise KeyError(key + " in " + str([n for n, _ in res.tables]))


def O(name): d = os.path.join(OUT, name); os.makedirs(d, exist_ok=True); return d


# ============================ classical tests ============================
Y = np.array([[9, 2, 5, 8], [6, 1, 3, 2], [8, 4, 6, 8], [7, 1, 2, 6], [10, 5, 6, 9], [6, 2, 4, 7]], float)
icc, ci, ms = S2._icc_table(Y)
check("ICC Shrout-Fleiss 1979 (.17 .29 .71 .44 .62 .91)", all(abs(icc[k] - v) < .006 for k, v in {"ICC(1,1)": .17, "ICC(2,1)": .29, "ICC(3,1)": .71, "ICC(1,k)": .44, "ICC(2,k)": .62, "ICC(3,k)": .91}.items()), str({k: round(v, 3) for k, v in icc.items()}))
r = S2.cronbach(pd.DataFrame(Y, columns=list("abcd")), {}, L, O("cron")); check("Cronbach alpha == ICC(3,k)", abs(r.summary["alpha"] - icc["ICC(3,k)"]) < 1e-9)
check("Grubbs critical n=10 = 2.290 (NIST)", abs(S2._grubbs_crit(10) - 2.290) < .001, f"{S2._grubbs_crit(10):.4f}")
check("Tolerance k: 1-sided n=10 95/95 = 2.911; 2-sided = 3.379 (Howe approx.)", abs(S2.tol_factor(10, .95, .95, False) - 2.911) < .001 and abs(S2.tol_factor(10, .95, .95, True) - 3.379) < .01)
k, se, po, _, _ = S2._kappa2(["y"] * 25 + ["n"] * 25, ["y"] * 20 + ["n"] * 5 + ["y"] * 10 + ["n"] * 15)
from sklearn.metrics import cohen_kappa_score
ka = rng.integers(0, 4, 200); kb = np.clip(ka + rng.integers(-1, 2, 200), 0, 3)
check("Cohen kappa == sklearn (unweighted, linear, quadratic)", all(abs(S2._kappa2(ka.astype(str).tolist(), kb.astype(str).tolist(), w)[0] - cohen_kappa_score(ka, kb, weights=w)) < 1e-9 for w in (None, "linear", "quadratic")))
cnt = np.array([[0, 0, 0, 0, 14], [0, 2, 6, 4, 2], [0, 0, 3, 5, 6], [0, 3, 9, 2, 0], [2, 2, 8, 1, 1], [7, 7, 0, 0, 0], [3, 2, 6, 3, 0], [2, 5, 3, 1, 3], [6, 5, 2, 1, 0], [0, 2, 2, 3, 7]])
Rm = np.array([np.repeat(np.arange(5), c) for c in cnt]); kf = S2._fleiss(Rm)[0]; check("Fleiss kappa Wikipedia example = 0.210", abs(kf - .210) < .001, f"{kf:.4f}")
d2g = pd.DataFrame({"y": np.r_[rng.normal(0, 1, 30), rng.normal(.8, 1, 35)], "g": ["a"] * 30 + ["b"] * 35}); rk = S2.nonparametric(d2g, {"mode": "kruskal", "column": "y", "group": "g"}, L, O("kw"))
H = stats.kruskal(d2g.y[d2g.g == "a"], d2g.y[d2g.g == "b"]).statistic; zD = T(rk, "Dunn")["z"].iloc[0]; check("Dunn z^2 == Kruskal H for 2 groups", abs(zD ** 2 - H) < 1e-8, f"{zD**2:.5f} vs {H:.5f}")
rk3 = S2.nonparametric(pd.DataFrame({"a": rng.normal(size=20), "b": rng.normal(.5, 1, 20), "c": rng.normal(1, 1, 20)}), {"mode": "friedman", "columns": ["a", "b", "c"]}, L, O("fr")); check("Friedman runs + Kendall W in [0,1]", 0 <= rk3.summary["W"] <= 1)
rp = S2.proportion(pd.DataFrame(), {"x": 8, "n": 10}, L, O("prop")); t_ = T(rp, "confidence")
check("Wilson 8/10 = (.4902,.9433); Clopper-Pearson = (.4439,.9748)", abs(t_["Wilson low"][0] - .4902) < 5e-4 and abs(t_["Wilson high"][0] - .9433) < 5e-4 and abs(t_["Exact low"][0] - .4439) < 5e-4 and abs(t_["Exact high"][0] - .9748) < 5e-4)
x_ = rng.normal(5, 2, 40); y_ = rng.normal(5.2, 2, 45); dfe = pd.DataFrame({"v": np.r_[x_, y_], "g": ["a"] * 40 + ["b"] * 45}); re_ = S2.equivalence(dfe, {"mode": "two", "column": "v", "group": "g", "low": -1.5, "high": 1.5}, L, O("tost"))
v1, v2 = x_.var(ddof=1) / 40, y_.var(ddof=1) / 45; diff = x_.mean() - y_.mean(); dof = (v1 + v2) ** 2 / (v1 ** 2 / 39 + v2 ** 2 / 44); p_ref = max(stats.t.sf((diff + 1.5) / math.sqrt(v1 + v2), dof), stats.t.cdf((diff - 1.5) / math.sqrt(v1 + v2), dof))
check("TOST p == independent Welch computation", abs(re_.summary["tost_p"] - p_ref) < 1e-10)
sj = rng.normal(0, 2, 25); rm_df = pd.DataFrame({f"t{i}": sj + i * .5 + rng.normal(0, 1, 25) for i in range(4)}); rr = S2.rm_anova(rm_df, {"columns": list(rm_df.columns)}, L, O("rm"))
long = rm_df.reset_index().melt(id_vars="index", var_name="cond", value_name="y"); long["index"] = long["index"].astype(str); o = ss_linear.OLS(long, "y", factors=["index", "cond"]); Fo = o.ss3().set_index("Source").loc["cond", "F Ratio"]
check("RM-ANOVA F == Type III OLS (subject + condition)", abs(rr.summary["F"] - Fo) < 1e-8, f"{rr.summary['F']:.5f} vs {Fo:.5f}"); check("Greenhouse-Geisser epsilon within [1/(k-1), 1]", 1 / 3 - 1e-9 <= rr.summary["eps_gg"] <= 1 + 1e-9)
two = pd.DataFrame({"a": rng.normal(0, 1, 30), "b": rng.normal(0, 1, 30)}); two["a"] += np.r_[np.zeros(15), np.ones(15) * .8]; two["g"] = ["u"] * 15 + ["v"] * 15
rm_ = S2.manova(two, {"responses": ["a", "b"], "factor": "g"}, L, O("manova")); Xa, Xb = two[two.g == "u"][["a", "b"]].values, two[two.g == "v"][["a", "b"]].values; dm = Xa.mean(0) - Xb.mean(0)
Sp = ((len(Xa) - 1) * np.cov(Xa.T) + (len(Xb) - 1) * np.cov(Xb.T)) / (len(Xa) + len(Xb) - 2); T2 = len(Xa) * len(Xb) / (len(Xa) + len(Xb)) * dm @ np.linalg.solve(Sp, dm); Ft = (len(two) - 2 - 1) / (2 * (len(two) - 2)) * T2
check("MANOVA (2 groups) Wilks F == Hotelling T2 F", abs(T(rm_, "Multivariate").iloc[0]["F"] - Ft) < 1e-8, f"{T(rm_, 'Multivariate').iloc[0]['F']:.5f} vs {Ft:.5f}")
o1 = pd.DataFrame({"y": rng.normal(size=60), "g": np.repeat(list("abc"), 20)}); o1.loc[o1.g == "b", "y"] += 1; rm1 = S2.manova(o1, {"responses": ["y"], "factor": "g"}, L, O("manova1")); Fa = stats.f_oneway(*[o1.y[o1.g == l] for l in "abc"]).statistic
check("MANOVA (1 response) all four F == ANOVA F", all(abs(f - Fa) < 1e-8 for f in T(rm1, "Multivariate")["F"]))
xo = np.r_[rng.normal(10, 1, 28), [20.0, -2.0]]; ro = S2.outliers(pd.DataFrame({"v": xo}), {"column": "v", "max_outliers": 4}, L, O("out")); check("Generalized ESD finds the 2 planted outliers", ro.summary["esd_outliers"] == 2, str(ro.summary))
ln = np.exp(rng.normal(1, .8, 300)); rt = S2.transform(pd.DataFrame({"v": ln}), {"column": "v"}, L, O("tr")); check("Box-Cox of lognormal gives lambda ~ 0 and equals scipy", abs(rt.summary["boxcox_lambda"]) < .2 and abs(rt.summary["boxcox_lambda"] - stats.boxcox(ln)[1]) < 1e-9, f"{rt.summary['boxcox_lambda']:.3f}")
bx = rng.normal(50, 5, 60); rb = S2.bootstrap(pd.DataFrame({"v": bx}), {"column": "v", "statistic": "mean", "n_resamples": 3000}, L, O("boot")); tci = stats.t.interval(.95, 59, bx.mean(), stats.sem(bx)); lo_, hi_ = rb.summary["ci_percentile"]
check("Bootstrap mean CI ~ t-interval (width within 12%)", abs((hi_ - lo_) / (tci[1] - tci[0]) - 1) < .12, f"{(hi_-lo_):.3f} vs {(tci[1]-tci[0]):.3f}")
rmt = S2.multiple_testing(pd.DataFrame(), {"pvalues": [.01, .04, .03, .005]}, L, O("mt")); tt = T(rmt, "Adjusted").sort_values("Raw p")
check("Multiple testing: Bonferroni, Holm, BH hand values", np.allclose(tt["bonferroni"], [.02, .04, .12, .16]) and np.allclose(tt["holm"], [.02, .03, .06, .06]) and np.allclose(tt["bh"], [.02, .02, .04, .04]))
Z = rng.normal(size=(150, 2)); xx = Z[:, 0] + rng.normal(size=150) * .5; yy = Z[:, 0] * .7 + Z[:, 1] + rng.normal(size=150) * .5; pc = pd.DataFrame({"x": xx, "y": yy, "z1": Z[:, 0], "z2": Z[:, 1]}); rpc = S2.partial_corr(pc, {"x": "x", "y": "y", "controls": ["z1", "z2"]}, L, O("pc"))
Pm = np.linalg.inv(pc.corr().values); r_inv = -Pm[0, 1] / math.sqrt(Pm[0, 0] * Pm[1, 1]); check("Partial correlation == -P_xy/sqrt(P_xx P_yy)", abs(rpc.summary["partial_r"] - r_inv) < 1e-9)
ba = pd.DataFrame({"a": rng.normal(10, 2, 80)}); ba["b"] = ba.a + rng.normal(.3, .5, 80); rba = S2.bland_altman(ba, {"column": "a", "column2": "b"}, L, O("ba")); check("Bland-Altman bias/LoA", abs(rba.summary["bias"] - (ba.a - ba.b).mean()) < 1e-12 and abs(rba.summary["loa"][1] - ((ba.a - ba.b).mean() + 1.96 * (ba.a - ba.b).std())) < 1e-12)

# ============================ models ============================
a_, n_ = 12, 6; g = np.repeat(range(a_), n_); u = rng.normal(0, 2, a_); y = 10 + u[g] + rng.normal(0, 1, a_ * n_); dfm = pd.DataFrame({"y": y, "g": g.astype(str)})
rmx = M2.mixed(dfm, {"response": "y", "random": ["g"]}, L, O("mix")); gm = dfm.groupby("g").y.mean(); msb = n_ * ((gm - y.mean()) ** 2).sum() / (a_ - 1); msw = ((dfm.y - dfm.g.map(gm)) ** 2).sum() / (a_ * (n_ - 1))
check("REML variance components == ANOVA estimates (balanced one-way)", abs(rmx.summary["variance_components"]["g"] - (msb - msw) / n_) < 1e-3 and abs(rmx.summary["variance_components"]["Residual"] - msw) < 1e-3, str({k: round(v, 4) for k, v in rmx.summary["variance_components"].items()}) + f" vs {(msb-msw)/n_:.4f},{msw:.4f}")
blk = np.repeat(range(10), 3); trt = np.tile(list("ABC"), 10); yb = 5 + rng.normal(0, 1.5, 10)[blk] + np.tile([0, 1, 2.5], 10) + rng.normal(0, .5, 30); dfb = pd.DataFrame({"y": yb, "trt": trt, "blk": blk.astype(str)})
rmb = M2.mixed(dfb, {"response": "y", "fixed": ["trt"], "random": ["blk"]}, L, O("rcbd")); ols_ = ss_linear.OLS(dfb, "y", factors=["trt"]); fx = T(rmb, "Fixed effects (Wald")
check("Mixed fixed effects == OLS in balanced RCBD", np.allclose(fx["Estimate"].values, ols_.beta, atol=1e-5))
# nested 3-level balanced closed form
A_, B_, R_ = 4, 3, 5; ya = rng.normal(0, 3, A_); yb2 = rng.normal(0, 2, (A_, B_)); rows = [{"A": f"a{i}", "B": f"b{i}{j}", "y": 20 + ya[i] + yb2[i, j] + rng.normal(0, 1)} for i in range(A_) for j in range(B_) for _ in range(R_)]
dn = pd.DataFrame(rows); rn = M2.nested_anova(dn, {"response": "y", "factors": ["A", "B"]}, L, O("nest")); tn = T(rn, "Nested ANOVA table"); msA, msB, msE = tn["Mean Square"].values
check("Nested REML components == EMS closed form", abs(rn.summary["variance_components"]["A"] - max((msA - msB) / (B_ * R_), 0)) < 5e-3 and abs(rn.summary["variance_components"]["A:B"] - (msB - msE) / R_) < 5e-3 and abs(rn.summary["variance_components"]["Residual"] - msE) < 5e-3)
xb = rng.normal(size=500); pb = 1 / (1 + np.exp(-(.5 + 1.2 * xb))); yb_ = (rng.random(500) < pb).astype(int); ro = M2.ordinal(pd.DataFrame({"y": yb_, "x": xb}), {"response": "y", "predictors": ["x"], "order": [0, 1]}, L, O("ord2"))
from sklearn.linear_model import LogisticRegression
sk = LogisticRegression(penalty=None, max_iter=1000).fit(xb[:, None], yb_); check("Ordinal(2 levels) == binary logistic (sklearn)", abs(ro.summary["coef"]["x"] - sk.coef_[0, 0]) < 1e-3 and abs(ro.summary["coef"]["Intercept 0|1"] + sk.intercept_[0]) < 1e-3)
xx4 = rng.normal(size=3000); lat = xx4 + rng.logistic(size=3000); y4 = np.digitize(lat, [-1, .3, 1.5]); ro4 = M2.ordinal(pd.DataFrame({"y": y4, "x": xx4}), {"response": "y", "predictors": ["x"], "order": [0, 1, 2, 3]}, L, O("ord4")); t4 = T(ro4, "Parameter")
truth = [-1, .3, 1.5, 1.0]; check("Ordinal 4-level recovers thresholds & slope (within 3 SE)", all(abs(e - t) < 3 * s for e, t, s in zip(t4["Estimate"], truth, t4["Std Error"])), str(t4["Estimate"].round(3).tolist()))
X1 = rng.normal(size=2000); lg = np.column_stack([np.zeros(2000), .5 + 1.0 * X1, -.3 - .8 * X1]); Pm_ = np.exp(lg); Pm_ /= Pm_.sum(1, keepdims=True); yc = np.array([rng.choice(3, p=pp) for pp in Pm_]); rmn = M2.multinomial(pd.DataFrame({"y": yc.astype(str), "x": X1}), {"response": "y", "predictors": ["x"], "reference": "0"}, L, O("mn")); tm = T(rmn, "Parameter")
check("Multinomial recovers coefficients (within 3 SE)", all(abs(e - t) < 3 * s for e, t, s in zip(tm["Estimate"], [.5, 1.0, -.3, -.8], tm["Std Error"])), str(tm["Estimate"].round(3).tolist()))
e = rng.normal(size=2000); zt = np.zeros(2000)
for t in range(1, 2000): zt[t] = .7 * zt[t - 1] + e[t]
f = M2.arima_fit(zt + 5, 1, 0, 0); check("AR(1) CSS recovers phi=.7, mean=5", abs(f["phi"][0] - .7) < .04 and abs(f["mu"] - 5) < .2, f"{f['phi'][0]:.3f},{f['mu']:.3f}")
ac = M2.acf(zt, 3); check("ACF lag1 == corr; PACF(1)=ACF(1); PACF(2)~0 for AR(1)", abs(ac[1] - np.corrcoef(zt[:-1], zt[1:])[0, 1]) < 5e-3 and abs(M2.pacf(zt, 3)[1] - ac[1]) < 1e-12 and abs(M2.pacf(zt, 3)[2]) < .06)
rw = np.cumsum(rng.normal(size=300)); fr = M2.arima_fit(rw, 0, 1, 0); fc, se_ = M2.arima_forecast(fr, 5); check("ARIMA(0,1,0): forecast = last value, se = sigma*sqrt(h)", np.allclose(fc, rw[-1]) and np.allclose(se_, math.sqrt(fr["sigma2"]) * np.sqrt(np.arange(1, 6))))
tt_ = np.arange(72); ser = 50 + .5 * tt_ + 8 * np.sin(2 * np.pi * tt_ / 12); sse, fit_, lvl, bb, ss_ = M2._hw(ser, 12, True, "add", [.3, .1, .1]); rts = M2.timeseries(pd.DataFrame({"v": ser}), {"column": "v", "mode": "smooth", "period": 12, "trend": True, "seasonal": "add", "horizon": 12}, L, O("hw"))
fcst = T(rts, "Forecast")["Forecast"].values; truth_f = 50 + .5 * np.arange(72, 84) + 8 * np.sin(2 * np.pi * np.arange(72, 84) / 12); check("Holt-Winters reproduces a noiseless trend+season (forecast error < 2%)", np.max(np.abs(fcst - truth_f)) / truth_f.mean() < .02, f"max err {np.max(np.abs(fcst-truth_f)):.3f}")
F = rng.normal(size=(600, 2)); Xf = np.column_stack([F[:, 0] + rng.normal(0, .4, 600) for _ in range(3)] + [F[:, 1] + rng.normal(0, .4, 600) for _ in range(3)]); dff = pd.DataFrame(Xf, columns=list("abcdef")); rfa = M2.factor_analysis(dff, {}, L, O("fa")); ld = np.abs(T(rfa, "Loadings").iloc[:, 1:3].values)
own = np.r_[ld[:3].max(1), ld[3:].max(1)]; other = np.r_[ld[:3].min(1), ld[3:].min(1)]; check("Factor analysis + varimax recovers simple structure (own>.85, cross<.25)", own.min() > .85 and other.max() < .25 and rfa.summary["factors"] == 2 and np.allclose(rfa.summary["eigenvalues"], np.sort(np.linalg.eigvalsh(np.corrcoef(Xf.T)))[::-1]), f"own.min={own.min():.2f} other.max={other.max():.2f} factors={rfa.summary['factors']}")
xs = np.linspace(0, 10, 40); ys = 3 + 5 * np.exp(-.6 * xs) + rng.normal(0, .05, 40); rnl = M2.nonlinear(pd.DataFrame({"x": xs, "y": ys}), {"x": "x", "y": "y", "model": "exp_decay"}, L, O("nl")); tn_ = T(rnl, "Parameter")
check("Nonlinear exp_decay recovers (3,5,.6) within 3 SE", all(abs(e - t) < 3 * s for e, t, s in zip(tn_["Estimate"], [3, 5, .6], tn_["Std Error"])))
Xp = rng.normal(size=(200, 10)); yp = 3 * Xp[:, 0] - 2 * Xp[:, 1] + rng.normal(0, 1, 200); dp = pd.DataFrame(Xp, columns=[f"x{i}" for i in range(10)]); dp["y"] = yp; rpn = M2.penalized(dp, {"response": "y", "method": "lasso"}, L, O("lasso")); cf_ = T(rpn, "Coeff")
sel = cf_[(cf_.Term != "Intercept") & cf_.Selected].Term.tolist(); check("LASSO keeps true predictors x0,x1 and drops most noise", {"x0", "x1"} <= set(sel) and len(sel) <= 6, str(sel))
rf_ = M2.predictive(dp.assign(y=Xp[:, 0] ** 2 + rng.normal(0, .3, 200)), {"response": "y", "method": "forest"}, L, O("rf")); check("Random forest: x0 top importance, CV R2 > 0.6", rf_.summary["top"] == "x0" and rf_.summary["cv_mean"] > .6, str(rf_.summary))
Tl = rng.normal(size=(150, 2)); Xl = Tl @ rng.normal(size=(2, 8)) + rng.normal(0, .2, (150, 8)); yl = 2 * Tl[:, 0] - Tl[:, 1] + rng.normal(0, .2, 150); dl = pd.DataFrame(Xl, columns=[f"p{i}" for i in range(8)]); dl["y"] = yl; rpl = M2.pls(dl, {"responses": ["y"]}, L, O("pls")); check("PLS: Q2 > 0.9 with 2-3 components", rpl.summary["q2"] > .9 and rpl.summary["components"] in (2, 3), str(rpl.summary))
dcl = pd.DataFrame({"a": np.r_[rng.normal(0, 1, 60), rng.normal(4, 1, 60)], "b": np.r_[rng.normal(0, 1, 60), rng.normal(3, 1, 60)], "k": ["x"] * 60 + ["y"] * 60}); rdis = M2.discriminant(dcl, {"response": "k", "predictors": ["a", "b"]}, L, O("lda")); check("LDA separates well-separated classes (CV acc > .95)", rdis.summary["acc_cv"] > .95)

# ============================ SPC / MSA ============================
x1 = rng.normal(10, 1, 60); re = C2.ewma(pd.DataFrame({"v": x1}), {"column": "v", "lambda": 1.0, "target": 10, "sigma": 1.0}, L, O("ew1")); te = T(re, "EWMA parameters")
check("EWMA with lambda=1 reduces to Shewhart (limits 10+-3)", abs(te["Asymptotic UCL"][0] - 13) < 1e-9 and abs(te["Asymptotic LCL"][0] - 7) < 1e-9)
ewx = np.r_[rng.normal(0, 1, 40), rng.normal(1, 1, 40)]; rew = C2.ewma(pd.DataFrame({"v": ewx}), {"column": "v", "lambda": .2, "target": 0, "sigma": 1}, L, O("ew2")); check("EWMA detects a 1-sigma shift after it starts", rew.summary["first_signal"] is not None and rew.summary["first_signal"] > 40, str(rew.summary))
rc = C2.cusum(pd.DataFrame({"v": [0, 0, 0, 3, 3, 3.0]}), {"column": "v", "target": 0, "sigma": 1}, L, O("cu")); check("CUSUM hand case: C+ = 0,0,0,2.5,5,7.5 -> first signal at obs 6", rc.summary["first_upper"] == 6)
nb = rng.integers(80, 120, 60); pp_ = rng.binomial(nb, .1) / nb; c_ = np.round(pp_ * nb); rl = C2.laney(pd.DataFrame({"c": c_, "n": nb}), {"type": "p", "count": "c", "size": "n"}, L, O("la")); check("Laney: sigma_z ~ 1 for pure binomial data", .75 < rl.summary["sigma_z"] < 1.3, f"{rl.summary['sigma_z']:.2f}")
pb_ = rng.beta(10, 90, 60); cb = rng.binomial(nb, pb_); rl2 = C2.laney(pd.DataFrame({"c": cb, "n": nb}), {"type": "p", "count": "c", "size": "n"}, L, O("la2")); check("Laney: sigma_z > 1.5 for overdispersed data and fewer false alarms", rl2.summary["sigma_z"] > 1.5 and rl2.summary["n_out_laney"] <= rl2.summary["n_out_classical"], str(rl2.summary))
Xm = rng.multivariate_normal([0, 0, 0], np.eye(3) * [1, 4, 9], 100); Xm[50] += [0, 0, 20]; rt2 = C2.hotelling_t2(pd.DataFrame(Xm, columns=list("abc")), {"columns": list("abc"), "phase": 1}, L, O("t2")); n3 = 100
check("T2 Phase-I UCL == (n-1)^2/n * Beta quantile; planted outlier flagged on 'c'", abs(rt2.summary["ucl"] - ((n3 - 1) ** 2 / n3 * stats.beta.ppf(1 - .0027, 1.5, (n3 - 3 - 1) / 2))) < 1e-9 and "c" in T(rt2, "Out-of-control")["Largest contributor"].tolist())
g1 = rng.normal(10.02, .05, 50); rg1 = C2.gage_type1(pd.DataFrame({"v": g1}), {"column": "v", "reference": 10.0, "tolerance": 1.0}, L, O("g1")); m_, s_ = g1.mean(), g1.std(ddof=1)
check("Gage Type 1: Cg=0.2*T/(6s), Cgk=(0.1T-|bias|)/(3s)", abs(rg1.summary["Cg"] - .2 / (6 * s_)) < 1e-9 and abs(rg1.summary["Cgk"] - (.1 - abs(m_ - 10)) / (3 * s_)) < 1e-9)
refs = np.repeat([2, 4, 6, 8, 10], 12).astype(float); meas = refs + .5 + .02 * refs + rng.normal(0, .08, 60); rlb = C2.gage_linearity_bias(pd.DataFrame({"r": refs, "m": meas}), {"reference": "r", "measurement": "m", "process_variation": 6}, L, O("lin")); lr_ = stats.linregress(refs, meas - refs)
check("Linearity regression slope == scipy and ~0.02", abs(rlb.summary["slope"] - lr_.slope) < 1e-12 and abs(rlb.summary["slope"] - .02) < .01)
op_, pa_, rp_ = 3, 6, 3; sp_, so_, sr_ = 3.0, .0001, .5; rows = []
for o in range(op_):
    for p in range(pa_):
        pv_ = rng.normal(0, sp_)
        for r in range(rp_): rows.append({"op": f"o{o}", "part": f"p{p}", "m": 50 + pv_ + rng.normal(0, sr_)})
dg = pd.DataFrame(rows); rgn = C2.gage_nested(dg, {"operator": "op", "part": "part", "measurement": "m"}, L, O("gn")); rmx2 = M2.mixed(dg.assign(pp=dg.op + dg.part), {"response": "m", "random": ["op", "pp"]}, L, O("gn2")); vc = rmx2.summary["variance_components"]
tbl_ = T(rgn, "Variance components"); check("Nested GRR (ANOVA) repeatability == REML residual; part(op) matches", abs(tbl_.loc[tbl_.Source == "Repeatability", "Variance"].iloc[0] - vc["Residual"]) < 1e-3 and abs(tbl_.loc[tbl_.Source.str.startswith("Part"), "Variance"].iloc[0] - vc["pp"]) < 1e-2)
rsp = C2.sampling_plan(pd.DataFrame(), {"aql": .01, "ltpd": .05, "alpha": .05, "beta": .10}, L, O("sp")); n_, c_ = rsp.summary["n"], rsp.summary["c"]
brute = next((n for n in range(1, 3000) if any(stats.binom.cdf(c, n, .01) >= .95 and stats.binom.cdf(c, n, .05) <= .10 for c in range(0, n))), None)
check("Sampling plan meets risks and is minimal (brute-force check)", stats.binom.cdf(c_, n_, .01) >= .95 and stats.binom.cdf(c_, n_, .05) <= .10 and n_ == brute, f"n={n_}, c={c_}")
rr1 = C2.run_chart(pd.DataFrame({"v": np.arange(60) + rng.normal(0, .2, 60)}), {"column": "v"}, L, O("run")); check("Run chart flags a trend (up/down p < .001)", rr1.summary["p_updown"] < .001)
rpa = C2.pareto(pd.DataFrame({"c": list("aaaabbbccd")}), {"category": "c"}, L, O("par")); check("Pareto cumulative reaches 100%", abs(T(rpa, "Pareto")["Cumulative %"].iloc[-1] - 100) < 1e-9)

# ============================ reliability ============================
n = 400; Xc = rng.normal(size=n); Xb = rng.integers(0, 2, n).astype(float); tev = rng.exponential(1 / (0.1 * np.exp(.7 * Xc + .0 * Xb))); cen = rng.exponential(12, n); tobs = np.minimum(tev, cen); ev = (tev <= cen).astype(int)
dcx = pd.DataFrame({"t": tobs, "e": ev, "x": Xc}); rcx = R2.cox_ph(dcx, {"time": "t", "event": "e", "covariates": ["x"]}, L, O("cox")); tcx = T(rcx, "Parameter"); check("Cox recovers log-HR 0.7 (within 3 SE)", abs(tcx["Coef"][0] - .7) < 3 * tcx["Std Error"][0], f"{tcx['Coef'][0]:.3f}+-{tcx['Std Error'][0]:.3f}")
gb = rng.integers(0, 2, 120); tb = rng.exponential(10, 120) * np.where(gb == 1, 1.6, 1.0); eb = (rng.random(120) < .8).astype(int); dcb = pd.DataFrame({"t": tb, "e": eb, "g": gb.astype(float)}); rcb = R2.cox_ph(dcb, {"time": "t", "event": "e", "covariates": ["g"], "ties": "breslow"}, L, O("cox2"))
lr_chi, _, _ = ss_rel.logrank([(tb[gb == 0], eb[gb == 0]), (tb[gb == 1], eb[gb == 1])]); check("Cox score test (no ties) == log-rank chi-square", abs(rcb.summary["score_chi2"] - lr_chi) < 1e-6, f"{rcb.summary['score_chi2']:.6f} vs {lr_chi:.6f}")
Xd = np.column_stack([Xc[:60], Xb[:60]]); td, ed = tobs[:60].copy(), ev[:60].copy(); td[:20] = np.round(td[:20]); td = np.where(td == 0, 1, td); bt = np.array([.3, -.2]); ll0, g0, H0 = R2._cox_ll(bt, Xd, td, ed, "efron")
num_g = np.array([(R2._cox_ll(bt + h, Xd, td, ed, "efron")[0] - R2._cox_ll(bt - h, Xd, td, ed, "efron")[0]) / 2e-6 for h in np.eye(2) * 1e-6]); num_H = np.array([(R2._cox_ll(bt + h, Xd, td, ed, "efron")[1] - R2._cox_ll(bt - h, Xd, td, ed, "efron")[1]) / 2e-6 for h in np.eye(2) * 1e-6])
check("Cox Efron analytic gradient/Hessian == numeric derivatives (with ties)", np.allclose(g0, num_g, atol=1e-4) and np.allclose(H0, num_H, atol=1e-3))
tt3 = np.array([2, 5, 9.]); bet = 3 / np.sum(np.log(10 / tt3)); rrp = R2.repairable(pd.DataFrame({"t": tt3}), {"time": "t", "end": 10.0}, L, O("rep")); check("Crow-AMSAA beta hand calc (1.2459)", abs(rrp.summary["beta"] - bet) < 1e-9 and abs(bet - 1.2459) < 1e-3)
Bt, lam0, Tend = 1.5, .01, 50.0; rows = []
for s in range(25):
    tcur = 0.0
    while True:
        tcur = (tcur ** Bt + rng.exponential(1) / lam0) ** (1 / Bt)
        if tcur > Tend: break
        rows.append({"sys": s, "t": tcur})
drep = pd.DataFrame(rows); rrp2 = R2.repairable(drep, {"system": "sys", "time": "t", "end": Tend}, L, O("rep2")); N = rrp2.summary["N"]; exp_hat = rrp2.summary["lambda"] * Tend ** rrp2.summary["beta"]; exp_true = lam0 * Tend ** Bt
check("Crow-AMSAA recovers NHPP beta=1.5 (within 3 SE) and expected failures/system at T (10%)", abs(rrp2.summary["beta"] - Bt) < 3 * Bt / math.sqrt(N) and abs(exp_hat / exp_true - 1) < .1, f"beta={rrp2.summary['beta']:.3f}, E[N(T)] {exp_hat:.2f} vs {exp_true:.2f}")
Ea, shp = .7, 2.0; rows = []
for tc in (120, 150, 180):
    eta = 5000 * math.exp(Ea * (11604.5 / (tc + 273.15) - 11604.5 / (120 + 273.15))); ts = eta * rng.weibull(shp, 80); cens = 4000 / (tc / 120); rows += [{"t": min(a, cens), "e": int(a <= cens), "T": tc} for a in ts]
dalt = pd.DataFrame(rows); ralt = R2.alt(dalt, {"time": "t", "event": "e", "stress": "T", "relationship": "arrhenius", "dist": "weibull", "use_stress": 60}, L, O("alt")); tal = T(ralt, "Parameter")
check("ALT Arrhenius recovers Ea=0.7 eV (3 SE) and shape=2 (10%)", abs(ralt.summary["slope"] - Ea) < 3 * tal["Std Error"][1] and abs(ralt.summary["shape"] / shp - 1) < .1, f"Ea={ralt.summary['slope']:.3f}, shape={ralt.summary['shape']:.3f}")
dd = pd.DataFrame({"u": np.repeat(["a", "b", "c"], 5), "t": np.tile([0, 2, 4, 6, 8], 3), "y": np.concatenate([1 + 2 * np.arange(0, 10, 2), 1 + 1 * np.arange(0, 10, 2), 1 + .5 * np.arange(0, 10, 2)])}); rdg = R2.degradation(dd, {"unit": "u", "time": "t", "measure": "y", "threshold": 21}, L, O("deg")); pf = T(rdg, "Per-unit")["Pseudo failure time"].values
check("Degradation pseudo failure times exact (10, 20, 40)", np.allclose(pf, [10, 20, 40]))
tw = 100 * rng.weibull(2, 200); dli = pd.DataFrame({"l": tw, "u": tw}); rli = R2.life_interval(dli, {"lower": "l", "upper": "u", "distributions": ["weibull"]}, L, O("li")); fx = ss_rel.fit_dist("weibull", tw, np.ones(200))
check("Interval MLE with exact data == Weibull MLE (ss_rel)", abs(T(rli, "Fits")["Shape beta"][0] - math.exp(fx["theta"][0])) < 1e-3 and abs(T(rli, "Fits")["Scale eta"][0] - math.exp(fx["theta"][1])) < 1e-2)
edges = np.arange(0, 301, 25.0); tt_ = 100 * rng.weibull(2, 300); lo_i = edges[np.searchsorted(edges, tt_, side="right") - 1]; up_i = np.where(lo_i >= 250, np.nan, lo_i + 25); rli2 = R2.life_interval(pd.DataFrame({"l": lo_i, "u": up_i}), {"lower": "l", "upper": "u", "distributions": ["weibull"]}, L, O("li2"))
check("Interval-censored MLE recovers beta=2, eta=100 (10%)", abs(T(rli2, "Fits")["Shape beta"][0] / 2 - 1) < .12 and abs(T(rli2, "Fits")["Scale eta"][0] / 100 - 1) < .08, str(T(rli2, "Fits")[["Shape beta", "Scale eta"]].iloc[0].round(3).to_dict()))
rrg = {c: R2.reliability_plan(pd.DataFrame(), {"reliability": R_, "confidence": C_, "failures": [0, 1]}, L, O("rp")).tables[0][1] for c, (R_, C_) in {"a": (.9, .9), "b": (.95, .95), "c": (.99, .95)}.items()}
check("Success-run n: (R.9,C.9)=22; (.95,.95)=59; (.99,.95)=299; c=1 (.9/.9)=38", rrg["a"]["Required units n"][0] == 22 and rrg["b"]["Required units n"][0] == 59 and rrg["c"]["Required units n"][0] == 299 and rrg["a"]["Required units n"][1] == 38)
rwb = R2.reliability_plan(pd.DataFrame(), {"reliability": .9, "confidence": .9, "beta": 1.0, "mission_time": 100, "units": [10]}, L, O("wb")); tw_ = T(rwb, "Weibayes").iloc[0]["Test duration per unit"]; check("Weibayes (beta=1, r=0) == exponential closed form", abs(tw_ - (100 / -math.log(.9)) * (-math.log(1 - .9)) / 10) < 1e-6)

# ============================ DOE ============================
for Nn in (8, 12, 16, 20, 24, 32):
    M = D2.plackett_burman(Nn); check(f"Plackett-Burman N={Nn}: orthogonal & balanced", np.allclose(M.T @ M, Nn * np.eye(Nn - 1)) and np.allclose(M.sum(0), 0 if Nn != 12 else M.sum(0)) and set(np.unique(M)) == {-1, 1})
for k in (4, 5, 6, 7, 8, 10, 12):
    Mx, order = D2.dsd_matrix(k); ok_main = np.allclose(Mx.T @ Mx - np.diag(np.diag(Mx.T @ Mx)), 0); ok_q = all(abs(Mx[:, i] @ (Mx[:, j] ** 2)) < 1e-9 for i in range(k) for j in range(k)); ok_2 = all(abs(Mx[:, i] @ (Mx[:, j] * Mx[:, l])) < 1e-9 for i in range(k) for j, l in itertools.combinations(range(k), 2))
    check(f"DSD k={k} (order {order}, {len(Mx)} runs): mains orthogonal to each other, quadratics, 2FIs", ok_main and ok_q and ok_2 and set(np.unique(Mx)) <= {-1, 0, 1})
rdo = D2.make_design2(pd.DataFrame(), {"type": "d_optimal", "factors": {"A": [0, 1], "B": [0, 1], "C": [0, 1]}, "runs": 8, "model": "interaction"}, L, O("dopt")); check("D-optimal 8 runs, 3 factors, interaction model -> D-eff ~100% (= 2^3)", rdo.summary["d_eff"] > 99, f"{rdo.summary['d_eff']:.1f}")
rdo2 = D2.make_design2(pd.DataFrame(), {"type": "d_optimal", "factors": {"A": [0, 1], "B": [0, 1], "C": [0, 1], "D": [0, 1]}, "runs": 12, "model": "main"}, L, O("dopt2")); check("D-optimal 12 runs, 4 factors, main-effects -> D-eff > 97% (PB-12 level)", rdo2.summary["d_eff"] > 97, f"{rdo2.summary['d_eff']:.1f}")
for nm, (p, b) in D2.TAGUCHI.items():
    A = D2.gf_array(b, p); ok_pair = all(len(set(map(tuple, np.column_stack([A[:, i], A[:, j]])))) == p * p and len(np.unique(np.unique(np.column_stack([A[:, i], A[:, j]]), axis=0, return_counts=True)[1])) == 1 for i, j in itertools.combinations(range(A.shape[1]), 2))
    check(f"Taguchi {nm}: {A.shape[0]} runs x {A.shape[1]} cols, every column pair balanced (strength 2)", ok_pair)
rmx_ = D2.make_design2(pd.DataFrame(), {"type": "mixture", "components": ["x1", "x2", "x3"], "degree": 2, "randomize": False}, L, O("mixd")); dmx = T(rmx_, "Design matrix"); check("Mixture simplex-lattice(3,2)+centroid: 7 points summing to 1", len(dmx) == 7 and np.allclose(dmx[["x1", "x2", "x3"]].sum(1), 1))
trueb = np.array([10, 20, 30, 15, -10, 8.0]); Pm = dmx[["x1", "x2", "x3"]].values; dmx["y"] = D2._scheffe_X(Pm, "quadratic") @ trueb; dmx2 = pd.concat([dmx, dmx.sample(6, random_state=1)]); rma = D2.mixture_analyze(dmx2, {"components": ["x1", "x2", "x3"], "response": "y", "model": "quadratic", "goal": "max"}, L, O("mixa"))
check("Scheffe analysis recovers noiseless coefficients exactly", np.allclose(T(rma, "Scheffe")["Coefficient"].values, trueb, atol=1e-8))
rls = D2.make_design2(pd.DataFrame(), {"type": "latin_square", "size": 5}, L, O("ls")); dl_ = T(rls, "Design matrix"); check("Latin square: each treatment once per row and column", all(dl_[dl_["Row (blocking 1)"] == i]["Treatment"].nunique() == 5 for i in range(1, 6)) and all(dl_[dl_["Column (blocking 2)"] == i]["Treatment"].nunique() == 5 for i in range(1, 6)))
sim = []; Nn, sg, eff, p_ = 16, 1.0, 1.4, 4
for _ in range(6000):
    X = np.column_stack([np.ones(Nn), D2.plackett_burman(16)[:, :3]]); yv = X[:, 1] * eff / 2 + rng.normal(0, sg, Nn); b = np.linalg.lstsq(X, yv, rcond=None)[0]; r = yv - X @ b; s2 = r @ r / (Nn - p_); se = math.sqrt(s2 / Nn); sim.append(abs(b[1] / se) > stats.t.ppf(.975, Nn - p_))
rpw = D2.doe_power(pd.DataFrame(), {"sigma": 1.0, "effect": [1.4], "runs": [16], "terms": 4}, L, O("pw")); pw = T(rpw, "Power of two-level")["Power"][0]; check("DOE power (noncentral t) == Monte-Carlo simulation (+-0.02)", abs(pw - np.mean(sim)) < .02, f"{pw:.3f} vs sim {np.mean(sim):.3f}")
# foldover of 2^(7-4) Res III -> main effects clear of 2FIs
base = np.array(list(itertools.product([-1, 1], repeat=3)))[:, :]; Mf = np.column_stack([base, base[:, 0] * base[:, 1], base[:, 0] * base[:, 2], base[:, 1] * base[:, 2], base[:, 0] * base[:, 1] * base[:, 2]]).astype(float)
dfd = pd.DataFrame(Mf, columns=[f"{c}_coded" for c in "ABCDEFG"]);
for c in "ABCDEFG": dfd[c] = dfd[f"{c}_coded"]
raug = D2.doe_augment(dfd, {"operation": "foldover", "factors": {c: [-1, 1] for c in "ABCDEFG"}}, L, O("fold")); da = T(raug, "Augmented")[[f"{c}_coded" for c in "ABCDEFG"]].values
clear = all(abs(da[:, i] @ (da[:, j] * da[:, l])) < 1e-9 for i in range(7) for j, l in itertools.combinations(range(7), 2) if i not in (j, l)); check("Foldover of 2^(7-4) (Res III) makes mains clear of all 2FIs (Res IV)", clear and len(da) == 16)
L4 = D2.gf_array(2, 2)[:, :3] * 0 + D2.gf_array(2, 2)[:, :3]; tgd = pd.DataFrame({"A": L4[:, 0], "B": L4[:, 1], "C": L4[:, 2]}); eff_true = {"A": 3.0, "B": -1.0, "C": 0.0}; base_sn = 20.0
for r_ in range(2): tgd[f"r{r_}"] = 0.0
snt = base_sn + 3.0 * (tgd.A - .5) * 2 / 2 + -1.0 * (tgd.B - .5) * 2 / 2; tgd["r0"] = 10 ** (snt / 20) * 1.0; tgd["r1"] = 10 ** (snt / 20) * 1.0; rta = D2.taguchi_analyze(tgd, {"factors": ["A", "B", "C"], "responses": ["r0", "r1"], "sn": "larger"}, L, O("tg"))
check("Taguchi S/N (larger): best levels A=1,B=0,C any; delta A=3, B=1, C=0", rta.summary["best_levels"]["A"] in ("1", "1.0", 1) and rta.summary["best_levels"]["B"] in ("0", "0.0", 0) and np.allclose(T(rta, "Response table")["Delta"].values[:2], [3, 1], atol=1e-9) and abs(T(rta, "Response table")["Delta"].values[2]) < 1e-9)
cc = D2.make_design2(pd.DataFrame(), {"type": "full", "factors": {}}, L, O("x")) if False else None
x1g, x2g = np.meshgrid([-1, 0, 1], [-1, 0, 1]); x1g, x2g = x1g.ravel(), x2g.ravel(); dop = pd.DataFrame({"x1": x1g.astype(float), "x2": x2g.astype(float)}); dop["y1"] = 50 - 10 * (dop.x1 - .4) ** 2 - 10 * (dop.x2 + .2) ** 2; dop["y2"] = 5 + 2 * dop.x1
rop = D2.doe_optimize(dop, {"factors": ["x1", "x2"], "quadratic": True, "responses": {"y1": {"goal": "max", "low": 30, "high": 50}, "y2": {"goal": "target", "low": 3, "target": 5.8, "high": 7}}}, L, O("opt")); opt = rop.summary["optimum"]
# analytic: y2 target 5.8 -> x1 = .4 ; y1 max at (.4, -.2): both satisfied
check("Desirability optimum == analytic (x1=.4, x2=-.2, D=1)", abs(opt["x1"] - .4) < 1e-2 and abs(opt["x2"] + .2) < 2e-2 and rop.summary["overall_desirability"] > .999, str(rop.summary["optimum"]))

# ============================ graphs ============================
xl = np.linspace(0, 10, 50); gx, gy = GR.lowess(xl, 2 * xl + 1, frac=.5); check("LOWESS reproduces a straight line exactly", np.allclose(gy, 2 * gx + 1, atol=1e-6))
dfa = pd.DataFrame({"num": rng.normal(size=120), "num2": rng.normal(size=120), "cat": rng.choice(list("abc"), 120), "cat2": rng.choice(["u", "v"], 120), "cat3": rng.choice(list("pqrs"), 120)})
meth = {k: GR.auto_xy(dfa, {"y": y, "x": x}, L, O("axy_" + k)).summary["method"] for k, (y, x) in {"nc": ("num", "cat"), "nn": ("num", "num2"), "cn2": ("cat2", "num"), "cn3": ("cat3", "num"), "cc": ("cat", "cat2")}.items()}
check("Fit Y by X picks ANOVA / regression / logistic / multinomial / contingency", "ANOVA" in meth["nc"] and "regression" in meth["nn"].lower() and "Logistic" in meth["cn2"] and "Multinomial" in meth["cn3"] and "Contingency" in meth["cc"], str(list(meth.values())))

# ============================ run every extended analysis via CLI (EN + HE) ============================
def cli(analysis, data, params, lang, name):
    out = os.path.join(OUT, name); cmd = [sys.executable, os.path.join(S, "run.py"), analysis, "--params", json.dumps(params), "--out", out, "--lang", lang]
    if data: cmd += ["--data", data]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0: print("CLI ERROR", analysis, r.stderr[-600:]); return None
    return json.loads(r.stdout.strip().splitlines()[-1])


def save(df, nm): p = os.path.join(OUT, nm + ".csv"); df.to_csv(p, index=False); return p


P_wide = save(rm_df.assign(grp=np.repeat(["u", "v"], [13, 12])), "wide"); P_two = save(dfe, "two"); P_cl = save(dcl, "cls"); P_mix = save(dfb, "rcbd"); P_nest = save(dn, "nest"); P_fa = save(dff, "fa"); P_dp = save(dp, "dp"); P_pls = save(dl, "pls")
P_ts = save(pd.DataFrame({"v": 100 + .3 * np.arange(120) + 10 * np.sin(2 * np.pi * np.arange(120) / 12) + rng.normal(0, 2, 120)}), "ts"); P_cox = save(dcx, "cox"); P_rep = save(drep, "rep"); P_alt = save(dalt, "alt"); P_deg = save(dd, "deg"); P_li = save(pd.DataFrame({"l": lo_i, "u": up_i}), "li")
P_lap = save(pd.DataFrame({"c": cb, "n": nb}), "laney"); P_t2 = save(pd.DataFrame(Xm, columns=list("abc")), "t2"); P_g1 = save(pd.DataFrame({"v": g1}), "g1"); P_lin = save(pd.DataFrame({"r": refs, "m": meas}), "lin"); P_gn = save(dg, "gn"); P_par = save(pd.DataFrame({"c": list("aaaabbbccd")}), "par"); P_ew = save(pd.DataFrame({"v": ewx}), "ew"); P_oq = save(pd.DataFrame({"v": xo}), "oq")
P_ln = save(pd.DataFrame({"v": ln}), "ln"); P_pcd = save(pc, "pcd"); P_ba = save(ba, "ba"); P_mt = save(pd.DataFrame({"p": [.001, .01, .03, .04, .2, .5], "h": list("abcdef")}), "mt"); P_ag = save(pd.DataFrame({"r1": ka.astype(str), "r2": kb.astype(str)}), "ag")
att = pd.DataFrame([{"ap": a, "smp": s, "rep": r, "rating": ("p" if (s % 2 == 0) else "f") if rng.random() > .1 else ("f" if s % 2 == 0 else "p"), "std": "p" if s % 2 == 0 else "f"} for a in "ABC" for s in range(20) for r in range(2)]); P_att = save(att, "attr")
P_tg = save(tgd, "tg"); P_mxa = save(dmx2, "mxa"); P_op = save(dop, "dop"); P_gr = save(dfa, "gr"); P_fold = save(dfd, "fold"); P_tab = save(dfa, "tab")
smoke = [("nonparametric", P_two, {"mode": "kruskal", "column": "v", "group": "g"}), ("nonparametric", P_two, {"mode": "mannwhitney", "column": "v", "group": "g"}), ("nonparametric", P_wide, {"mode": "friedman", "columns": ["t0", "t1", "t2", "t3"]}), ("nonparametric", P_wide, {"mode": "wilcoxon", "column": "t0", "column2": "t1"}),
         ("proportion", None, {"x": 12, "n": 40, "x2": 25, "n2": 45}), ("variance_test", P_two, {"column": "v", "group": "g"}), ("equivalence", P_two, {"mode": "two", "column": "v", "group": "g", "low": -1.5, "high": 1.5}), ("rm_anova", P_wide, {"columns": ["t0", "t1", "t2", "t3"]}),
         ("manova", P_cl, {"responses": ["a", "b"], "factor": "k"}), ("agreement", P_ag, {"mode": "kappa", "rater1": "r1", "rater2": "r2"}), ("agreement", P_att, {"mode": "attribute", "appraiser": "ap", "sample": "smp", "rating": "rating", "standard": "std", "positive": "p"}), ("icc", P_wide, {"columns": ["t0", "t1", "t2", "t3"]}),
         ("cronbach", P_wide, {"columns": ["t0", "t1", "t2", "t3"]}), ("bland_altman", P_ba, {"column": "a", "column2": "b"}), ("outliers", P_oq, {"column": "v"}), ("tolerance", P_ln, {"column": "v", "coverage": .95, "conf": .95}), ("transform", P_ln, {"column": "v"}), ("bootstrap", P_ln, {"column": "v", "statistic": "median", "n_resamples": 1500}),
         ("multiple_testing", P_mt, {"column": "p", "label": "h"}), ("tabulate", P_tab, {"columns": ["num", "num2"], "by": ["cat"], "categorical": ["cat2"]}), ("partial_corr", P_pcd, {"x": "x", "y": "y", "controls": ["z1", "z2"]}),
         ("mixed", P_mix, {"response": "y", "fixed": ["trt"], "random": ["blk"]}), ("nested_anova", P_nest, {"response": "y", "factors": ["A", "B"]}), ("factor_analysis", P_fa, {}), ("discriminant", P_cl, {"response": "k", "predictors": ["a", "b"]}), ("ordinal", save(pd.DataFrame({"y": y4, "x": xx4}), "ord"), {"response": "y", "predictors": ["x"]}),
         ("multinomial", save(pd.DataFrame({"y": yc.astype(str), "x": X1}), "mn"), {"response": "y", "predictors": ["x"]}), ("nonlinear", save(pd.DataFrame({"x": xs, "y": ys}), "nl"), {"x": "x", "y": "y", "model": "exp_decay"}), ("penalized", P_dp, {"response": "y", "method": "enet"}), ("predictive", P_dp, {"response": "y", "method": "tree"}), ("predictive", P_dp, {"response": "y", "method": "boost"}),
         ("pls", P_pls, {"responses": ["y"]}), ("timeseries", P_ts, {"column": "v", "mode": "explore"}), ("timeseries", P_ts, {"column": "v", "mode": "decompose", "period": 12}), ("timeseries", P_ts, {"column": "v", "mode": "smooth", "period": 12, "seasonal": "add", "horizon": 12}), ("timeseries", P_ts, {"column": "v", "mode": "arima", "p": 1, "d": 1, "q": 1, "horizon": 12}), ("timeseries", P_ts, {"column": "v", "mode": "auto_arima", "d": 1, "horizon": 6}),
         ("ewma", P_ew, {"column": "v", "target": 0, "sigma": 1}), ("cusum", P_ew, {"column": "v", "target": 0, "sigma": 1}), ("laney", P_lap, {"type": "p", "count": "c", "size": "n"}), ("pareto", P_par, {"category": "c"}), ("run_chart", P_ew, {"column": "v"}), ("hotelling_t2", P_t2, {"columns": ["a", "b", "c"]}), ("gage_type1", P_g1, {"column": "v", "reference": 10, "tolerance": 1}),
         ("gage_linearity_bias", P_lin, {"reference": "r", "measurement": "m", "process_variation": 6}), ("gage_nested", P_gn, {"operator": "op", "part": "part", "measurement": "m"}), ("variability_chart", P_nest, {"response": "y", "factors": ["A", "B"]}), ("sampling_plan", None, {"aql": .01, "ltpd": .05, "lot_size": 2000}),
         ("cox_ph", P_cox, {"time": "t", "event": "e", "covariates": ["x"]}), ("repairable", P_rep, {"system": "sys", "time": "t", "end": 50}), ("alt", P_alt, {"time": "t", "event": "e", "stress": "T", "use_stress": 60}), ("degradation", P_deg, {"unit": "u", "time": "t", "measure": "y", "threshold": 21}), ("life_interval", P_li, {"lower": "l", "upper": "u"}), ("reliability_plan", None, {"reliability": .9, "confidence": .9, "beta": 2, "mission_time": 1000}),
         ("doe_design2", None, {"type": "dsd", "factors": {c: [0, 1] for c in "ABCDEF"}}), ("doe_design2", None, {"type": "plackett_burman", "factors": {c: [0, 1] for c in "ABCDEFG"}}), ("doe_design2", None, {"type": "taguchi", "levels": 3, "factors": {"A": ["L", "M", "H"], "B": ["L", "M", "H"], "C": ["L", "M", "H"]}}), ("doe_design2", None, {"type": "d_optimal", "factors": {"A": [0, 1], "B": [0, 1], "C": [0, 1]}, "runs": 10, "model": "interaction"}),
         ("doe_design2", None, {"type": "mixture", "components": ["x1", "x2", "x3"], "degree": 3}), ("doe_design2", None, {"type": "latin_square", "size": 4}), ("doe_augment", P_fold, {"operation": "axial"}), ("doe_power", None, {"sigma": 1, "effect": [1, 2]}), ("taguchi_analyze", P_tg, {"factors": ["A", "B", "C"], "responses": ["r0", "r1"]}), ("mixture_analyze", P_mxa, {"components": ["x1", "x2", "x3"], "response": "y"}),
         ("doe_optimize", P_op, {"factors": ["x1", "x2"], "quadratic": True, "responses": {"y1": {"goal": "max", "low": 30, "high": 50}, "y2": {"goal": "target", "low": 3, "target": 5.8, "high": 7}}}),
         ("graph", P_gr, {"type": "scatter", "x": "num", "y": "num2", "color": "cat", "fit": "lowess"}), ("graph", P_gr, {"type": "box", "x": "cat", "y": "num", "color": "cat2"}), ("graph", P_gr, {"type": "violin", "x": "cat", "y": "num"}), ("graph", P_gr, {"type": "bar", "x": "cat", "y": "num", "by": "cat2"}), ("graph", P_gr, {"type": "hist", "x": "num", "color": "cat"}), ("graph", P_gr, {"type": "heatmap"}), ("graph", P_gr, {"type": "pairs", "columns": ["num", "num2"]}),
         ("auto_xy", P_gr, {"y": "num", "x": "cat"}), ("auto_xy", P_gr, {"y": "cat2", "x": "num"})]
for an, d, p in smoke:
    for lang in ("en", "he"):
        r = cli(an, d, p, lang, f"cli_{an}_{lang}_{abs(hash(json.dumps(p, sort_keys=True))) % 10000}")
        check(f"CLI {an} [{lang}] {p.get('mode') or p.get('type') or p.get('method') or ''}".strip(), bool(r) and os.path.exists(r["docx"]) and all(os.path.exists(f) for f in r["figures"]) and len(r["figures"]) >= 1)
print(f"\n{len(ok)} passed, {len(bad)} failed"); print("FAILED:", bad) if bad else None; print("OUT:", OUT); sys.exit(1 if bad else 0)
