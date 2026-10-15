"""Verification suite 1 for the JMP+SPSS skill (core modules): checks numbers against scipy/numpy/sklearn and known theory, then exercises every analysis (EN + HE)."""
import json, os, subprocess, sys, tempfile
import numpy as np, pandas as pd
from scipy import stats
HERE = os.path.dirname(os.path.abspath(__file__)); S = os.path.join(HERE, "..", "scripts")
sys.path.insert(0, S)
OUT = sys.argv[1] if len(sys.argv) > 1 else tempfile.mkdtemp()
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(42)
ok, bad = [], []


def check(name, cond, detail=""):
    (ok if cond else bad).append(name)
    print(("PASS " if cond else "FAIL ") + name + (f"  [{detail}]" if detail else ""))


def cli(analysis, data, params, lang="en", name=None):
    out = os.path.join(OUT, name or analysis + "_" + lang)
    cmd = [sys.executable, os.path.join(S, "run.py"), analysis, "--params", json.dumps(params), "--out", out, "--lang", lang]
    if data: cmd += ["--data", data]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print("CLI ERROR", analysis, r.stderr[-800:]); return None
    return json.loads(r.stdout.strip().splitlines()[-1])


# ---------------- datasets ----------------
g = np.repeat(["A", "B", "C", "D"], 15); mu = {"A": 10, "B": 12, "C": 10.5, "D": 15}
oneway_df = pd.DataFrame({"grp": g, "y": [rng.normal(mu[k], 1.5) for k in g]}); P = os.path.join(OUT, "oneway.csv"); oneway_df.to_csv(P, index=False)
two_df = pd.DataFrame({"y": np.r_[rng.normal(50, 5, 25), rng.normal(54, 6, 30)], "g": ["x"] * 25 + ["y"] * 30}); P2 = os.path.join(OUT, "two.csv"); two_df.to_csv(P2, index=False)
n = 120; X1 = rng.normal(0, 1, n); X2 = rng.normal(0, 1, n); Xc = rng.choice(["lo", "hi"], n)
reg_df = pd.DataFrame({"y": 3 + 2 * X1 - 1.5 * X2 + (Xc == "hi") * 1.2 + rng.normal(0, .7, n), "x1": X1, "x2": X2, "c": Xc}); P3 = os.path.join(OUT, "reg.csv"); reg_df.to_csv(P3, index=False)
# GRR: 10 parts x 3 operators x 3 reps, known sigmas
sp, so, sr = 2.0, 0.4, 0.5
pv = rng.normal(0, sp, 10); ov = rng.normal(0, so, 3); rows = []
for i in range(10):
    for j in range(3):
        for r in range(3): rows.append({"part": f"P{i+1}", "op": f"O{j+1}", "meas": 50 + pv[i] + ov[j] + rng.normal(0, sr)})
grr_df = pd.DataFrame(rows); P4 = os.path.join(OUT, "grr.csv"); grr_df.to_csv(P4, index=False)
cap_df = pd.DataFrame({"x": rng.normal(10, 1, 500)}); P5 = os.path.join(OUT, "cap.csv"); cap_df.to_csv(P5, index=False)
ctl_df = pd.DataFrame({"x": np.r_[rng.normal(100, 2, 40), rng.normal(106, 2, 5)]}); P6 = os.path.join(OUT, "ctl.csv"); ctl_df.to_csv(P6, index=False)
sub_df = pd.DataFrame({"x": rng.normal(20, 1, 125), "sg": np.repeat(range(25), 5)}); P6b = os.path.join(OUT, "sub.csv"); sub_df.to_csv(P6b, index=False)
beta_t, eta_t = 2.0, 100.0
tt = eta_t * rng.weibull(beta_t, 300); cens = rng.uniform(40, 160, 300); life_df = pd.DataFrame({"t": np.minimum(tt, cens), "fail": (tt <= cens).astype(int), "grp": rng.choice(["a", "b"], 300)})
P7 = os.path.join(OUT, "life.csv"); life_df.to_csv(P7, index=False)
# DOE 2^3 with known effects (coef = effect/2)
import itertools
des = np.array(list(itertools.product([-1, 1], repeat=3))); A, B, C = des.T
doe_df = pd.DataFrame({"A": 100 + 10 * A, "B": 5 + 2 * B, "C": 50 + 25 * C, "yld": 60 + 4 * A + 2 * B + 0.0 * C + 3 * A * B})
doe_rep = pd.concat([doe_df, doe_df], ignore_index=True); doe_rep["yld"] += rng.normal(0, .3, len(doe_rep)); P8 = os.path.join(OUT, "doe.csv"); doe_rep.to_csv(P8, index=False)
P8s = os.path.join(OUT, "doe_sat.csv"); doe_df.to_csv(P8s, index=False)
bin_df = pd.DataFrame({"x": rng.normal(0, 1, 300)}); bin_df["y"] = (rng.random(300) < 1 / (1 + np.exp(-(0.5 + 1.5 * bin_df["x"])))).astype(int); P9 = os.path.join(OUT, "bin.csv"); bin_df.to_csv(P9, index=False)
mv = pd.DataFrame(rng.normal(0, 1, (150, 4)), columns=list("abcd")); mv["b"] += mv["a"]; mv["grp"] = rng.choice(["u", "v", "w"], 150); P10 = os.path.join(OUT, "mv.csv"); mv.to_csv(P10, index=False)

# ---------------- numerical verification ----------------
import ss_linear, ss_spc, ss_rel, ss_desc
from ss_common import Lang
L = Lang("en")
# 1. one-way ANOVA vs scipy
m = ss_linear.OLS(oneway_df, "y", ["grp"]); a = m.anova_table(); F_sp = stats.f_oneway(*[d["y"] for _, d in oneway_df.groupby("grp")])
check("oneway F matches scipy", abs(a.iloc[0]["F Ratio"] - F_sp.statistic) < 1e-8, f"{a.iloc[0]['F Ratio']:.6f} vs {F_sp.statistic:.6f}")
check("oneway p matches scipy", abs(a.iloc[0]["Prob > F"] - F_sp.pvalue) < 1e-10)
# 2. regression vs numpy lstsq
mr = ss_linear.OLS(reg_df, "y", [], ["x1", "x2"]); Xd = np.column_stack([np.ones(n), reg_df.x1, reg_df.x2]); b_np = np.linalg.lstsq(Xd, reg_df.y.values, rcond=None)[0]
check("regression coefficients match lstsq", np.allclose(mr.beta, b_np, atol=1e-10))
lr = stats.linregress(reg_df.x1, reg_df.y); m1 = ss_linear.OLS(reg_df, "y", [], ["x1"])
check("simple regression slope/SE/R2 match scipy", abs(m1.beta[1] - lr.slope) < 1e-10 and abs(m1.se[1] - lr.stderr) < 1e-10 and abs(m1.r2 - lr.rvalue ** 2) < 1e-10)
# 3. Type III on ANCOVA-style model: sum of Type III p for single term equals t-test p
mc = ss_linear.OLS(reg_df, "y", ["c"], ["x1"], degree=1); t3 = mc.ss3(); pt = mc.params_table()
check("Type III F == t^2 for 1-df term", abs(t3[t3.Source == "x1"]["F Ratio"].iloc[0] - pt[pt.Term == "x1"]["t Ratio"].iloc[0] ** 2) < 1e-8)
# 4. t-test vs scipy
rt = cli("ttest", P2, {"mode": "two", "column": "y", "group": "g"}); ws = stats.ttest_ind(two_df[two_df.g == "x"].y, two_df[two_df.g == "y"].y, equal_var=False)
check("Welch t-test p matches scipy", rt and abs(rt["summary"]["welch_p"] - ws.pvalue) < 1e-10)
# 5. SPC constants
c5 = ss_spc.spc_constants(5)
check("SPC constants n=5 (D4=2.114, A2=0.577, c4=0.9400, B4=2.089)", abs(c5["D4"] - 2.114) < 2e-3 and abs(c5["A2"] - 0.577) < 1e-3 and abs(c5["c4"] - 0.9400) < 1e-4 and abs(c5["B4"] - 2.089) < 2e-3, str({k: round(v, 4) for k, v in c5.items()}))
# 6. Capability: N(10,1) LSL 7 USL 13 -> Cp~1
rc = cli("capability", P5, {"column": "x", "LSL": 7, "USL": 13}); check("Cp/Cpk approx 1.0 for N(10,1), spec 7..13", rc and abs(rc["summary"]["Cpk"] - 1.0) < 0.12 and abs(rc["summary"]["Pp"] - 1.0) < .08, str({k: round(v, 3) for k, v in rc["summary"].items() if isinstance(v, float)}) if rc else "")
xs = cap_df.x.values; Pp_exact = 6 / (6 * xs.std(ddof=1)); check("Pp formula exact", abs(rc["summary"]["Pp"] - Pp_exact) < 1e-9)
# 7. Gage R&R: components recovered + identities
rg = cli("gage_rr", P4, {"part": "part", "operator": "op", "measurement": "meas", "tolerance": 20})
s = rg["summary"]; check("GRR repeatability sigma ~0.5", abs(s["sd_repeatability"] - sr) < .08, f"{s['sd_repeatability']:.3f}")
check("GRR part sigma ~2 (sampled parts)", abs(s["sd_part"] - pv.std(ddof=1)) < .6, f"{s['sd_part']:.3f} vs sample {pv.std(ddof=1):.3f}")
tot_check = s["sd_part"] ** 2 + s["sd_repeatability"] ** 2 + s["sd_reproducibility"] ** 2
check("GRR %Study Var consistent with components", abs(s["pct_study_var"] - 100 * np.sqrt(s["sd_repeatability"] ** 2 + s["sd_reproducibility"] ** 2) / np.sqrt(tot_check)) < 1e-6)
# 8. Weibull recovery (with censoring) & uncensored vs scipy
t_, e_ = life_df.t.values, life_df.fail.values; fw = ss_rel.fit_dist("weibull", t_, e_)
check("Weibull MLE beta,eta recovered (censored)", abs(np.exp(fw["theta"][0]) - beta_t) < .25 and abs(np.exp(fw["theta"][1]) - eta_t) < 10, f"beta={np.exp(fw['theta'][0]):.3f} eta={np.exp(fw['theta'][1]):.2f}")
tu = eta_t * rng.weibull(beta_t, 200); fu = ss_rel.fit_dist("weibull", tu, np.ones(200)); c_sp, _, sc_sp = stats.weibull_min.fit(tu, floc=0)
check("Weibull MLE matches scipy (uncensored)", abs(np.exp(fu["theta"][0]) - c_sp) < 1e-3 and abs(np.exp(fu["theta"][1]) - sc_sp) / sc_sp < 1e-3, f"{np.exp(fu['theta'][0]):.4f} vs {c_sp:.4f}")
# 9. Kaplan-Meier hand example: times 1,2,3,4,5 events 1,1,0,1,0 -> S(1)=.8, S(2)=.6, S(4)=.6*(1-1/2)=.3
k_ = ss_rel.km(np.array([1, 2, 3, 4, 5]), np.array([1, 1, 0, 1, 0])); check("Kaplan-Meier hand calc (0.8, 0.6, 0.3)", np.allclose(k_["Survival"].values, [.8, .6, .3]))
# 10. Log-rank vs known: identical groups -> p large; different -> small
c1, d1, p1 = ss_rel.logrank([(rng.exponential(10, 80), np.ones(80, int)), (rng.exponential(10, 80), np.ones(80, int))]); c2, d2, p2 = ss_rel.logrank([(rng.exponential(10, 80), np.ones(80, int)), (rng.exponential(25, 80), np.ones(80, int))])
check("Log-rank sensible (null p>0.01, alt p<0.001)", p1 > .01 and p2 < .001, f"p0={p1:.3f} p1={p2:.2e}")
# 11. DOE effects recovered exactly (saturated, no noise): A=8, B=4, AB=6, C=0
rd = cli("doe_analyze", P8s, {"response": "yld", "factors": ["A", "B", "C"], "degree": 2}, name="doe_sat")
ef = rd["summary"]["effects"]
check("DOE effects exact (A=8, B=4, A*B=6, C=0)", abs(ef["A"] - 8) < 1e-9 and abs(ef["B"] - 4) < 1e-9 and abs(ef["A*B"] - 6) < 1e-9 and abs(ef["C"]) < 1e-9, str({k: round(v, 3) for k, v in ef.items()}))
check("DOE Lenth flags A (and A*B) as significant", "A" in rd["summary"]["significant"], str(rd["summary"]["significant"]))
# 12. Logistic vs sklearn
rl = cli("glm", P9, {"response": "y", "predictors": ["x"], "family": "binomial"})
from sklearn.linear_model import LogisticRegression
sk = LogisticRegression(penalty=None, max_iter=1000).fit(bin_df[["x"]], bin_df.y); mm = ss_linear  # coefficient check via CLI table
tab = pd.read_csv([os.path.join(OUT, "glm_en", f) for f in os.listdir(os.path.join(OUT, "glm_en")) if "Parameter" in f][0])
check("Logistic coef matches sklearn (unpenalized)", abs(tab.Estimate.iloc[1] - sk.coef_[0][0]) < 1e-4 and abs(tab.Estimate.iloc[0] - sk.intercept_[0]) < 1e-4, f"{tab.Estimate.iloc[1]:.5f} vs {sk.coef_[0][0]:.5f}")
# 13. Control chart: I-MR limits and an out-of-control signal
rcc = cli("control_chart", P6, {"type": "imr", "column": "x"}); check("I-MR detects the shifted tail (OOC>0)", rcc and rcc["summary"]["n_ooc"] > 0, str(rcc["summary"]))
x_ = ctl_df.x.values[:40]; sig_ = np.abs(np.diff(ctl_df.x.values)).mean() / 1.128
# 14. power: n for d=0.5, alpha .05, power .8 two-sample  -> 64 per group
rp = cli("power", None, {"test": "ttest2", "effect_size": 0.5, "alpha": .05, "power": .8}); check("Power: d=0.5 needs n=64/group", rp and rp["summary"]["n_required_per_group"] in (63, 64, 65), str(rp["summary"] if rp else ""))

# ---------------- exercise every analysis (EN + HE) ----------------
runs = [("distribution", P, {"columns": ["y"], "by": "grp"}), ("ttest", P2, {"mode": "two", "column": "y", "group": "g"}), ("correlation", P10, {"columns": ["a", "b", "c", "d"]}),
        ("crosstab", P10, {"row": "grp", "col": "grp"}), ("pca", P10, {"columns": ["a", "b", "c", "d"], "color_by": "grp"}), ("cluster", P10, {"columns": ["a", "b", "c", "d"]}),
        ("oneway", P, {"response": "y", "factor": "grp"}), ("anova", P3, {"response": "y", "factors": ["c"], "covariates": ["x1"]}), ("regression", P3, {"response": "y", "predictors": ["x1", "x2"], "factors": ["c"], "stepwise": True}),
        ("glm", P9, {"response": "y", "predictors": ["x"]}), ("control_chart", P6b, {"type": "xbar_r", "column": "x", "subgroup": "sg"}), ("control_chart", P6b, {"type": "xbar_s", "column": "x", "subgroup_size": 5}),
        ("capability", P5, {"column": "x", "LSL": 7, "USL": 13, "target": 10, "distribution": "normal"}), ("gage_rr", P4, {"part": "part", "operator": "op", "measurement": "meas"}),
        ("life_distribution", P7, {"time": "t", "event": "fail", "at_times": [50, 100]}), ("survival", P7, {"time": "t", "event": "fail", "group": "grp"}),
        ("doe_analyze", P8, {"response": "yld", "factors": ["A", "B", "C"], "goal": "max"})]
for an, d, p in runs:
    for lang in ("en", "he"):
        r = cli(an, d, p, lang, name=f"{an}_{lang}_{abs(hash(json.dumps(p)))%1000}")
        check(f"run {an} [{lang}] produces docx+figures", bool(r) and os.path.exists(r["docx"]) and all(os.path.exists(f) for f in r["figures"]) and len(r["figures"]) >= 1)
for typ, p in (("full", {"type": "full", "factors": {"T": [150, 200], "P": [1, 3], "t": [10, 30]}, "center_points": 3}), ("fractional", {"type": "fractional", "factors": {"A": [0, 1], "B": [0, 1], "C": [0, 1], "D": [0, 1]}, "generators": ["D=ABC"]}),
               ("ccd", {"type": "ccd", "factors": {"A": [0, 1], "B": [0, 1]}}), ("boxbehnken", {"type": "boxbehnken", "factors": {"A": [0, 1], "B": [0, 1], "C": [0, 1]}})):
    r = cli("doe_design", None, p, name=f"design_{typ}"); check(f"doe_design {typ}", bool(r) and r["summary"]["runs"] > 0, str(r["summary"]) if r else "")
print(f"\n{len(ok)} passed, {len(bad)} failed"); print("FAILED:", bad) if bad else None
print("OUT:", OUT)
sys.exit(1 if bad else 0)
