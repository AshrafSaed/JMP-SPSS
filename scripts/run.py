#!/usr/bin/env python3
"""run.py - single entry point for the "JMP + SPSS Analysis" skill (ניתוח JMP + SPSS).

Usage:
  python run.py profile  --data file.xlsx [--sheet NAME]
  python run.py <analysis> --data file.csv --params '{"response":"Y","factor":"A"}' --out outdir [--lang en|he] [--name report] [--sheet NAME]
  python run.py list

Prints a JSON line with paths (docx, figures) and the machine-readable summary.
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
from ss_common import Lang, load_data, profile, finish
import ss_desc, ss_linear, ss_spc, ss_rel, ss_doe
import ss_stats2, ss_models2, ss_spc2, ss_rel2, ss_doe2, ss_graph

ANALYSES = {
    # descriptive / multivariate
    "distribution": (ss_desc.distribution, "Distribution: moments, quantiles, normality tests, histogram/box/QQ  [JMP Distribution | SPSS Explore]"),
    "ttest": (ss_desc.ttest, "t-tests one/two/paired + nonparametric  [JMP Fit Y by X | SPSS Compare Means]"),
    "correlation": (ss_desc.correlation, "Correlation matrix + scatterplot matrix  [JMP Multivariate | SPSS Correlate]"),
    "crosstab": (ss_desc.crosstab, "Contingency table + chi-square/Fisher  [JMP Contingency | SPSS Crosstabs]"),
    "pca": (ss_desc.pca, "Principal components  [JMP PCA | SPSS Factor/PCA]"),
    "cluster": (ss_desc.cluster, "K-means / hierarchical clustering  [JMP Cluster | SPSS Cluster]"),
    # linear models
    "oneway": (ss_linear.oneway, "One-way ANOVA + Welch + Kruskal + Tukey/Dunnett  [JMP Fit Y by X | SPSS One-Way ANOVA]"),
    "anova": (ss_linear.glm_anova, "Multi-way ANOVA / ANCOVA, Type III  [JMP Fit Model | SPSS GLM Univariate]"),
    "regression": (ss_linear.regression, "Multiple/polynomial regression, stepwise, diagnostics  [JMP Fit Model | SPSS Linear Regression]"),
    "glm": (ss_linear.glm, "Logistic / Poisson regression  [JMP Fit Model (nominal) | SPSS Logistic]"),
    "power": (ss_linear.power, "Sample size & power (no data)  [JMP DOE>Sample Size | SPSS Power]"),
    # SPC / MSA
    "control_chart": (ss_spc.control_chart, "I-MR, Xbar-R, Xbar-S, P, NP, C, U charts with Nelson rules  [JMP Control Chart Builder | SPSS Control Charts]"),
    "capability": (ss_spc.capability, "Cp/Cpk/Pp/Ppk/Cpm, PPM, non-normal  [JMP Process Capability]"),
    "gage_rr": (ss_spc.gage_rr, "Crossed Gage R&R (ANOVA), ndc, variability chart  [JMP MSA | Minitab-style]"),
    # reliability
    "life_distribution": (ss_rel.life_distribution, "Weibull/lognormal/exponential with censoring  [JMP Life Distribution]"),
    "survival": (ss_rel.survival, "Kaplan-Meier + log-rank  [JMP Survival | SPSS Kaplan-Meier]"),
    # DOE
    "doe_design": (ss_doe.make_design, "Generate full/fractional/CCD/Box-Behnken design  [JMP DOE]"),
    "doe_analyze": (ss_doe.doe_analyze, "Analyze DOE: effects, Lenth, Pareto, profiler-style optimum  [JMP Fit Model + Profiler]"),
    # ---- classical tests & agreement (ss_stats2)
    "nonparametric": (ss_stats2.nonparametric, "Mann-Whitney, Wilcoxon, Kruskal+Dunn, Friedman, sign, McNemar, median, runs  [SPSS Nonparametric | JMP Fit Y by X]"),
    "proportion": (ss_stats2.proportion, "One/two-sample proportions: Wilson/exact CI, z, Fisher, RR, OR  [JMP Distribution/Contingency | SPSS Crosstabs]"),
    "variance_test": (ss_stats2.variance_test, "Levene, Brown-Forsythe, Bartlett, Fligner, F-test  [JMP UnEqual Variance | SPSS Explore]"),
    "equivalence": (ss_stats2.equivalence, "TOST equivalence test one/two/paired  [JMP Equivalence Test]"),
    "rm_anova": (ss_stats2.rm_anova, "Repeated-measures ANOVA, Mauchly, Greenhouse-Geisser/Huynh-Feldt  [SPSS GLM Repeated Measures | JMP Fit Model]"),
    "manova": (ss_stats2.manova, "One-way MANOVA: Wilks/Pillai/Hotelling-Lawley/Roy + canonical plot  [JMP Fit Model MANOVA | SPSS GLM Multivariate]"),
    "agreement": (ss_stats2.agreement, "Cohen/weighted/Fleiss kappa and attribute agreement analysis  [JMP Attribute Gauge | SPSS Crosstabs Kappa]"),
    "icc": (ss_stats2.icc, "Intraclass correlation (Shrout-Fleiss 1-3, single/average)  [SPSS Reliability]"),
    "cronbach": (ss_stats2.cronbach, "Cronbach alpha, item-total, alpha-if-deleted  [SPSS Reliability | JMP Item Reliability]"),
    "bland_altman": (ss_stats2.bland_altman, "Bland-Altman limits of agreement, proportional bias  [method comparison]"),
    "outliers": (ss_stats2.outliers, "Grubbs, generalized ESD, Tukey, modified z, Mahalanobis  [JMP Explore Outliers | SPSS Explore]"),
    "tolerance": (ss_stats2.tolerance, "Normal & distribution-free tolerance intervals, prediction intervals  [JMP Distribution > Tolerance Interval]"),
    "transform": (ss_stats2.transform, "Box-Cox / Yeo-Johnson normalizing transform  [JMP Distribution > Normal Quantile/Johnson]"),
    "bootstrap": (ss_stats2.bootstrap, "Bootstrap CI (percentile & BCa) for mean/median/sd/corr/Cpk  [JMP Bootstrapping | SPSS Bootstrap]"),
    "multiple_testing": (ss_stats2.multiple_testing, "Bonferroni/Sidak/Holm/Hochberg/BH/BY adjusted p-values  [multiple comparison]"),
    "tabulate": (ss_stats2.tabulate, "Summary tables by groups + frequencies  [JMP Tabulate | SPSS Descriptives/Frequencies/Means]"),
    "partial_corr": (ss_stats2.partial_corr, "Partial & semi-partial correlation  [SPSS Partial Correlations]"),
    # ---- models (ss_models2)
    "mixed": (ss_models2.mixed, "Mixed model / variance components by REML (random effects, nested)  [JMP Fit Model REML | SPSS MIXED]"),
    "nested_anova": (ss_models2.nested_anova, "Nested ANOVA with variance components  [JMP Variance Components | SPSS VARCOMP]"),
    "factor_analysis": (ss_models2.factor_analysis, "KMO, Bartlett, PC/ML factors, varimax, scree  [JMP Factor Analysis | SPSS Factor]"),
    "discriminant": (ss_models2.discriminant, "LDA/QDA with cross-validation  [JMP Discriminant | SPSS Discriminant]"),
    "ordinal": (ss_models2.ordinal, "Proportional-odds ordinal logistic regression  [JMP Fit Model (ordinal) | SPSS PLUM]"),
    "multinomial": (ss_models2.multinomial, "Multinomial logistic regression  [JMP Fit Model (nominal) | SPSS NOMREG]"),
    "nonlinear": (ss_models2.nonlinear, "Nonlinear curve fit (exp, logistic 3P/4P, MM, Gompertz, custom expr)  [JMP Fit Curve/Nonlinear | SPSS NLR]"),
    "penalized": (ss_models2.penalized, "Ridge / Lasso / Elastic net with CV  [JMP Generalized Regression]"),
    "predictive": (ss_models2.predictive, "Decision tree, random forest, gradient boosting with CV  [JMP Partition/Bootstrap Forest/Boosted Tree | SPSS Decision Tree]"),
    "pls": (ss_models2.pls, "Partial least squares with VIP and CV  [JMP PLS]"),
    "timeseries": (ss_models2.timeseries, "ACF/PACF, decomposition, Holt-Winters, ARIMA, auto-ARIMA, forecast  [JMP Time Series | SPSS Forecasting]"),
    # ---- advanced SPC / MSA (ss_spc2)
    "ewma": (ss_spc2.ewma, "EWMA control chart  [JMP Control Chart Builder]"),
    "cusum": (ss_spc2.cusum, "Tabular CUSUM chart with shift estimate  [JMP CUSUM]"),
    "laney": (ss_spc2.laney, "Laney P'/U' charts for overdispersed attribute data  [Control Chart Builder]"),
    "pareto": (ss_spc2.pareto, "Pareto chart  [JMP Pareto Plot]"),
    "run_chart": (ss_spc2.run_chart, "Run chart with runs tests  [Minitab-style run chart]"),
    "hotelling_t2": (ss_spc2.hotelling_t2, "Multivariate T2 control chart (Phase I/II)  [JMP Multivariate Control Chart]"),
    "gage_type1": (ss_spc2.gage_type1, "Type 1 gage study: Cg/Cgk, bias  [MSA]"),
    "gage_linearity_bias": (ss_spc2.gage_linearity_bias, "Gage linearity & bias  [JMP MSA Bias and Linearity]"),
    "gage_nested": (ss_spc2.gage_nested, "Nested (destructive) Gage R&R  [JMP MSA | Minitab Nested GRR]"),
    "variability_chart": (ss_spc2.variability_chart, "Variability / multi-vari chart with nested variance components  [JMP Variability Chart]"),
    "sampling_plan": (ss_spc2.sampling_plan, "Attribute acceptance sampling plan, OC curve, AOQ (no data)  [Minitab/JMP Acceptance Sampling]"),
    # ---- advanced reliability (ss_rel2)
    "cox_ph": (ss_rel2.cox_ph, "Cox proportional hazards (Efron/Breslow), HR, PH check  [JMP Proportional Hazards | SPSS Cox Regression]"),
    "repairable": (ss_rel2.repairable, "Crow-AMSAA power-law NHPP, Laplace trend, MCF  [JMP Reliability Growth / Recurrence]"),
    "alt": (ss_rel2.alt, "Accelerated life test: Arrhenius / inverse power / linear, Weibull or lognormal  [JMP Fit Life by X]"),
    "degradation": (ss_rel2.degradation, "Degradation paths -> pseudo failure times -> life distribution  [JMP Degradation]"),
    "life_interval": (ss_rel2.life_interval, "Interval/left/right-censored life data MLE  [JMP Life Distribution]"),
    "reliability_plan": (ss_rel2.reliability_plan, "Success-run and Weibayes demonstration test plans (no data)  [JMP Reliability Test Plan]"),
    # ---- advanced DOE (ss_doe2)
    "doe_design2": (ss_doe2.make_design2, "Plackett-Burman, DSD, D-optimal, Taguchi, mixture, Latin square designs  [JMP DOE]"),
    "doe_augment": (ss_doe2.doe_augment, "Foldover / axial / center-point augmentation  [JMP DOE Augment]"),
    "doe_power": (ss_doe2.doe_power, "Power & runs for two-level factorial effects (no data)  [JMP DOE Evaluate Design]"),
    "taguchi_analyze": (ss_doe2.taguchi_analyze, "Taguchi S/N response tables, ANOVA, predicted optimum  [JMP Taguchi]"),
    "mixture_analyze": (ss_doe2.mixture_analyze, "Scheffe mixture model, ternary contour, blend optimum  [JMP Mixture]"),
    "doe_optimize": (ss_doe2.doe_optimize, "Multi-response desirability optimisation + prediction profiler  [JMP Prediction Profiler]"),
    # ---- graphs
    "graph": (ss_graph.graph, "Graph Builder: scatter/line/bar/box/violin/hist/density/heatmap/bubble/pairs/area/pie + fits, colour, facets  [JMP Graph Builder | SPSS Chart Builder]"),
    "auto_xy": (ss_graph.auto_xy, "Fit Y by X: auto-picks ANOVA / regression / logistic / contingency  [JMP Fit Y by X]"),
}
NO_DATA = {"power", "doe_design", "doe_design2", "doe_power", "reliability_plan", "sampling_plan", "proportion"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("analysis")
    ap.add_argument("--data"); ap.add_argument("--sheet")
    ap.add_argument("--params", default="{}"); ap.add_argument("--out", default="./stats_out")
    ap.add_argument("--lang", default="en"); ap.add_argument("--name"); ap.add_argument("--title")
    a = ap.parse_args()
    if a.analysis == "list":
        for k, (_, d) in ANALYSES.items(): print(f"{k:18s} {d}")
        return
    if a.analysis == "profile":
        df = load_data(a.data, a.sheet)
        print(json.dumps({"rows": len(df), "columns": json.loads(profile(df).to_json(orient="records", force_ascii=False))}, ensure_ascii=False))
        return
    if a.analysis not in ANALYSES:
        sys.exit(f"Unknown analysis '{a.analysis}'. Run: python run.py list")
    params = json.loads(a.params)
    L = Lang(a.lang); os.makedirs(a.out, exist_ok=True)
    df = load_data(a.data, a.sheet) if a.data else pd.DataFrame()
    if a.analysis not in NO_DATA and df.empty:
        sys.exit("--data is required for this analysis")
    res = ANALYSES[a.analysis][0](df, params, L, a.out)
    paths = finish(res, a.out, L, basename=a.name or a.analysis, title=a.title,
                   meta={"Data": os.path.basename(a.data) if a.data else "-", "Rows": len(df), L.t("Generated by"): "JMP+SPSS Analysis (MIGATRONIX)"})
    print(json.dumps({"docx": paths["docx"], "figures": paths["figures"], "summary": res.summary, "notes": res.notes},
                     ensure_ascii=False, default=lambda o: o.item() if hasattr(o, "item") else str(o)))


if __name__ == "__main__":
    main()
