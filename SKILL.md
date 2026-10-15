---
name: jmp-spss-analysis
description: "ניתוח JMP + SPSS - JMP/SPSS-style statistical analysis in Python without either program (71 analyses): descriptive, t-tests, nonparametric, ANOVA/MANOVA/repeated measures, regression/GLM/mixed models, factor/discriminant/PCA/cluster, SPC (control charts incl. EWMA/CUSUM/Laney/T2), Cpk, Gage R&R/MSA/attribute agreement, reliability (Weibull, Cox, ALT, Crow-AMSAA), DOE (DSD, D-optimal, Taguchi, mixture, desirability), time series, graph builder. Use when the user mentions JMP, SPSS, ניתוח JMP+SPSS, industrial statistics, Cpk, Gage R&R, DOE, Weibull, ANOVA, regression, control chart, from CSV/Excel/.sav data. Asks for parameters, runs, checks assumptions, delivers Word report + PNG graphs."
---

# ניתוח JMP + SPSS  (JMP + SPSS Analysis) — one skill, no licences

Entry point: `scripts/run.py` (numpy / scipy / pandas / matplotlib / python-docx / scikit-learn; `pyreadstat` only for `.sav`).
Every analysis writes: Word report (`.docx`: tables + figures + notes), PNG figures, CSV tables, `*_summary.json` with the key numbers.
Developed by MIGATRONIX – Automation & Control Solutions (Engineering & Development: Ashraf Saed).

```
python scripts/run.py list                                   # all 71 analyses with JMP/SPSS equivalents
python scripts/run.py profile --data data.xlsx [--sheet S]    # columns + modeling types (continuous / ordinal / nominal)
python scripts/run.py <analysis> --data data.csv --params '{...json...}' --out ./out --lang en|he [--name base] [--title "..."]
python scripts/make_catalog.py                               # regenerate references/catalog.md (exact params of every analysis)
```
Analyses that need no data file: `power`, `doe_design`, `doe_design2`, `doe_power`, `reliability_plan`, `sampling_plan`, `proportion` (when counts are given).

## Workflow (follow in order)

1. **Understand the goal.** If the user writes Hebrew, first restate the request in one clean line of Hebrew (never silently "correct" reversed Hebrew), then continue. Pick the analysis from the tables below. If genuinely ambiguous (e.g. "compare machines" = ANOVA vs Gage R&R), ask.
2. **Load & profile the data** with `profile`. Show column names with modeling types. `.jmp` cannot be read: ask for export to Excel/CSV (JMP: *File > Save As*). `.sav` needs `pyreadstat`.
3. **Ask for the parameters** in ONE `AskUserQuestion` call (max 4 questions), only what is missing and not inferable from the profile: response / factor / part / operator / time / event columns; **spec limits, tolerance, equivalence margins, censoring flags (never invent these)**; subgroup column/size; alpha/confidence; report language (**English or Hebrew — ask per analysis**). Offer sensible defaults first (marked Recommended). Exact keys per analysis: `references/catalog.md`.
4. **Run** `run.py`; read the JSON (`summary`, `notes`). Open at least one figure with `Read` (labels, Hebrew, empty panels).
5. **Check assumptions before interpreting** (module notes): normality of residuals, equal variance, independence/stability, balance, censoring %, sample size, model fit (R², Q², CV). Report violations plainly and re-run with the right alternative (Welch, Kruskal, non-normal capability, transform, Laney, REML…).
6. **Deliver.** `SendUserFile` the `.docx` (+ key PNGs if asked). 3–6 lines: result in one sentence, key numbers with units, assumption status, the decision it supports. Tag uncertain statements [ASSUMPTION]/[ESTIMATE]/[REQUIRES DATA].

## Which analysis? (JMP / SPSS equivalents)

| Area | User wants → `analysis` | JMP | SPSS | Doc |
|---|---|---|---|---|
| Descriptive | `distribution`, `tabulate`, `correlation`, `partial_corr`, `crosstab`, `pca`, `cluster` | Distribution, Tabulate, Multivariate, PCA, Cluster | Explore, Descriptives, Correlate, Crosstabs, Factor, Cluster | descriptive.md, stats-extra.md |
| Compare means / ranks | `ttest`, `nonparametric`, `equivalence`, `variance_test`, `proportion`, `multiple_testing` | Fit Y by X, Equivalence, Unequal Var | Compare Means, Nonparametric | descriptive.md, stats-extra.md |
| ANOVA family | `oneway`, `anova`, `rm_anova`, `manova`, `mixed`, `nested_anova` | Fit Model (+REML), Variance Components | One-Way, GLM Univariate/Repeated/Multivariate, MIXED, VARCOMP | linear-models.md, stats-extra.md, models-extra.md |
| Regression / GLM | `regression`, `glm`, `ordinal`, `multinomial`, `nonlinear`, `penalized`, `power` | Fit Model, Fit Curve, Generalized Regression | Linear/Logistic/PLUM/NOMREG/NLR | linear-models.md, models-extra.md |
| Multivariate / ML | `factor_analysis`, `discriminant`, `predictive`, `pls`, `hotelling_t2` | Factor Analysis, Discriminant, Partition/Bootstrap Forest, PLS | Factor, Discriminant, Decision Tree | models-extra.md |
| Time series | `timeseries` | Time Series | Forecasting | models-extra.md |
| Distribution fitting | `transform`, `tolerance`, `outliers`, `bootstrap` | Distribution, Explore Outliers, Bootstrap | Explore, Bootstrap | stats-extra.md |
| Agreement | `agreement`, `icc`, `cronbach`, `bland_altman` | Attribute Gauge | Reliability, Crosstabs Kappa | stats-extra.md |
| SPC | `control_chart`, `ewma`, `cusum`, `laney`, `run_chart`, `pareto`, `hotelling_t2`, `sampling_plan` | Control Chart Builder, Pareto, Multivariate CC | Control Charts | spc-msa.md, spc-rel-extra.md |
| Capability / MSA | `capability`, `gage_rr`, `gage_type1`, `gage_linearity_bias`, `gage_nested`, `variability_chart`, `agreement` | Process Capability, MSA, Variability Chart | — | spc-msa.md, spc-rel-extra.md |
| Reliability | `life_distribution`, `life_interval`, `survival`, `cox_ph`, `repairable`, `alt`, `degradation`, `reliability_plan` | Life Distribution, Survival, Proportional Hazards, Fit Life by X, Degradation, Reliability Growth | Kaplan-Meier, Cox | reliability.md, spc-rel-extra.md |
| DOE | `doe_design`, `doe_design2`, `doe_augment`, `doe_power`, `doe_analyze`, `taguchi_analyze`, `mixture_analyze`, `doe_optimize` | DOE, Fit Model, Prediction Profiler | — | doe.md, doe-graph-extra.md |
| Graphs | `graph` (scatter/line/bar/box/violin/hist/density/heatmap/bubble/pairs/area/pie + fits, colour, facets), `auto_xy` (Fit Y by X) | Graph Builder, Fit Y by X | Chart Builder | doe-graph-extra.md |

## Rules that keep results trustworthy

- **Numbers come from the code, never from memory.** Quote the report/summary; do not re-derive by hand.
- **Verified:** `tests/run_tests.py` (core, 59 checks) and `tests/run_tests2.py` (extended, 237 checks) compare against scipy, numpy, sklearn, closed forms, published values (Shrout-Fleiss ICC, Fleiss kappa, NIST Grubbs, success-run sample sizes…), simulation truth and cross-method identities (Cox score = log-rank, REML = ANOVA, TOST vs Welch, DOE power vs Monte-Carlo). Re-run both after any code change: `python tests/run_tests.py out1; python tests/run_tests2.py out2` (≈6 min for the second).
- **Where results can differ slightly from JMP/SPSS** (explain if the user compares): Type III SS use sum-to-zero coding (SPSS parameter estimates use reference-cell coding); within-σ for individuals = MRbar/1.128; Gage R&R = AIAG crossed ANOVA, interaction pooled at α = 0.25; Weibull/ALT/Cox CIs are Wald; mixed-model fixed-effect tests are large-sample Wald (no Satterthwaite/Kenward-Roger); ARIMA by conditional least squares; Howe approximation for two-sided tolerance factors; DSD sizes follow prime-order conference matrices. Confirm once on an analysis the user already knows from JMP/SPSS.
- **Thresholds are conventions, not standards:** Cpk ≥ 1.33, %GRR < 10 %/30 %, ndc ≥ 5, kappa > 0.75, Cg/Cgk ≥ 1.33, alpha ≥ 0.7, VIP > 1, KMO > 0.6 are guidelines. Never claim ISO/IEC/AIAG compliance from these outputs; the customer specification governs.
- **Stability before capability:** Cpk/Ppk only for a stable process — run `control_chart` first.
- **Never invent** spec limits, equivalence margins, censoring indicators, reference values or tolerances; they come from the user.
- **Small samples:** warn for n < 30 (capability), < 10 failures (life data), < 20 per group (ANOVA diagnostics), few runs (DOE: check error df and model R²).
- **Model-based optima** (DOE, desirability, Taguchi) are predictions inside the tested region: require confirmation runs.
- **Not implemented** (say so, do not fake): split-plot / restricted-randomization DOE, exact (ML) ARIMA & seasonal ARIMA terms, GLMM, Kenward-Roger df, unit-root tests, survey/complex-sample analysis, competing-risks and frailty survival, reading `.jmp` files directly. If a library or file type is missing, state exactly what is missing.
