# JMP + SPSS Analysis — a Claude Skill for Industrial Statistics

*Skill display name (Hebrew): ניתוח JMP + SPSS · slug: `jmp-spss-analysis`*

A Claude Skill that reproduces JMP- and SPSS-style statistical analyses in pure Python, with no licence for either package. For each request it collects the required parameters, runs the analysis, checks the underlying assumptions, and delivers a Word report with PNG figures. The report language (English or Hebrew) is chosen per analysis.

| | |
|---|---|
| Scope | 75 analyses in 12 areas, driven by one master `SKILL.md` plus module reference documents |
| Stack | numpy, scipy, pandas, matplotlib, python-docx, scikit-learn (`pyreadstat` only for `.sav` files); no statsmodels |
| Output | `.docx` report, `.png` figures, `.csv` tables, `*_summary.json` with key results |
| Verification | 359 automated checks in three suites (see [Verification](#verification)) |
| Author | Ashraf Saed, MIGATRONIX – Automation & Control Solutions |

---

## Coverage

| Area | Analyses (`run.py` names) | JMP / SPSS counterpart |
|---|---|---|
| Descriptive & multivariate | `distribution`, `tabulate`, `correlation`, `partial_corr`, `crosstab`, `pca`, `cluster` | Distribution, Multivariate, PCA, Cluster / Explore, Correlate, Crosstabs, Factor |
| Comparisons | `ttest`, `nonparametric` (Mann-Whitney, Wilcoxon, Kruskal-Wallis + Dunn, Friedman, sign, McNemar, median, runs), `equivalence` (TOST), `variance_test`, `proportion`, `multiple_testing` | Fit Y by X, Equivalence / Compare Means, Nonparametric Tests |
| ANOVA family | `oneway`, `anova`, `rm_anova` (Mauchly, Greenhouse-Geisser / Huynh-Feldt), `manova` (multi-factor, MANCOVA), `mixed` (REML, Satterthwaite df), `nested_anova`, `split_plot` | Fit Model, Variance Components / GLM, MIXED, VARCOMP |
| Regression & GLM | `regression`, `glm`, `glmm`, `ordinal`, `multinomial`, `nonlinear`, `penalized` (LASSO / Ridge / Elastic Net), `power` | Fit Model, Fit Curve, Generalized Regression / Linear, Logistic, PLUM, NOMREG, NLR |
| Multivariate & machine learning | `factor_analysis` (KMO, Bartlett, varimax), `discriminant` (LDA / QDA), `predictive` (tree, forest, boosting), `pls` (VIP) | Factor Analysis, Discriminant, Partition, PLS |
| Time series | `timeseries` (ACF/PACF, exact-ML SARIMA, auto-ARIMA, Holt-Winters), `stationarity` (ADF + KPSS) | Time Series / Forecasting |
| Distributions & outliers | `transform` (Box-Cox, Yeo-Johnson), `tolerance`, `outliers` (Grubbs, generalized ESD), `bootstrap` | Distribution, Explore Outliers |
| Agreement & scale reliability | `agreement` (Cohen / Fleiss kappa, attribute agreement), `icc`, `cronbach`, `bland_altman` | Attribute Gauge / Reliability Analysis, Kappa |
| Statistical process control | `control_chart`, `ewma`, `cusum`, `laney`, `run_chart`, `pareto`, `hotelling_t2`, `sampling_plan` | Control Chart Builder / Control Charts |
| Capability & MSA | `capability` (Cp, Cpk, Pp, Ppk), `gage_rr`, `gage_rr_reml`, `gage_type1`, `gage_linearity_bias`, `gage_nested`, `variability_chart` | Process Capability, Measurement Systems Analysis |
| Reliability & survival | `life_distribution`, `life_interval`, `survival` (Kaplan-Meier), `cox_ph` (with Grambsch-Therneau PH test), `repairable` (Crow-AMSAA, Nelson MCF), `alt` (Arrhenius / power law), `degradation`, `reliability_plan` (Weibayes, success-run) | Life Distribution, Survival, Fit Life by X, Reliability Growth |
| Design of experiments | `doe_design` (full / fractional factorial, CCD, Box-Behnken), `doe_design2` (DSD, Plackett-Burman, D-optimal, Taguchi, mixture, Latin square, split-plot), `doe_augment`, `doe_power`, `doe_analyze`, `taguchi_analyze`, `mixture_analyze`, `doe_optimize` (desirability + profiler) | DOE, Prediction Profiler |
| Graphs | `graph` (scatter, line, bar, box, violin, histogram, density, heatmap, bubble, pairs, area, pie), `auto_xy` | Graph Builder, Fit Y by X / Chart Builder |

The complete parameter reference for every analysis is generated from the code: [`references/catalog.md`](references/catalog.md). Per-area methodology notes are in `references/*.md`.

## Workflow

1. **Request.** The user describes the analysis in English or Hebrew; Claude restates it in one line before running anything.
2. **Profile.** `profile` reads CSV, Excel or `.sav` data and infers each column's modeling type (continuous, ordinal, nominal).
3. **Parameters.** Claude asks once for whatever is missing and cannot be inferred: response, factor and grouping columns, specification limits, significance level, censoring flags, report language. Specification limits, margins and censoring indicators are never invented.
4. **Run and check.** The analysis executes together with the relevant assumption checks (normality, equal variance, stability, balance, censoring, sample size).
5. **Deliver.** Word report, PNG figures, CSV tables and a JSON summary.

```bash
python scripts/run.py list
python scripts/run.py profile --data data.xlsx
python scripts/run.py capability --data data.csv \
    --params '{"column":"x","lsl":9.5,"usl":10.5}' --out ./out --lang en
```

## Installation

Place the folder in Claude's skills directory (for example `~/.claude/skills/jmp-spss-analysis/`) or upload it as a Skill in Claude.

```bash
pip install numpy scipy pandas matplotlib python-docx scikit-learn openpyxl
pip install pyreadstat     # optional, only for .sav files
```

## Verification

Three test suites (59 + 237 + 63 = 359 checks) cover the code. Each result is compared with an independent computation:

- scipy and scikit-learn reference implementations;
- closed-form results and published values (Shrout-Fleiss ICC, Fleiss kappa, NIST Grubbs, success-run sample sizes, ADF/KPSS critical values);
- simulation truth (parameter recovery, test size and power);
- identities between methods (Cox score test = log-rank, REML = ANOVA on balanced data, Kalman likelihood = Toeplitz likelihood, DOE power vs Monte-Carlo);
- end-to-end runs of every analysis in both English and Hebrew.

```bash
python tests/run_tests.py  out1
python tests/run_tests2.py out2     # about 6 min
python tests/run_tests3.py out3     # about 3 min
```

## Methodological notes

Recent additions are documented, with their verification, in [`references/limits-fixes.md`](references/limits-fixes.md):

- Satterthwaite denominator degrees of freedom for mixed-model t and F tests (Fai-Cornelius approach for multi-df terms)
- exact Gaussian maximum-likelihood ARIMA/SARIMA by Kalman filter
- ADF and KPSS stationarity tests
- Grambsch-Therneau proportional-hazards test for Cox models
- multi-factor MANOVA and MANCOVA
- random-intercept GLMM by adaptive Gauss-Hermite quadrature
- REML Gage R&R for unbalanced data and covariates
- split-plot design generation and analysis
- SPSS-style reference-cell parameter coding (`"coding": "reference"`)

Where results can differ slightly from JMP or SPSS: Type III sums of squares use sum-to-zero coding unless `"coding": "reference"` is requested; the ADF p-value is Monte-Carlo rather than a MacKinnon response surface; confidence intervals for Weibull, ALT and Cox models are Wald-based.

## Limitations

- **`.jmp` files** cannot be read directly (proprietary format). Export to Excel or CSV from JMP. `.sav` requires `pyreadstat`.
- **Kenward-Roger** is not implemented; only Satterthwaite. With few levels of a random effect, p-values may be slightly optimistic.
- **GLMM** supports a single random intercept (binomial or Poisson); random slopes and crossed effects are not implemented.
- **Split-plot** designs are limited to full two-level whole-plot × sub-plot factorials; split-split-plot and fractional strata are not supported.
- **Not implemented:** complex-survey analysis, MANOVA with random factors, ARIMA with external regressors, competing risks, frailty models.
- **Evaluation thresholds** (Cpk ≥ 1.33, %GRR bands, ndc ≥ 5, kappa and alpha cut-offs) are industry conventions, not standards. Output must not be read as demonstrating compliance with ISO or AIAG requirements; the customer specification governs.
- Results are model-based estimates. Optimal settings from DOE, desirability or Taguchi analyses are predictions within the tested region and require confirmation runs.

---

Engineering & Development: Ashraf Saed | MIGATRONIX – Automation & Control Solutions
