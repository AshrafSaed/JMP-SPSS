# Analysis catalog (auto-generated from code - do not edit by hand)

Run `python scripts/make_catalog.py` to refresh. Parameters are JSON keys passed with `--params`. `[optional]`/defaults are as written by the function's author.

## Descriptive & multivariate  (`ss_desc.py`)

### `distribution`
Distribution: moments, quantiles, normality tests, histogram/box/QQ  [JMP Distribution | SPSS Explore]

```
params: columns (list, default all numeric), by (optional grouping column), spec limits not used here.
```

### `ttest`
t-tests one/two/paired + nonparametric  [JMP Fit Y by X | SPSS Compare Means]

```
params: mode (one|two|paired), column, group (two), column2 (paired), mu0 (one), equal_var (bool),
alternative.
```

### `correlation`
Correlation matrix + scatterplot matrix  [JMP Multivariate | SPSS Correlate]

```
params: columns (list), method (pearson|spearman|kendall).
```

### `crosstab`
Contingency table + chi-square/Fisher  [JMP Contingency | SPSS Crosstabs]

```
params: row, col.
```

### `pca`
Principal components  [JMP PCA | SPSS Factor/PCA]

```
params: columns, n_components (default all), standardize (default True), color_by (optional).
```

### `cluster`
K-means / hierarchical clustering  [JMP Cluster | SPSS Cluster]

```
params: columns, k (default: chosen by silhouette 2..8), method (kmeans|ward).
```

## Linear models, GLM, power  (`ss_linear.py`)

### `oneway`
One-way ANOVA + Welch + Kruskal + Tukey/Dunnett  [JMP Fit Y by X | SPSS One-Way ANOVA]

```
params: response, factor, posthoc (tukey|dunnett), control (level for dunnett).
```

### `anova`
Multi-way ANOVA / ANCOVA, Type III  [JMP Fit Model | SPSS GLM Univariate]

```
Multi-way ANOVA / ANCOVA / GLM univariate.  params: response, factors (list), covariates (list), degree
(interaction order, default 2 if >1 factor).
```

### `regression`
Multiple/polynomial regression, stepwise, diagnostics  [JMP Fit Model | SPSS Linear Regression]

```
params: response, predictors (numeric list), factors (optional), degree (interaction order; 1 default), poly
(polynomial degree per numeric predictor, default 1), stepwise (bool, backward), alpha (0.05), predict_at
(dict optional).
```

### `glm`
Logistic / Poisson regression  [JMP Fit Model (nominal) | SPSS Logistic]

```
Logistic (binomial) / Poisson regression by IRLS.  params: response, predictors, factors, family
(binomial|poisson), positive (level treated as 1 for binomial).
```

### `power` - **no data file needed**
Sample size & power (no data)  [JMP DOE>Sample Size | SPSS Power]

```
Sample-size / power calculator (no data needed).  params: test (ttest2|ttest1|anova), effect_size (d for
t-test, Cohen f for ANOVA), alpha, power (target), k (groups for ANOVA), n (optional - compute power at n).
```

## SPC, capability, Gage R&R  (`ss_spc.py`)

### `control_chart`
I-MR, Xbar-R, Xbar-S, P, NP, C, U charts with Nelson rules  [JMP Control Chart Builder | SPSS Control Charts]

```
params: type (imr|xbar_r|xbar_s|p|np|c|u), column (measurement or defect count), subgroup (column name with
subgroup id) OR subgroup_size (int),
n (column with sample size, for p/np/u), rules (list, default
[1,2,3,5,6]), phase_split (optional row index to freeze limits before).
```

### `capability`
Cp/Cpk/Pp/Ppk/Cpm, PPM, non-normal  [JMP Process Capability]

```
params: column, LSL, USL (at least one), target (optional), subgroup (column) or subgroup_size (default 1 ->
MR-based sigma), distribution (normal|lognormal|weibull).
```

### `gage_rr`
Crossed Gage R&R (ANOVA), ndc, variability chart  [JMP MSA | Minitab-style]

```
Crossed Gage R&R (ANOVA). params: part, operator, measurement, tolerance (USL-LSL, optional), study_sigma (6
default), alpha_int (0.25).
```

## Reliability & survival  (`ss_rel.py`)

### `life_distribution`
Weibull/lognormal/exponential with censoring  [JMP Life Distribution]

```
params: time, event (1=failure,0=right-censored; omit if all failures), distributions (default
[weibull,lognormal,exponential]), group (optional), at_times (list for reliability at time), unit (label).
```

### `survival`
Kaplan-Meier + log-rank  [JMP Survival | SPSS Kaplan-Meier]

```
Kaplan-Meier + log-rank.  params: time, event, group (optional), unit.
```

## DOE (classical)  (`ss_doe.py`)

### `doe_design` - **no data file needed**
Generate full/fractional/CCD/Box-Behnken design  [JMP DOE]

```
params: type (full|fractional|ccd|boxbehnken), factors (dict name -> [low, high]) , generators (for
fractional, e.g. ['D=ABC']), center_points (int), replicates (int), randomize (bool), seed, alpha (ccd:
'face'|'rotatable').
```

### `doe_analyze`
Analyze DOE: effects, Lenth, Pareto, profiler-style optimum  [JMP Fit Model + Profiler]

```
params: response, factors (list of columns - actual or coded values, 2-level or 3-level), degree (interaction
order, default 2), quadratic (bool, for CCD/BB),
goal (max|min|target), target (value).
```

## Classical tests, agreement, reliability of scales, transforms  (`ss_stats2.py`)

### `nonparametric`
Mann-Whitney, Wilcoxon, Kruskal+Dunn, Friedman, sign, McNemar, median, runs  [SPSS Nonparametric | JMP Fit Y by X]

```
mode: mannwhitney|wilcoxon|kruskal|friedman|sign|mcnemar|median|runs.
column, group
(mannwhitney/kruskal/median), column2 (wilcoxon paired), columns (friedman), mu0 (wilcoxon/sign/runs center).
```

### `proportion` - **no data file needed**
One/two-sample proportions: Wilson/exact CI, z, Fisher, RR, OR  [JMP Distribution/Contingency | SPSS Crosstabs]

```
One-sample: (column+success) or (x,n); p0. Two-sample: (column+group+success) or (x,n,x2,n2).
```

### `variance_test`
Levene, Brown-Forsythe, Bartlett, Fligner, F-test  [JMP UnEqual Variance | SPSS Explore]

```
params: column, group. Levene, Brown-Forsythe, Bartlett, Fligner-Killeen (+ F test and ratio CI for 2 groups).
```

### `equivalence`
TOST equivalence test one/two/paired  [JMP Equivalence Test]

```
mode one|two|paired; column; group | column2; low/high margins (difference) ; alpha; mu0 for one.
```

### `rm_anova`
Repeated-measures ANOVA, Mauchly, Greenhouse-Geisser/Huynh-Feldt  [SPSS GLM Repeated Measures | JMP Fit Model]

```
columns: list of repeated-measure columns (wide format; one row per subject).
```

### `manova`
One-way MANOVA: Wilks/Pillai/Hotelling-Lawley/Roy + canonical plot  [JMP Fit Model MANOVA | SPSS GLM Multivariate]

```
responses: list; factor: one grouping column.  Wilks, Pillai, Hotelling-Lawley, Roy + univariate ANOVAs.
```

### `agreement`
Cohen/weighted/Fleiss kappa and attribute agreement analysis  [JMP Attribute Gauge | SPSS Crosstabs Kappa]

```
mode kappa: rater1, rater2 [weights linear|quadratic].  mode fleiss: raters (list).
mode attribute: appraiser,
sample, rating, [standard], [positive level].
```

### `icc`
Intraclass correlation (Shrout-Fleiss 1-3, single/average)  [SPSS Reliability]

```
columns: raters/measurements (wide: one row per subject/target).
```

### `cronbach`
Cronbach alpha, item-total, alpha-if-deleted  [SPSS Reliability | JMP Item Reliability]

```
params: columns (item columns; default all numeric). Alpha, standardized alpha, corrected item-total r, alpha
if item deleted.
```

### `bland_altman`
Bland-Altman limits of agreement, proportional bias  [method comparison]

```
params: column (method A), column2 (method B). Bias, 95% limits of agreement, proportional-bias slope.
```

### `outliers`
Grubbs, generalized ESD, Tukey, modified z, Mahalanobis  [JMP Explore Outliers | SPSS Explore]

```
column (univariate) or columns (multivariate Mahalanobis).  max_outliers for generalized ESD.
```

### `tolerance`
Normal & distribution-free tolerance intervals, prediction intervals  [JMP Distribution > Tolerance Interval]

```
params: column, coverage (default 0.99), conf (default 0.95). Normal two-/one-sided tolerance factors,
distribution-free [min,max], confidence & prediction intervals.
```

### `transform`
Box-Cox / Yeo-Johnson normalizing transform  [JMP Distribution > Normal Quantile/Johnson]

```
column; method boxcox|yeojohnson|both.
```

### `bootstrap`
Bootstrap CI (percentile & BCa) for mean/median/sd/corr/Cpk  [JMP Bootstrapping | SPSS Bootstrap]

```
column; statistic mean|median|std|var|trimmed_mean|corr|cpk; column2 (corr); lsl/usl (cpk); n_resamples; seed.
```

### `multiple_testing`
Bonferroni/Sidak/Holm/Hochberg/BH/BY adjusted p-values  [multiple comparison]

```
column with raw p-values (or params['pvalues'] list); optional label column.
```

### `tabulate`
Summary tables by groups + frequencies  [JMP Tabulate | SPSS Descriptives/Frequencies/Means]

```
columns (numeric; default all), by (list of grouping columns), stats list from N,Mean,Std
Dev,Min,Q1,Median,Q3,Max,Sum,CV%,Missing; categorical (list) for frequency tables.
```

### `partial_corr`
Partial & semi-partial correlation  [SPSS Partial Correlations]

```
x, y, controls (list).
```

## Mixed models, factor/discriminant, ordinal/multinomial, nonlinear, ML, time series  (`ss_models2.py`)

### `mixed`
Mixed model / variance components by REML (random effects, nested)  [JMP Fit Model REML | SPSS MIXED]

```
response; fixed (list of columns); categorical (list, default non-numeric); random (list, e.g.
['Batch','Batch/Sample'] or ['A:B']); degree (fixed interactions order, default 1).
```

### `nested_anova`
Nested ANOVA with variance components  [JMP Variance Components | SPSS VARCOMP]

```
response; factors: hierarchical list outer->inner, e.g. ['Plant','Batch','Sample'] (Batch nested in Plant
...). Variance components by REML + classical SS table.
```

### `factor_analysis`
KMO, Bartlett, PC/ML factors, varimax, scree  [JMP Factor Analysis | SPSS Factor]

```
columns; n_factors (default Kaiser eigenvalue>1); method pc|ml; rotation varimax|none.
```

### `discriminant`
LDA/QDA with cross-validation  [JMP Discriminant | SPSS Discriminant]

```
response (class); predictors; method lda|qda; cv: 'loo' or k.
```

### `ordinal`
Proportional-odds ordinal logistic regression  [JMP Fit Model (ordinal) | SPSS PLUM]

```
response (ordered categories; give 'order' list low->high); predictors; categorical (list).
```

### `multinomial`
Multinomial logistic regression  [JMP Fit Model (nominal) | SPSS NOMREG]

```
response (nominal); predictors; categorical; reference (optional level).
```

### `nonlinear`
Nonlinear curve fit (exp, logistic 3P/4P, MM, Gompertz, custom expr)  [JMP Fit Curve/Nonlinear | SPSS NLR]

```
x, y, model in {exponential, exp_decay, power, logistic3, logistic4, michaelis_menten, gompertz,
weibull_growth, custom}; custom: expr (e.g. 'a*np.exp(-b*x)+c'), names ['a','b','c'], p0 [..].
```

### `penalized`
Ridge / Lasso / Elastic net with CV  [JMP Generalized Regression]

```
response; predictors; method ridge|lasso|enet; l1_ratio (enet); cv folds.
```

### `predictive`
Decision tree, random forest, gradient boosting with CV  [JMP Partition/Bootstrap Forest/Boosted Tree | SPSS Decision Tree]

```
response; predictors; method tree|forest|boost; task auto|regression|classification; cv folds.
```

### `pls`
Partial least squares with VIP and CV  [JMP PLS]

```
responses (list) ; predictors; max_components; cv folds.
```

### `timeseries`
ACF/PACF, decomposition, Holt-Winters, ARIMA, auto-ARIMA, forecast  [JMP Time Series | SPSS Forecasting]

```
column; mode: explore|decompose|smooth|arima|auto_arima; period (m); horizon; p,d,q,D; trend (bool), seasonal
add|mul|none (smooth).
```

## Advanced SPC & MSA  (`ss_spc2.py`)

### `ewma`
EWMA control chart  [JMP Control Chart Builder]

```
column; [subgroup]; lambda (0.2); L (3); target (default phase-I mean); sigma (default estimated).
```

### `cusum`
Tabular CUSUM chart with shift estimate  [JMP CUSUM]

```
column; [subgroup]; k (0.5 sigma); h (5 sigma); target; sigma.
```

### `laney`
Laney P'/U' charts for overdispersed attribute data  [Control Chart Builder]

```
type p|u; count (defects/defectives column); size (n or area column).  Laney P'/U' for overdispersed attribute
data.
```

### `pareto`
Pareto chart  [JMP Pareto Plot]

```
category; [value] (count/cost column; default = row counts); top (n categories, rest -> Other).
```

### `run_chart`
Run chart with runs tests  [Minitab-style run chart]

```
params: column. Runs about the median and runs up/down tests with a run chart.
```

### `hotelling_t2`
Multivariate T2 control chart (Phase I/II)  [JMP Multivariate Control Chart]

```
columns (p variables, individual observations); phase 1|2; alpha (0.0027); for phase 2 give 'ref_rows' count
of in-control reference rows at the start (default: all).
```

### `gage_type1`
Type 1 gage study: Cg/Cgk, bias  [MSA]

```
column (repeated measurements of one reference part); reference; tolerance (USL-LSL); k_tol (0.2); k_sd (6).
```

### `gage_linearity_bias`
Gage linearity & bias  [JMP MSA Bias and Linearity]

```
reference (true value of each part); measurement; process_variation (e.g. 6*sigma process or tolerance) for
%linearity.
```

### `gage_nested`
Nested (destructive) Gage R&R  [JMP MSA | Minitab Nested GRR]

```
operator; part (unique within operator - destructive testing); measurement; tolerance (optional).
```

### `variability_chart`
Variability / multi-vari chart with nested variance components  [JMP Variability Chart]

```
response; factors: list (outer->inner, up to 3).  Multi-vari / JMP Variability chart with cell means and
variance components.
```

### `sampling_plan` - **no data file needed**
Attribute acceptance sampling plan, OC curve, AOQ (no data)  [Minitab/JMP Acceptance Sampling]

```
Attribute single sampling plan. Give n and c to get OC curve; or aql, alpha, ltpd, beta to search the smallest
plan; optional lot_size N for AOQ/ATI.
```

## Advanced reliability  (`ss_rel2.py`)

### `cox_ph`
Cox proportional hazards (Efron/Breslow), HR, PH check  [JMP Proportional Hazards | SPSS Cox Regression]

```
time, event, covariates (list); categorical (list; default non-numeric); ties efron|breslow.
```

### `repairable`
Crow-AMSAA power-law NHPP, Laplace trend, MCF  [JMP Reliability Growth / Recurrence]

```
system (id column, optional); time (cumulative age at each failure); end (observation end per system: column
name or number; default max time per system).
```

### `alt`
Accelerated life test: Arrhenius / inverse power / linear, Weibull or lognormal  [JMP Fit Life by X]

```
time, event, stress; relationship arrhenius (stress in C) | power (inverse power law, ln stress) | linear;
dist weibull|lognormal; use_stress; B-life p (default 0.1).
```

### `degradation`
Degradation paths -> pseudo failure times -> life distribution  [JMP Degradation]

```
unit; time; measure; threshold; path linear|exponential (ln y linear); dist weibull|lognormal.
```

### `life_interval`
Interval/left/right-censored life data MLE  [JMP Life Distribution]

```
lower, upper (failure known to lie in (lower, upper]; lower==upper exact; upper empty/inf = right-censored;
lower 0 = left-censored); freq (optional); dist weibull|lognormal|exponential.
```

### `reliability_plan` - **no data file needed**
Success-run and Weibayes demonstration test plans (no data)  [JMP Reliability Test Plan]

```
No data.  reliability R, confidence C, allowed failures list; optional Weibull beta + mission time for
Weibayes zero-failure test duration; optional n units.
```

## Advanced DOE  (`ss_doe2.py`)

### `doe_design2` - **no data file needed**
Plackett-Burman, DSD, D-optimal, Taguchi, mixture, Latin square designs  [JMP DOE]

```
type: plackett_burman | dsd | d_optimal | taguchi | mixture | latin_square.
factors {name:[low,high]} (not for
mixture/latin/taguchi labels); extra: runs (PB N / d_optimal n), model (main|interaction|quadratic), array
(taguchi), components + degree/centroid (mixture), size (latin_square), seed, randomize.
```

### `doe_augment`
Foldover / axial / center-point augmentation  [JMP DOE Augment]

```
df: a design with *_coded columns (from make_design). operation foldover (all signs, or 'flip': one factor) |
axial (alpha face|rotatable|number) | center (n).
```

### `doe_power` - **no data file needed**
Power & runs for two-level factorial effects (no data)  [JMP DOE Evaluate Design]

```
No data. sigma (error SD), effect (list; difference between high and low means), runs (list), terms (model
parameters incl. intercept), alpha.
```

### `taguchi_analyze`
Taguchi S/N response tables, ANOVA, predicted optimum  [JMP Taguchi]

```
factors (list of columns); responses (list of replicate columns) ; sn larger|smaller|nominal ; ANOVA of S/N.
```

### `mixture_analyze`
Scheffe mixture model, ternary contour, blend optimum  [JMP Mixture]

```
components (list); response; model linear|quadratic|special_cubic; goal max|min|target (+target).
```

### `doe_optimize`
Multi-response desirability optimisation + prediction profiler  [JMP Prediction Profiler]

```
factors (list of columns), responses {name: {goal: max|min|target, low, high, [target], [weight],
[importance]}}, degree (1 or 2; default 2), quadratic (bool).
```

## Graphs & Fit Y by X  (`ss_graph.py`)

### `graph`
Graph Builder: scatter/line/bar/box/violin/hist/density/heatmap/bubble/pairs/area/pie + fits, colour, facets  [JMP Graph Builder | SPSS Chart Builder]

```
type: scatter|line|bar|box|violin|hist|density|heatmap|bubble|pairs|area|pie.  x, y, color (group col), size
(bubble), by (facet col), fit (none|linear|quadratic|cubic|lowess), ci (bool),
agg (bar:
mean|median|sum|count), error (bar: sd|se|ci|none), bins, title, columns (pairs/heatmap), logx, logy.
```

### `auto_xy`
Fit Y by X: auto-picks ANOVA / regression / logistic / contingency  [JMP Fit Y by X]

```
Fit Y by X: y, x; picks one-way ANOVA (num~cat), regression (num~num), logistic/multinomial (cat~num),
contingency (cat~cat). Optional 'by' not supported.
```
