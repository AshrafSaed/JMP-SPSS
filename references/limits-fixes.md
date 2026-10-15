# Limitation fixes: what changed, how it was verified, what remains

All verifications are in `tests/run_tests3.py` (63 checks). Each compares with an independent computation, a closed form, or simulation truth.

| Former limitation | Now | Parameters | Verified against |
|---|---|---|---|
| Mixed-model tests were large-sample Wald | `mixed` (and `split_plot`) use Satterthwaite denominator df for t and F tests (multi-df terms: Fai-Cornelius approximation). `df_method: "wald"` restores the old behaviour. | `df_method` | balanced one-way: intercept df = a-1 exactly; classical RCBD F and df = 18; classical split-plot whole-plot F and df |
| ARIMA by conditional least squares only | Exact Gaussian ML by Kalman filter, incl. seasonal AR/MA (P, Q), standard errors, AICc. CSS still available. | `timeseries`: `method` ml (default) / css, `P`, `Q`, `D`, `period` | Kalman likelihood = multivariate-normal Toeplitz likelihood (5 models) and AR(1) closed form; parameter recovery incl. airline model |
| No ADF / KPSS | `stationarity`: ADF (AIC lag selection, Monte-Carlo p and critical values for the actual n/lags) + KPSS (published critical values) + combined verdict | `column`, `trend` c/ct, `lags`, `kpss_lags`, `difference`, `mc_reps` | MC critical values match published -3.43/-2.86/-2.57 and -3.96/-3.41/-3.12; size ~5 %; power vs AR(0.5); KPSS size ~5 % |
| Approximate PH check in Cox | Grambsch-Therneau score test (per covariate + global) on scaled Schoenfeld residuals, plots | `cox_ph`: `ph_time` km/rank/log/identity | size ~5 %, uniform p under H0, power ~100 % for crossing hazards |
| MANOVA one factor only | multi-factor, interactions and covariates (MANCOVA), Type III multivariate tests per effect + univariate Type III tables | `manova`: `factors`, `covariates`, `degree` | hand-computed balanced-design Wilks; equals one-way MANOVA for one factor |
| No GLMM | Random-intercept GLMM, binomial/Poisson, adaptive Gauss-Hermite (nAGQ 15; 1 = Laplace), LR test for sigma_u | `glmm`: `response`, `family`, `fixed`, `group`, `trials`, `nagq` | marginal likelihood = scipy.integrate.quad per group; recovers truth; sigma->0 equals sklearn logistic |
| GRR needed balanced data, no covariates | `gage_rr_reml`: unbalanced data, fixed covariates | `part`, `operator`, `measurement`, `covariates`, `tolerance` | equals ANOVA Gage R&R on balanced data (interaction kept); recovers covariate effect |
| No split-plot | Design (`doe_design2` type `split_plot`) and analysis (`split_plot`) with error strata | `whole_plot_factors`, `subplot_factors`, `whole_plot_reps`; analysis: `whole_plot`, `subplot`, `plot`, `block` | structure checks; analysis equals classical split-plot ANOVA |
| Effect vs reference coding differences | `"coding": "reference"` gives SPSS-style parameter estimates (last level = reference) in `anova`, `regression`, `glm`; tests never change | `coding` | equals dummy-variable least squares / sklearn |

## Cannot be fixed here (and why)
- **`.jmp` files**: proprietary binary format with no public specification; export to Excel/CSV from JMP.
- **`.sav` in this sandbox**: needs `pyreadstat`, which cannot be installed without PyPI access; on your own machine `pip install pyreadstat` is enough.
- **Thresholds (Cpk 1.33, %GRR, ndc)**: they are industry conventions, not standards or defects. They stay guidance; the customer specification decides.
- **Kenward-Roger**: only Satterthwaite is implemented.
- **Complex-survey analysis, random-slope/crossed GLMM, fractional/multi-stratum split-plot designs, ARIMA with regressors**: not implemented.
