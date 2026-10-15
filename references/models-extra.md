# Module: mixed, multivariate, categorical, nonlinear, ML, time series  (`ss_models2.py`)

| Need | analysis | Notes / caveats |
|---|---|---|
| Random effects, repeated structure, blocks | `mixed` | REML variance components + F/t tests for fixed effects with Satterthwaite denominator df (`df_method: "wald"` for large-sample Wald). `random`: `["Batch"]`, `["Batch/Sample"]` (nested → Batch and Batch:Sample), `["A:B"]`. Kenward-Roger's covariance adjustment is not applied (can be slightly optimistic with very few random-effect levels). Split-plot / blocked designs: use `split_plot`. Boundary (variance=0) is allowed. |
| Hierarchical variance (plant>batch>sample) | `nested_anova` | Sequential SS table (F vs next level) + REML components. Balanced design gives the classical EMS estimates (verified). |
| Latent factors | `factor_analysis` | Reports KMO, Bartlett, scree, rotated loadings, communalities. `method` pc (principal components) or ml; varimax uses multiple starts. Number of factors defaults to eigenvalue>1 (a rule of thumb; override with `n_factors`). |
| Classify into groups | `discriminant` | LDA/QDA with leave-one-out (or k-fold) CV - quote the CV accuracy, not resubstitution. |
| Ordered categories | `ordinal` | Proportional-odds logit (thresholds θ_j − xβ). The parallel-lines assumption is NOT tested; compare with `multinomial` if doubtful. |
| Unordered categories (>2) | `multinomial` | Reference class; Wald SE from the information matrix; OR = exp(coef). |
| Curve fitting | `nonlinear` | Built-ins: exponential, exp_decay, power, logistic3, logistic4 (dose-response), michaelis_menten, gompertz, weibull_growth, or `custom` expression with `names` and `p0`. SE/band are asymptotic (Jacobian). Try several `p0` - local minima are common. |
| Many predictors / collinearity | `penalized` | Ridge/Lasso/Elastic-net; CV-chosen λ; coefficient path. Predictors are standardized internally, coefficients reported on the original scale. |
| Non-linear / interaction-heavy prediction | `predictive` | tree / random forest / gradient boosting; **cross-validated** performance and permutation importance. Fixed seed. Trees are for exploration; an unvalidated tree overfits. |
| Collinear predictors, few runs, several responses | `pls` | Components chosen by CV Q² (add while Q² improves >0.01; override `n_components`); VIP>1 = important by convention. |
| Forecasting | `timeseries` (`mode` explore, decompose, smooth, arima, auto_arima) | smooth = Holt-Winters (additive/multiplicative; residual-bootstrap intervals). ARIMA/SARIMA by exact Gaussian ML (Kalman filter; `P`, `Q`, `D`, `period` for seasonal terms; `method: "css"` for conditional least squares), Gaussian intervals; auto_arima searches p,q ≤ 3 by AICc at a given `d`. Decide `d` with `stationarity` (ADF + KPSS, see below). |

Verified: REML = ANOVA estimates (balanced one-way and 3-level nested), mixed fixed effects = OLS in balanced RCBD, ordinal(2 levels) = sklearn logistic, ordinal/multinomial parameter recovery within 3 SE, AR(1) recovery, random-walk forecast & interval, Holt-Winters on a noiseless seasonal trend, varimax simple structure (40 random seeds), nonlinear recovery, LASSO support recovery, PLS Q².

## Added later (see limits-fixes.md for verification)
| Need | analysis | Notes |
|---|---|---|
| Is the series stationary? How many differences? | `stationarity` | ADF (H0 unit root; AIC-selected lags; Monte-Carlo p-value/critical values for the actual n) and KPSS (H0 stationary; published critical values). Verdict combines both; `difference` tests the differenced series. |
| Binary / count data with repeated groups | `glmm` | Random-intercept GLMM (binomial logit or Poisson log), adaptive Gauss-Hermite (`nagq`, default 15; 1 = Laplace). Fixed effects as OR/RR with Wald CI, sigma_u, latent ICC, LR test vs no random effect. One random intercept only. |
| Split-plot / hard-to-change factors | `split_plot` | REML with random whole plot; effects table shows which error stratum each test used (Satterthwaite df). Generate the design with `doe_design2` `type: split_plot`. |
