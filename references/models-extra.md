# Module: mixed, multivariate, categorical, nonlinear, ML, time series  (`ss_models2.py`)

| Need | analysis | Notes / caveats |
|---|---|---|
| Random effects, repeated structure, blocks | `mixed` | REML variance components + Wald tests for fixed effects. `random`: `["Batch"]`, `["Batch/Sample"]` (nested → Batch and Batch:Sample), `["A:B"]`. Fixed-effect p-values are large-sample Wald (JMP/SPSS use Satterthwaite/Kenward-Roger df → can be less significant in small designs). Boundary (variance=0) is allowed. |
| Hierarchical variance (plant>batch>sample) | `nested_anova` | Sequential SS table (F vs next level) + REML components. Balanced design gives the classical EMS estimates (verified). |
| Latent factors | `factor_analysis` | Reports KMO, Bartlett, scree, rotated loadings, communalities. `method` pc (principal components) or ml; varimax uses multiple starts. Number of factors defaults to eigenvalue>1 (a rule of thumb; override with `n_factors`). |
| Classify into groups | `discriminant` | LDA/QDA with leave-one-out (or k-fold) CV - quote the CV accuracy, not resubstitution. |
| Ordered categories | `ordinal` | Proportional-odds logit (thresholds θ_j − xβ). The parallel-lines assumption is NOT tested; compare with `multinomial` if doubtful. |
| Unordered categories (>2) | `multinomial` | Reference class; Wald SE from the information matrix; OR = exp(coef). |
| Curve fitting | `nonlinear` | Built-ins: exponential, exp_decay, power, logistic3, logistic4 (dose-response), michaelis_menten, gompertz, weibull_growth, or `custom` expression with `names` and `p0`. SE/band are asymptotic (Jacobian). Try several `p0` - local minima are common. |
| Many predictors / collinearity | `penalized` | Ridge/Lasso/Elastic-net; CV-chosen λ; coefficient path. Predictors are standardized internally, coefficients reported on the original scale. |
| Non-linear / interaction-heavy prediction | `predictive` | tree / random forest / gradient boosting; **cross-validated** performance and permutation importance. Fixed seed. Trees are for exploration; an unvalidated tree overfits. |
| Collinear predictors, few runs, several responses | `pls` | Components chosen by CV Q² (add while Q² improves >0.01; override `n_components`); VIP>1 = important by convention. |
| Forecasting | `timeseries` (`mode` explore, decompose, smooth, arima, auto_arima) | smooth = Holt-Winters (additive/multiplicative; residual-bootstrap intervals). ARIMA by conditional least squares (not exact ML), Gaussian intervals; auto_arima searches p,q ≤ 3 by AICc at a given `d`. No ADF/KPSS unit-root test - decide `d` from plots/ACF. |

Verified: REML = ANOVA estimates (balanced one-way and 3-level nested), mixed fixed effects = OLS in balanced RCBD, ordinal(2 levels) = sklearn logistic, ordinal/multinomial parameter recovery within 3 SE, AR(1) recovery, random-walk forecast & interval, Holt-Winters on a noiseless seasonal trend, varimax simple structure (40 random seeds), nonlinear recovery, LASSO support recovery, PLS Q².
