# Modules: advanced SPC/MSA (`ss_spc2.py`) and advanced reliability (`ss_rel2.py`)

## SPC / MSA
| Need | analysis | Notes |
|---|---|---|
| Detect small sustained shifts | `ewma` (λ 0.2, L 3), `cusum` (k 0.5, h 5) | Give `target` and `sigma` from an in-control reference period when the data may already contain the shift. CUSUM reports the first signal and the estimated shift start/new mean. |
| Attribute data with extra-binomial variation | `laney` (`type` p/u) | σ_z ≈ 1 → classical chart is fine; σ_z ≫ 1 → classical limits cause false alarms. |
| Prioritise causes | `pareto` | 80 % line; `top` groups the rest as "Other". |
| Trends / clustering test | `run_chart` | Runs about median and runs up/down. |
| Several correlated characteristics | `hotelling_t2` | Phase I (Beta limit) or Phase II (`phase`:2, `ref_rows`). Contributors are simple standardized deviations. |
| Gauge capability on one reference part | `gage_type1` | Cg/Cgk with K=20 %, 6s (VDA 5 / Minitab convention - not a universal standard). |
| Gauge accuracy across range | `gage_linearity_bias` | Regression of bias on reference; give `process_variation` for % values. |
| Destructive testing | `gage_nested` | Part nested in operator; balanced → ANOVA, unbalanced → REML. |
| Variation sources overview | `variability_chart` | Multi-vari picture + nested REML components. |
| Lot acceptance | `sampling_plan` | Binomial OC curve; searches the smallest (n,c) meeting AQL/α and LTPD/β. For contracts follow ISO 2859-1 / ANSI Z1.4 tables. |

## Reliability
| Need | analysis | Notes |
|---|---|---|
| Effect of covariates on time-to-event | `cox_ph` | Efron (default) or Breslow ties; HR with CI; LR/Wald/score; concordance; Grambsch-Therneau PH test (per covariate and global, scaled Schoenfeld residuals vs g(t), plots; `ph_time` km/rank/log/identity). |
| Repairable systems | `repairable` | Pooled power-law NHPP (Crow-AMSAA) MLE, Laplace trend test, Nelson MCF. β<1 improving, β>1 deteriorating. Assumes identical systems from age 0. |
| Accelerated testing | `alt` | Weibull or lognormal with Arrhenius (stress in °C → Ea in eV), inverse power law, or linear relationship; common shape. Predicts B-life and acceleration factors at `use_stress`. Extrapolation is a model assumption - check per-stress fits first. |
| Wear-out measurements | `degradation` | Per-unit linear/exponential path → pseudo failure time at `threshold` → life distribution (non-crossing units censored). Two-stage approximation. |
| Inspection-interval data | `life_interval` | Interval / left / right censoring MLE (Weibull, lognormal, exponential). |
| How many units to test | `reliability_plan` (no data) | Success-run (binomial) n for R/C and allowed failures; Weibayes test duration for a known β. |

Verified: Cox score test = log-rank (no ties), Cox analytic gradient/Hessian = numeric derivatives (with ties), Crow-AMSAA hand calc and NHPP simulation, ALT recovery of Ea/shape, exact degradation extrapolation, interval-MLE = exact-data Weibull MLE, success-run n = 22/59/299/38 (known values), Cg/Cgk formulas, EWMA(λ=1)=Shewhart, CUSUM hand case, T² Phase-I limit, Laney σ_z, nested GRR = REML, minimal sampling plan by brute force.
