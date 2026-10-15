# Module: linear models & power  (`ss_linear.py`, own numpy/scipy OLS engine — no statsmodels needed)

| analysis | `--params` keys (defaults) | Outputs |
|---|---|---|
| `oneway` | `response`, `factor`, `posthoc` tukey\|dunnett (tukey), `control` (dunnett level) | Summary of fit, ANOVA, means ± CI, Levene/Brown-Forsythe, Welch ANOVA, Kruskal-Wallis, Tukey HSD (CI plot) or Dunnett, diagnostics |
| `anova` | `response`, `factors` [..], `covariates` [..], `degree` (2 if >1 variable else 1) | ANOVA, Type III effect tests, parameter estimates (effect-coded), LS means, main-effect & interaction plots, diagnostics |
| `regression` | `response`, `predictors` [numeric], `factors` [..], `degree` (1), `poly` (1; >1 adds centered polynomial terms), `stepwise` (false; backward, hierarchy-safe), `alpha` (.05) | Fit summary, ANOVA, estimates + CI, effect tests, VIF, fit line + CI (1 predictor), 6-panel diagnostics, outlier note |
| `glm` | `response`, `predictors`, `factors`, `family` binomial\|poisson, `positive` (level = 1) | IRLS estimates, odds/rate ratios + CI, LR χ², McFadden R², AIC/BIC, ROC/AUC, confusion matrix |
| `power` (no data) | `test` ttest2\|ttest1\|anova, `effect_size` (d, or Cohen f), `alpha`, `power`, `k`, `n` | Required n per group, power curve |

Diagnostics panel: actual-by-predicted, residual-by-predicted, studentized-residual Q-Q, histogram, residual order, Cook's D.

Interpretation checklist
- ANOVA assumptions: residuals ≈ normal (Q-Q), equal variances (Levene) — if violated, quote Welch / Kruskal; independence (residual-by-order shows trends).
- Unbalanced multi-way data: Type III is reported (same default as JMP / SPSS GLM). Interaction significant ⇒ interpret simple effects, not main effects alone.
- Regression: R² alone proves nothing; check residual patterns, VIF > 10 (collinearity), |studentized residual| > 3, Cook's D. Stepwise selection overfits — validate on new data.
- Extrapolating outside the data range is not supported by the fit CI.
- Logistic: needs ≥ ~10 events per predictor; check AUC and the confusion matrix at 0.5 (threshold is a business choice).
- Power: effect_size is the *smallest effect worth detecting* — must come from the user/engineering judgement, not from pilot noise.

Example
```
python scripts/run.py oneway --data d.csv --params '{"response":"Strength","factor":"Supplier"}' --out out
python scripts/run.py anova --data d.csv --params '{"response":"Yield","factors":["Temp","Catalyst"],"covariates":["Pressure"]}' --out out --lang he
```
