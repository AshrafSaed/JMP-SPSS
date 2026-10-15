# Module: reliability & survival  (`ss_rel.py`)

## `life_distribution`
Params: `time` (> 0), `event` (1 = failure, 0 = right-censored; omit if all failed), `distributions` (default weibull, lognormal, exponential), `at_times` (list → R(t)), `unit` (label).
- MLE with right censoring (Nelder-Mead; Hessian-based Wald 95 % CI on the log scale). Verified: recovers simulated Weibull β, η under 40 % censoring and equals scipy `weibull_min.fit` for uncensored data.
- Outputs: distribution comparison (log-likelihood, AICc, BIC), parameters + CI, MTTF, B1/B5/B10/median/B63.2, R(t); Weibull probability plot (Johnson adjusted ranks, Benard), survival curves vs Kaplan-Meier, hazard curves.
- β < 1 infant mortality, ≈ 1 random, > 1 wear-out (the report states it). Fewer than 10 failures ⇒ warning: asymptotic CIs are unreliable.
- Not covered: interval/left censoring, competing failure modes, accelerated-life models, Bayesian/likelihood-ratio intervals. Say so if needed.

## `survival`
Params: `time`, `event`, `group` (optional), `unit`.
- Kaplan-Meier table (Greenwood SE, log-log 95 % CI), step plot with CI bands and censor ticks, median survival, restricted mean, log-rank test for ≥ 2 groups (assumes proportional hazards — check that curves do not cross).

Examples
```
python scripts/run.py life_distribution --data life.csv --params '{"time":"Hours","event":"Failed","at_times":[500,1000],"unit":"h"}' --out out
python scripts/run.py survival --data life.csv --params '{"time":"Hours","event":"Failed","group":"Supplier"}' --out out --lang he
```
