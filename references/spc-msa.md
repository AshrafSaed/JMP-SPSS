# Module: SPC, capability, Gage R&R  (`ss_spc.py`)

## `control_chart`
Params: `type` imr\|xbar_r\|xbar_s\|p\|np\|c\|u (imr), `column`; subgroups: `subgroup` (column with subgroup id) or `subgroup_size`; attribute charts: `n` (column of sample sizes) or `sample_size`; `rules` (default [1,2,3,5,6]).
- Limits: I-MR σ = MRbar/1.128, MR UCL = 3.267·MRbar; Xbar-R / Xbar-S from d2, D3/D4, c4, B3/B4 (computed, checked against published n=5 values); p/np/u variable limits per sample size.
- Rules (Nelson/WE): 1 = point > 3σ, 2 = 9 same side, 3 = 6 trending, 4 = 14 alternating, 5 = 2 of 3 > 2σ, 6 = 4 of 5 > 1σ. Flagged points are circled with rule numbers and listed in a table.
- Limits are estimated from the supplied data. Use stable Phase-I data to set limits; do not claim a future process is "in control" from a chart that includes the very shifts you want to detect.

## `capability`
Params: `column`, `LSL` and/or `USL` (user-supplied), `target`, `subgroup` or `subgroup_size` (default 1 ⇒ within σ = MRbar/1.128), `distribution` normal\|lognormal\|weibull.
- Outputs: Cp, CPU, CPL, Cpk (within σ); Pp, PPU, PPL, Ppk (overall σ); Cpm (if target); 95 % CI; observed and expected PPM; Shapiro-Wilk; histogram with within/overall curves and spec lines; Q-Q.
- If normality is rejected the report says so; re-run with `lognormal`/`weibull` (percentile method: 0.135 % / 50 % / 99.865 %) or transform.
- Cp ≫ Cpk ⇒ off-centre. Ppk ≪ Cpk ⇒ instability between subgroups.
- Convention only: Cpk ≥ 1.33 (≥ 1.67 critical characteristics). The customer requirement governs.

## `gage_rr` (crossed, ANOVA method, AIAG MSA)
Params: `part`, `operator`, `measurement`, `tolerance` (= USL−LSL, optional but recommended), `study_sigma` (6), `alpha_int` (0.25).
- Needs balanced data: every part × operator measured the same number of times (≥ 2). Unbalanced ⇒ clear error.
- Interaction kept if p < alpha_int, otherwise pooled into repeatability. Negative variance components are set to 0.
- Outputs: ANOVA table, variance components (% contribution, σ, 6σ study var, % study var, % tolerance), ndc = ⌊1.41·σ_part/σ_GRR⌋, verdict, and a 6-panel report: components of variation, measurements by part/operator, part×operator interaction, box plots, range chart.
- Guideline: %GRR < 10 % acceptable, 10–30 % marginal (depends on application/cost), > 30 % unacceptable; ndc ≥ 5. The selected parts must span the real process range — otherwise %Study Var and ndc are misleading. Prefer %Tolerance when the gauge is used for pass/fail decisions.
- Related (see `spc-rel-extra.md`): `gage_nested` (destructive), `gage_type1`, `gage_linearity_bias`, `agreement` (attribute MSA). `gage_rr_reml` handles unbalanced data and fixed covariates. Not covered: expanded GRR with >2 random factors.

Examples
```
python scripts/run.py control_chart --data d.csv --params '{"type":"xbar_r","column":"Dia","subgroup":"Sample"}' --out out
python scripts/run.py capability --data d.csv --params '{"column":"Dia","LSL":9.9,"USL":10.1,"target":10}' --out out --lang he
python scripts/run.py gage_rr --data grr.csv --params '{"part":"Part","operator":"Op","measurement":"Value","tolerance":0.2}' --out out
```
