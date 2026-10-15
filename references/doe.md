# Module: Design of Experiments  (`ss_doe.py`)

## `doe_design`  (no data file)
Params: `type` full\|fractional\|ccd\|boxbehnken (full); `factors` {name: [low, high]}; `generators` e.g. ["D=ABC"] (fractional; letters = factor order A,B,C…); `center_points`; `replicates`; `randomize` (true); `seed`; `alpha` face\|rotatable (ccd).
- Writes `design_<type>.csv` (Run, Std order, coded and actual settings, empty `Response` column) and a Word report. Fractional designs list the aliases of main effects/2FI and the resolution (III / IV / V…).
- Run in the randomized order; replicate/center points give a pure-error estimate and a curvature test.

## `doe_analyze`
Params: `response`, `factors` [columns holding actual or coded values — auto-coded to −1…+1 from min/max], `degree` (2 = main + 2-way interactions), `quadratic` (false; true for CCD/Box-Behnken), `goal` max\|min\|target, `target`.
- Coded model; Effect = 2 × coefficient. With replicates/df: ANOVA, Type III tests, t-based Pareto. Saturated/unreplicated: Lenth PSE with ME and SME (significance by Lenth, not p-values — stated in the report).
- Plots: Pareto, half-normal, main effects, interaction plots, contour (2 factors or quadratic), diagnostics.
- Curvature test appears when center points exist (2-level designs).
- `goal`: optimum found inside the tested region (multi-start L-BFGS-B) and returned in actual units with the predicted response. It is a model prediction — run 3–5 confirmation runs before adopting it.
- Interpretation: significant interactions first; aliased effects in fractional designs cannot be separated without follow-up runs (fold-over).

Examples
```
python scripts/run.py doe_design --params '{"type":"fractional","factors":{"A":[0,1],"B":[0,1],"C":[0,1],"D":[0,1]},"generators":["D=ABC"],"center_points":3}' --out out
python scripts/run.py doe_analyze --data results.csv --params '{"response":"Yield","factors":["Temp","Press","Time"],"goal":"max"}' --out out --lang he
```
