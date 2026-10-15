# Modules: advanced DOE (`ss_doe2.py`) and graphs (`ss_graph.py`)

## DOE
| Need | analysis | Notes |
|---|---|---|
| Screen many factors, few runs | `doe_design2` `type`: `dsd` (Definitive Screening), `plackett_burman` | DSD: 3 levels, mains orthogonal to each other, to all quadratics and to all 2FIs (checked for k=4…12). Runs = 2·order+1 (order = smallest prime-based conference-matrix order ≥ k; extra columns dropped). PB = main effects only (Res III). |
| Constrained/odd run count | `type`: `d_optimal` (`model` main/interaction/quadratic, `runs`) | Fedorov exchange, 15 starts; reports D-/A-efficiency. |
| Orthogonal arrays | `type`: `taguchi` (`levels` 2/3, optional `array` L4/L8/L16/L32/L9/L27) | GF(p) construction; every column pair balanced. Interactions are confounded - main-effects only. |
| Formulations | `type`: `mixture` (`components`, `degree` or `centroid`, `lower`) | Simplex-lattice/centroid (+ overall centroid). |
| Blocking in two directions | `type`: `latin_square` | Randomized. |
| Add runs to an existing design | `doe_augment` (`foldover` all/one factor, `axial`, `center`) | Needs the `*_coded` columns of the original design CSV. Block column added. |
| Detectable effect / runs | `doe_power` (no data) | Noncentral-t power, effect = high−low mean; verified by Monte-Carlo. |
| Taguchi analysis | `taguchi_analyze` (`sn` larger/smaller/nominal) | S/N response table, ANOVA on S/N, additive optimum - confirm by a run. |
| Mixture analysis | `mixture_analyze` (`model` linear/quadratic/special_cubic) | Scheffé model, ternary contour (3 components), optimum blend within bounds. |
| Several responses, one setting | `doe_optimize` | Derringer-Suich desirability (`goal` max/min/target with `low`,`high`,`target`,`weight`,`importance`) + Prediction-Profiler grid. Model quality (R²) is reported: a weak model gives an unreliable optimum. |
Always randomise run order and confirm the predicted optimum with 3–5 confirmation runs. Classical designs (`doe_design`/`doe_analyze`) are in `doe.md`.

## Graphs
- `graph` (`type`: scatter, line, bar, box, violin, hist, density, heatmap, bubble, pairs, area, pie): `x`,`y`,`color`,`size`,`by` (facets),`fit` (linear/quadratic/cubic/lowess),`agg`,`error`. Bar error bars: sd/se/ci. Heatmap with no x/y = correlation matrix.
- `auto_xy` (JMP "Fit Y by X"): continuous Y × categorical X → one-way ANOVA; continuous × continuous → regression; binary Y → logistic; nominal Y (>2) → multinomial; categorical × categorical → contingency.
