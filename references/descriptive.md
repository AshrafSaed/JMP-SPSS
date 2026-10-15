# Module: descriptive, tests, multivariate  (`ss_desc.py`)

| analysis | `--params` keys (defaults) | Outputs |
|---|---|---|
| `distribution` | `columns` (all numeric), `by` (grouping column) | Moments, quantiles, Shapiro-Wilk / D'Agostino / Anderson-Darling, histogram + normal fit, box plot, normal Q-Q |
| `ttest` | `mode` one\|two\|paired (two); `column`; `group` (two-sample, exactly 2 levels); `column2` (paired); `mu0` (one); `alternative` two-sided\|less\|greater | Welch + pooled t, Levene, Mann-Whitney / Wilcoxon, Cohen's d, 95 % CI, plots |
| `correlation` | `columns`, `method` pearson\|spearman\|kendall | r and p matrices, heat map, scatterplot matrix (≤ 8 vars) |
| `crosstab` | `row`, `col` | Observed/expected, χ², Cramér's V, Fisher exact (2×2), stacked bars |
| `pca` | `columns`, `n_components`, `standardize` (true), `color_by` | Eigenvalues, % variance, loadings, scree, score plot with loading arrows |
| `cluster` | `columns`, `method` kmeans\|ward, `k` (auto by silhouette 2–8) | Silhouette by k, sizes, cluster means, PCA projection, dendrogram, `cluster_assignments.csv` |

Interpretation checklist
- Distribution: skew/kurtosis, outliers (box plot), normality p. With n > ~200 Shapiro rejects trivial departures — judge by the Q-Q plot too.
- t-test: if Levene p < .05 use the Welch row. Report effect size (d), not just p. Paired mode requires the same rows to be pairs.
- Correlation ≠ causation; Spearman if relationships are monotone but not linear or data has outliers.
- Crosstab: if expected counts < 5 the report warns; use Fisher (2×2).
- PCA: use standardized variables when units differ; keep components with eigenvalue > 1 or by cumulative % (scree elbow).
- Cluster: clusters are always produced — check silhouette (> 0.5 reasonable, < 0.25 weak structure).

Example
```
python scripts/run.py distribution --data d.xlsx --params '{"columns":["Weight"],"by":"Machine"}' --out out --lang he
python scripts/run.py ttest --data d.csv --params '{"mode":"two","column":"Cycle","group":"Shift"}' --out out
```
