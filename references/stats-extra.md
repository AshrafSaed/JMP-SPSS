# Module: classical tests, agreement, scale reliability, transforms  (`ss_stats2.py`)
Exact `--params` keys: `catalog.md`. Notes below = when to use, assumptions, how to read.

| Need | analysis | Key points |
|---|---|---|
| Rank-based comparison | `nonparametric` (`mode`: mannwhitney, wilcoxon, kruskal, friedman, sign, mcnemar, median, runs) | Kruskal adds Dunn post-hoc (Holm). Friedman adds Kendall W + pairwise Wilcoxon (Holm). Report effect size (rank-biserial, epsilon², W), not only p. |
| One/two proportions | `proportion` | Gives Wilson AND exact Clopper-Pearson CI; two-sample: z, Fisher, RR, OR, Newcombe CI. Works from counts (`x`,`n`) without a data file. |
| Equal variances | `variance_test` | Levene/Brown-Forsythe are robust; Bartlett assumes normality. Use before choosing pooled vs Welch. |
| "Not different" claim | `equivalence` (TOST) | The margins (`low`,`high`) are an engineering decision - never default them. Non-significant t-test ≠ equivalence. |
| Same subjects, ≥3 conditions | `rm_anova` | Wide format (one row per subject). Check Mauchly; if sphericity violated use the Greenhouse-Geisser row. No between-subject factor (use `mixed` for that). |
| Several responses, one or more factors / covariates | `manova` (`factor` for one-way; `factors`, `covariates`, `degree` for multi-factor / MANCOVA with Type III tests per effect) | Wilks/Pillai/HL/Roy. Pillai is most robust to violations; Roy's p is an upper bound. Needs n ≫ number of responses. |
| Rater/appraiser agreement | `agreement` (`kappa`, `fleiss`, `attribute`) | Attribute mode = within appraiser, vs standard, between appraisers, miss/false-alarm. Kappa depends on prevalence; report % agreement too. |
| Continuous rater agreement | `icc`, `bland_altman` | Choose the ICC form that matches the design (1 = different raters per target, 2 = random raters absolute agreement, 3 = fixed raters consistency). Bland-Altman: judge limits of agreement against what is clinically/engineering acceptable. |
| Questionnaire/item reliability | `cronbach` | alpha ≥ 0.7 is a convention. High alpha ≠ unidimensional. |
| Suspect points | `outliers` | Grubbs (1 outlier), generalized ESD (several; assumes normality), Tukey/MAD rules, Mahalanobis (multivariate). Flag, investigate the cause, then decide - never delete automatically. |
| Interval covering a % of population | `tolerance` | Different from CI (mean) and prediction interval (next value). Normal assumption; distribution-free option gives the confidence of [min,max]. |
| Non-normal data | `transform` | Box-Cox needs positive data (auto shift is reported). Transform changes the scale of conclusions - back-transform for reporting. |
| Uncertainty without formulas | `bootstrap` | Percentile and BCa CI; `statistic` mean/median/std/var/trimmed_mean/corr/cpk. Seed fixed (1) for reproducibility. |
| Many p-values | `multiple_testing` | FWER (Bonferroni/Sidak/Holm/Hochberg) vs FDR (BH/BY). |
| Summary tables | `tabulate` | Any stats list, grouped; frequency tables for categorical columns. |
| Control for confounders | `partial_corr` | Linear partial and semi-partial correlation. |

Verified against: Shrout-Fleiss (1979) ICC example, Fleiss kappa (Wikipedia example 0.210), sklearn Cohen kappa (incl. weights), NIST Grubbs critical value, exact tolerance factors, Clopper-Pearson/Wilson published values, TOST by independent Welch calc, RM-ANOVA vs Type III OLS, MANOVA vs ANOVA/Hotelling T², partial correlation via precision matrix.
