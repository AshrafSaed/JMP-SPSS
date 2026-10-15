# ניתוח JMP + SPSS  |  JMP + SPSS Analysis (Claude Skill)

**עברית:** Skill ל-Claude שמריץ ניתוחים סטטיסטיים תעשייתיים בסגנון JMP ו-SPSS, בלי צורך באף אחת מהתוכנות. Claude שואל את הפרמטרים, מריץ את הניתוח, בודק הנחות, ומפיק דוח Word + גרפים PNG, בעברית או באנגלית לפי בחירה בכל ניתוח.

**English:** A Claude Skill that reproduces JMP- and SPSS-style industrial-statistics analyses in pure Python (no licences). It asks for parameters, runs the analysis, checks assumptions, and delivers a Word report + PNG graphs, in Hebrew or English chosen per analysis.

- 75 analyses in 12 areas, one master `SKILL.md` + module docs.
- Stack: numpy, scipy, pandas, matplotlib, python-docx, scikit-learn (`pyreadstat` only for `.sav`). No statsmodels.
- Engineering & Development: Ashraf Saed | MIGATRONIX – Automation & Control Solutions

---

## מה ה-Skill כולל / What it covers

| תחום / Area | ניתוחים / Analyses (`run.py` names) | מקביל ב-JMP / SPSS |
|---|---|---|
| תיאורי ורב-משתני / Descriptive | `distribution`, `tabulate`, `correlation`, `partial_corr`, `crosstab`, `pca`, `cluster` | Distribution, Multivariate, PCA, Cluster / Explore, Correlate, Crosstabs, Factor |
| השוואת ממוצעים ודרגות / Comparisons | `ttest`, `nonparametric` (Mann-Whitney, Wilcoxon, Kruskal+Dunn, Friedman, sign, McNemar, median, runs), `equivalence` (TOST), `variance_test`, `proportion`, `multiple_testing` | Fit Y by X, Equivalence / Compare Means, Nonparametric |
| משפחת ANOVA / ANOVA family | `oneway`, `anova`, `rm_anova` (Mauchly, GG/HF), `manova` (multi-factor, MANCOVA), `mixed` (REML + Satterthwaite df), `nested_anova`, `split_plot` | Fit Model, Variance Components / GLM, MIXED, VARCOMP |
| רגרסיה ו-GLM / Regression & GLM | `regression`, `glm`, `glmm`, `ordinal`, `multinomial`, `nonlinear`, `penalized` (LASSO/Ridge/ENet), `power` | Fit Model, Fit Curve, Generalized Regression / Linear, Logistic, PLUM, NOMREG, NLR |
| רב-משתני וחיזוי / Multivariate & ML | `factor_analysis` (KMO, Bartlett, varimax), `discriminant` (LDA/QDA), `predictive` (tree, forest, boosting), `pls` (VIP) | Factor, Discriminant, Partition, PLS |
| סדרות עתיות / Time series | `timeseries` (ACF/PACF, exact-ML SARIMA, auto-ARIMA, Holt-Winters), `stationarity` (ADF + KPSS) | Time Series / Forecasting |
| התפלגויות וחריגים / Distributions & outliers | `transform` (Box-Cox, Yeo-Johnson), `tolerance`, `outliers` (Grubbs, ESD), `bootstrap` | Distribution, Explore Outliers |
| הסכמה ואמינות מדידה / Agreement | `agreement` (kappa, Fleiss, attribute), `icc`, `cronbach`, `bland_altman` | Attribute Gauge / Reliability, Kappa |
| בקרת תהליכים / SPC | `control_chart`, `ewma`, `cusum`, `laney`, `run_chart`, `pareto`, `hotelling_t2`, `sampling_plan` | Control Chart Builder / Control Charts |
| יכולת ו-MSA / Capability & MSA | `capability` (Cp/Cpk/Pp/Ppk), `gage_rr`, `gage_rr_reml`, `gage_type1`, `gage_linearity_bias`, `gage_nested`, `variability_chart` | Process Capability, MSA |
| אמינות / Reliability | `life_distribution`, `life_interval`, `survival` (Kaplan-Meier), `cox_ph` (+ Grambsch-Therneau PH test), `repairable` (Crow-AMSAA, Nelson MCF), `alt` (Arrhenius/power), `degradation`, `reliability_plan` (Weibayes, success-run) | Life Distribution, Survival, Fit Life by X, Reliability Growth |
| תכנון ניסויים / DOE | `doe_design` (full/fractional/CCD/Box-Behnken), `doe_design2` (DSD, Plackett-Burman, D-optimal, Taguchi, mixture, Latin square, split-plot), `doe_augment`, `doe_power`, `doe_analyze`, `taguchi_analyze`, `mixture_analyze`, `doe_optimize` (desirability + profiler) | DOE, Prediction Profiler |
| גרפים / Graphs | `graph` (scatter, line, bar, box, violin, histogram, density, heatmap, bubble, pairs, area, pie), `auto_xy` | Graph Builder, Fit Y by X / Chart Builder |

הרשימה המלאה עם כל הפרמטרים של כל ניתוח: [`references/catalog.md`](references/catalog.md). מסמכי המודולים: `references/*.md`.

## איך זה עובד / Workflow

1. מתארים בעברית או באנגלית מה רוצים לנתח (Claude מנסח מחדש בשורה אחת ברורה לפני הביצוע).
2. `profile` קורא CSV / Excel / `.sav` ומזהה סוגי עמודות.
3. Claude שואל את הפרמטרים החסרים בשאלה אחת: עמודות, גבולות מפרט, רמת מובהקות, שפת הדוח.
4. הרצה, בדיקת הנחות (נורמליות, שונות, יציבות, איזון, צנזור), ודוח.
5. פלט: `.docx` + `.png` + `.csv` + `*_summary.json`.

```
python scripts/run.py list
python scripts/run.py profile --data data.xlsx
python scripts/run.py capability --data data.csv --params '{"column":"x","lsl":9.5,"usl":10.5}' --out ./out --lang he
```

## התקנה / Install

הניחו את התיקייה בנתיב ה-skills של Claude (למשל `~/.claude/skills/jmp-spss-analysis/`) או העלו אותה כ-Skill ב-Claude. דרישות: `pip install numpy scipy pandas matplotlib python-docx scikit-learn openpyxl` (ו-`pyreadstat` ל-`.sav`).

## אימות / Verification

שלוש חבילות בדיקה: `tests/run_tests.py` (59), `tests/run_tests2.py` (237) ו-`tests/run_tests3.py` (63) - סה"כ 359 בדיקות. הן משוות ל-scipy / sklearn, לצורות סגורות, לערכים מפורסמים (ICC של Shrout-Fleiss, Fleiss kappa, Grubbs, גדלי מדגם success-run), לסימולציה ולזהויות בין שיטות (Cox score = log-rank, REML = ANOVA, עוצמת DOE מול Monte-Carlo), ומריצות כל ניתוח בעברית ובאנגלית. כל הבדיקות עוברות.

```
python tests/run_tests.py out1
python tests/run_tests2.py out2      # ~6 דקות
python tests/run_tests3.py out3      # ~3 דקות
```

## מגבלות / Limitations

מה שתוקן לאחרונה (פירוט ואימות ב-[`references/limits-fixes.md`](references/limits-fixes.md)): בדיקות F/t במודל מעורב עם df של Satterthwaite; ARIMA/SARIMA בנראות מקסימלית מדויקת; ADF ו-KPSS; בדיקת PH של Grambsch-Therneau; MANOVA רב-גורמי ו-MANCOVA; GLMM עם חותך אקראי; Gage R&R ב-REML (נתונים לא מאוזנים ומשתני עזר); ניתוח ותכנון Split-plot; קידוד reference בסגנון SPSS (`"coding": "reference"`).

מה שנשאר, בכנות:
- קבצי `.jmp` לא נקראים ישירות (פורמט סגור): יש לייצא ל-Excel/CSV. `.sav` דורש `pip install pyreadstat` במחשב שלך.
- Satterthwaite ולא Kenward-Roger: עם מעט רמות של אפקט אקראי, ערכי p עלולים להיות אופטימיים מעט.
- GLMM: חותך אקראי יחיד (binomial / Poisson), בלי שיפועים אקראיים או אפקטים חוצים.
- Split-plot: תכנון 2-רמות מלא (whole-plot x sub-plot) בלבד; אין תכנוני split-split-plot או שברים.
- לא כלולים: סקרים מורכבים, MANOVA עם גורמים אקראיים, ARIMA עם רגרסורים חיצוניים, סיכונים מתחרים ו-frailty.
- הבדלים קטנים מול JMP/SPSS אפשריים (למשל p של ADF הוא Monte-Carlo, לא משטח MacKinnon).
- ספי הערכה (Cpk ≥ 1.33, %GRR, ndc ≥ 5 וכו') הם מוסכמות ולא תקנים, ולכן נשארו כהנחיה בלבד: אין להסיק מהפלט עמידה ב-ISO/AIAG; מפרט הלקוח קובע.
