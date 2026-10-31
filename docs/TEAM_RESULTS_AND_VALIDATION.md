# Reading the final team presentation correctly

**Project:** *Predicting ATP Tennis Match Outcomes Using Machine Learning* — Group 6 (Hugo, Quentin, Raphaël, Matthieu and Roslane), final academic presentation, **26 June 2026**.

## What was reported

- **Primary comparison (slide 11):** hybrid XGBoost **67.75% accuracy** vs bookmaker favorite **67.94%**; bookmaker remains slightly ahead on that original cohort. The matched original test contains **9,798** matches, with the chronological test period beginning **2021-08-27** (slide 8).
- **Additional player attributes (slide 14):** a Jeff Sackmann-associated player dataset of **1,482** players / **64** columns was joined to historical match data, yielding a **35,438-match** enriched study sample through **2023**. Data matching and point-in-time availability matter.
- **Fatigue extension (slide 15):** adding recent workload did **not** improve the outcome (**67.82%**), compared with a standard odds-inclusive enriched model **68.07%** and the **67.89%** bookmaker baseline in the enriched comparison.
- **Exploratory feature optimization (slides 16–17):** reported best combination of **Age_Diff, Height_Diff, HighRank_Diff, hand_P1_is_L** added to the core + odds setup: **68.468% accuracy, F1 67.71%, ROC-AUC 0.7498, log loss 0.5848**. The period-matched bookmaker winner baseline was **67.89%**. The reported **+0.578 percentage-point** gap is narrow.

## Why the headline requires a qualification

The last cell of the archived team notebook iterates through candidate combinations and evaluates each one on the same 20%-holdout (`X_test`), then picks the highest-scoring candidate. **Selecting on the test set turns that set into a model-selection set**. The best reported 68.468% is a historical exploratory measurement, **not an untouched out-of-sample test after feature selection**. A proper future evaluation should reserve a new post-selection test period and audit point-in-time availability of the player-profile features.

Because the stronger model **uses bookmaker-implied probabilities**, beating the bookmaker's winner-choice *accuracy* by a small margin is also **not the same thing as proving superior probability estimates or a profitable betting strategy**. For the primary benchmark, the bookmaker's ROC-AUC and log loss were marginally better than the hybrid's (slide 11).

## What was improved for code review

The repo's `src/atp_ml/` demonstrates a date-aware feature builder, chronological **train / validation / test** split, and behavioral tests that do not need the historical dataset. It deliberately leaves the enriched player-profile features out rather than imply clean real-time availability. **Its output is not the result that produced the published team slides**.

## Source attribution

Results summarized from the team's original 26 June 2026 presentation (slides 8, 11, 14–20) and the team's archival notebook. This project and its historical results are collaborative; the newer public code refactor is a separate engineering demonstration.
