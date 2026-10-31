# Code review, changes and methodological limitations

This repository has two layers, and they must not be confused:

1. `notebooks/atp_match_prediction.ipynb` preserves the team's **original exploratory notebook** (cell outputs removed). It includes Colab-specific upload cells, additional datasets and exploratory features. It is **not guaranteed to execute from a clean clone** and is not presented as production code.
2. `src/atp_ml/` contains a **small, tested, refactored demonstration** of the study's core ideas: chronological Elo/form/H2H, bookmaker probability normalization, date-based splits, baselines and held-out evaluation. This was adapted for public code review. It is **not the code that produced the historical slides**.

## Updated interpretation of the **final** team presentation (26 June 2026)

The previous portfolio README only highlighted the **initial** benchmark in the final team presentation, in which XGBoost + odds scored **67.75%** and the bookmaker favorite **67.94%**. The same presentation also reports an **enriched feature-extension result** using player profiles (Jeff Sackmann data), where an explored feature combination scored **68.468%** vs the **same-period** bookmaker baseline of **67.89%**.

This is a genuine, documented *reported* result of the team's study and should be included. However, in the original notebook's final feature-search cell, each candidate feature combination is evaluated on `X_test`, and the winning combination is **selected on that same set**. Thus the winning test accuracy is subject to **selection optimism**. The archived notebook also integrates additional player-level data for which historical point-in-time availability has not been established here. The clean `src/atp_ml/` demonstrator deliberately avoids these risks and uses a separate validation split, but **does not independently reproduce the enriched team score**.

The result is best worded as **"exploratory enriched model achieved a reported 68.47% accuracy, slightly above the 67.89% same-period bookmaker baseline; independent re-validation remains necessary."** No economic/ROI or generalized bookmaker-beating claim is warranted.

## What was good in the original code

- Dynamic features were emitted before outcomes updated the Elo/form/H2H state.
- Predictions used historical player rankings, points, and surface information.
- Evaluated logistic regression, random forests, XGBoost and betting-market baselines.
- Used chronological rather than randomly shuffled train/test splitting.
- Evaluated probability quality through ROC-AUC, log loss and calibration plots, not only accuracy.

## Risks identified during the audit

- **Feature selection on the test set (original notebook, final cell):** the exhaustive combination search ranks candidates by accuracy and AUC on the same set later used as the final evaluation. The winning score is *optimistically biased* and should not be described as an untouched generalization test. The clean demo instead selects a model on a **validation period** and reports a separate held-out test period.
- **Matches on the same date:** the dataset provides dates but not precise match start times. Chronological row iteration could update a player's rating from one match before another match on the same calendar day. The clean feature builder takes **all same-date snapshots before any same-date results** are incorporated. This is conservative and prevents an accidental within-date look-ahead; it sacrifices some valid intraday information.
- **External player statistics:** season aggregates from the additional `players.csv` may contain post-match or end-of-season information. To avoid temporal leakage, verify each statistic's **as-of date** before including it in a pre-match forecast. The clean demonstrator deliberately excludes the auxiliary player-level enrichment.
- **Model selection and calibration:** model probabilities are not automatically calibrated. The original notebook shows reliability plots; future work should include a dedicated calibration split or out-of-fold calibration, never calibrating on the final test period.
- **Original environment:** the notebook requires Colab and CSV files not included in the public repo. External data redistribution rights were not verified.
- **Team attribution:** the source notebook is group work. The extracted module is a documentation/refactor layer for this portfolio, not evidence of sole authorship of every original algorithm.

## What a recruiter can review

- `src/atp_ml/features.py`: pre-event snapshots, state updates, date batching, robust odds handling.
- `src/atp_ml/train.py`: train/validation/test boundaries, train-only preprocessing, model selection and bookmaker benchmarking.
- `tests/test_features.py`: synthetic behavior tests that do not need the original CSV.
- `notebooks/atp_match_prediction.ipynb`: original team's experimental exploration and discussion (for historical context).

## Scientific honesty

The published team slide numbers are **reported original results** and are **not being attributed to the new demonstration pipeline**. Do not compare the two as if they used identical splits, feature selection and configurations. The source CSV is historical sports data, but future information availability must be verified before interpreting such results as real-world live forecasting performance.

## Verification performed for this release

Five behavior-focused unit tests passed and the refactored end-to-end pipeline executed successfully on the user-supplied historical match CSV. The raw dataset and newly measured results are not distributed here; the historical team metrics in the README are sourced to the original presentation.
