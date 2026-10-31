# External data (not included)

The source team's raw archive included `atp_tennis.csv` (match-level ATP results and bookmaker odds) and `players.csv` (player-level aggregated statistics). We do **not** redistribute these files because the exact source, version and license must be checked.

## Expected columns for the tested demonstrator

`Date`, `Player_1`, `Player_2`, `Winner`, `Surface`, `Series`, `Court`, `Best of`, `Rank_1`, `Rank_2`, `Pts_1`, `Pts_2`, `Odd_1`, `Odd_2`.

- `Date`: YYYY-MM-DD or pandas-readable calendar date.
- `Winner`: must exactly equal `Player_1` or `Player_2`.
- Missing ranking/points values may use `-1` and are converted to missing values.
- Decimal odds <= 1 are discarded before constructing normalized market probabilities.

## Provenance/temporal caveat

Some player-level features in the exploratory notebook reference third-party player summaries and ATP records. **Season aggregates must be timestamped as of the match date to be valid predictors.** The extracted pipeline excludes the auxiliary `players.csv` rather than guessing the availability of those features.

The historical notebook's reported performance is not guaranteed to replicate with a newer/different CSV snapshot; it uses a different experiment-selection protocol.
