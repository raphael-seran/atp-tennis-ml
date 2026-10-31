"""Leakage-aware, chronological tennis feature construction.

Refactored from the project's original 2026 exploratory notebook.
It deliberately does not expose same-match outcomes to its predictors.
Matches sharing a calendar date are processed as a batch: no match played
on that date updates another match's pre-match features.
"""
from collections import defaultdict, deque
import numpy as np
import pandas as pd

REQUIRED = {"Date", "Player_1", "Player_2", "Surface", "Winner"}


def pre_match_features(matches: pd.DataFrame, k_factor: float = 32., form_window: int = 10) -> pd.DataFrame:
    """Return historical Elo, surface Elo, recent form and head-to-head features.

    For a dated batch, features are computed from *prior dates only*. Outcomes
    are used exclusively to update state after feature snapshots are recorded.

    Initial Elo = 1500, missing form = NaN, and H2H counts start at zero.
    This is a defensible conservative proxy for match-time ordering when only
    a calendar date (and no match start time) is available in the source CSV.
    """
    missing = REQUIRED - set(matches.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if form_window < 1 or k_factor <= 0:
        raise ValueError("form_window and k_factor must both be positive")
    df = matches.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="raise")
    df = df.sort_values("Date", kind="stable").reset_index(drop=True)
    df["Target"] = (df["Winner"] == df["Player_1"]).astype(int)
    invalid = ~(df["Winner"].eq(df["Player_1"]) | df["Winner"].eq(df["Player_2"]))
    if invalid.any():
        raise ValueError(f"Winner is not one of the two players in {int(invalid.sum())} rows")

    elo = defaultdict(lambda: 1500.)
    surface_elo = defaultdict(lambda: 1500.)
    recent = defaultdict(lambda: deque(maxlen=form_window))
    h2h = defaultdict(int)
    history = {}

    for day, batch in df.groupby("Date", sort=True):
        # Take all snapshots for the date BEFORE updating any result from it.
        for idx, row in batch.iterrows():
            a, b, surface = row["Player_1"], row["Player_2"], row["Surface"]
            history[idx] = {
                "Elo_1": elo[a], "Elo_2": elo[b],
                "Elo_Surf_1": surface_elo[(a, surface)],
                "Elo_Surf_2": surface_elo[(b, surface)],
                "Form_1": np.mean(recent[a]) if recent[a] else np.nan,
                "Form_2": np.mean(recent[b]) if recent[b] else np.nan,
                "H2H_1": h2h[(a, b)], "H2H_2": h2h[(b, a)],
            }
        for _, row in batch.iterrows():
            a, b, surface, target = row["Player_1"], row["Player_2"], row["Surface"], int(row["Target"])
            p = 1 / (1 + 10**((elo[b]-elo[a])/400))
            elo[a] += k_factor*(target-p)
            elo[b] += k_factor*((1-target)-(1-p))
            p_surf = 1 / (1 + 10**((surface_elo[(b,surface)]-surface_elo[(a,surface)])/400))
            surface_elo[(a,surface)] += k_factor*(target-p_surf)
            surface_elo[(b,surface)] += k_factor*((1-target)-(1-p_surf))
            recent[a].append(target)
            recent[b].append(1-target)
            h2h[(a,b) if target == 1 else (b,a)] += 1

    features = pd.DataFrame.from_dict(history,orient="index").sort_index()
    df = pd.concat([df, features], axis=1)
    for left,right,diff in [
        ("Elo_1", "Elo_2", "Elo_Diff"),
        ("Elo_Surf_1", "Elo_Surf_2", "Elo_Surf_Diff"),
        ("Form_1", "Form_2", "Form_Diff"),
        ("H2H_1", "H2H_2", "H2H_Diff"),
    ]:
        df[diff] = df[left]-df[right]
    return df


def with_market_probabilities(df: pd.DataFrame) -> pd.DataFrame:
    """Use market odds with overround normalization; invalid odds become NaN."""
    out = df.copy()
    for col in ("Odd_1", "Odd_2"):
        out[col] = pd.to_numeric(out[col], errors="coerce")
        out.loc[out[col] <= 1., col] = np.nan
    inv_1, inv_2 = 1/out["Odd_1"], 1/out["Odd_2"]
    denom=inv_1+inv_2
    out["Prob_1"] = inv_1/denom
    out["Prob_2"] = inv_2/denom
    return out
