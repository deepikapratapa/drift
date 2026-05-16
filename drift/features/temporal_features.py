"""
drift/features/temporal_features.py

Computes time-based behavioral features per user.
Captures WHEN users are active — revealing patterns like
"Sunday night binge shoppers" or "payday spenders".

Features computed:
    - Preferred hour of day (cyclical encoded)
    - Preferred day of week (cyclical encoded)
    - Weekend activity ratio
    - Payday spike indicator (activity around 1st and 15th)
    - Session recency trend (are they becoming more or less active?)
    - Night owl score (% of sessions between 10pm–4am)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def cyclical_encode(series: pd.Series, max_val: int) -> tuple[pd.Series, pd.Series]:
    """
    Encode a cyclical feature (hour, day) as sin/cos pair.
    This preserves the circular nature — hour 23 is close to hour 0.

    Args:
        series: numeric series to encode
        max_val: maximum value in the cycle (24 for hours, 7 for days)

    Returns:
        Tuple of (sin_encoded, cos_encoded) series.
    """
    sin_enc = np.sin(2 * np.pi * series / max_val)
    cos_enc = np.cos(2 * np.pi * series / max_val)
    return sin_enc, cos_enc


def compute_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute temporal behavioral features per user.

    Args:
        df: raw events DataFrame

    Returns:
        DataFrame with one row per user_id and temporal features.
    """
    print("Computing temporal features ...")

    df = df.copy()
    df["event_time"] = pd.to_datetime(df["event_time"], utc=True)

    # Extract time components
    df["hour"] = df["event_time"].dt.hour
    df["day_of_week"] = df["event_time"].dt.dayofweek  # 0=Monday, 6=Sunday
    df["day_of_month"] = df["event_time"].dt.day
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    df["is_night"] = df["hour"].between(22, 23) | df["hour"].between(0, 4)

    # Payday window: 1st–3rd and 14th–16th of month
    df["is_payday_window"] = (
        df["day_of_month"].between(1, 3) | df["day_of_month"].between(14, 16)
    ).astype(int)

    # ------------------------------------------------------------------ #
    # 1. Preferred hour — mode hour per user
    # ------------------------------------------------------------------ #
    preferred_hour = (
        df.groupby("user_id")["hour"]
        .agg(lambda x: x.mode().iloc[0])
        .rename("preferred_hour")
    )

    # ------------------------------------------------------------------ #
    # 2. Preferred day of week — mode day per user
    # ------------------------------------------------------------------ #
    preferred_day = (
        df.groupby("user_id")["day_of_week"]
        .agg(lambda x: x.mode().iloc[0])
        .rename("preferred_day_of_week")
    )

    # ------------------------------------------------------------------ #
    # 3. Weekend activity ratio
    # ------------------------------------------------------------------ #
    weekend_ratio = (
        df.groupby("user_id")["is_weekend"]
        .mean()
        .rename("weekend_activity_ratio")
        .round(4)
    )

    # ------------------------------------------------------------------ #
    # 4. Night owl score
    # ------------------------------------------------------------------ #
    night_owl = (
        df.groupby("user_id")["is_night"]
        .mean()
        .rename("night_owl_score")
        .round(4)
    )

    # ------------------------------------------------------------------ #
    # 5. Payday spike ratio
    # ------------------------------------------------------------------ #
    payday_ratio = (
        df.groupby("user_id")["is_payday_window"]
        .mean()
        .rename("payday_activity_ratio")
        .round(4)
    )

    # ------------------------------------------------------------------ #
    # 6. Recency trend — is the user becoming more or less active?
    #    Compare event count in first half vs second half of obs window.
    # ------------------------------------------------------------------ #
    midpoint = df["event_time"].min() + (df["event_time"].max() - df["event_time"].min()) / 2

    first_half = (
        df[df["event_time"] <= midpoint]
        .groupby("user_id")
        .size()
        .rename("events_first_half")
    )
    second_half = (
        df[df["event_time"] > midpoint]
        .groupby("user_id")
        .size()
        .rename("events_second_half")
    )

    trend = pd.concat([first_half, second_half], axis=1).fillna(0)
    trend["activity_trend"] = (
        (trend["events_second_half"] - trend["events_first_half"])
        / (trend["events_first_half"] + trend["events_second_half"] + 1)
    ).round(4)
    # Positive = growing activity, negative = declining (churning signal)

    # ------------------------------------------------------------------ #
    # 7. Combine all temporal features
    # ------------------------------------------------------------------ #
    temporal = (
        pd.concat([preferred_hour, preferred_day, weekend_ratio,
                   night_owl, payday_ratio, trend["activity_trend"]], axis=1)
        .reset_index()
    )

    # Cyclical encoding of preferred hour and day
    sin_hour, cos_hour = cyclical_encode(temporal["preferred_hour"], 24)
    sin_day, cos_day = cyclical_encode(temporal["preferred_day_of_week"], 7)

    temporal["preferred_hour_sin"] = sin_hour.round(4)
    temporal["preferred_hour_cos"] = cos_hour.round(4)
    temporal["preferred_day_sin"] = sin_day.round(4)
    temporal["preferred_day_cos"] = cos_day.round(4)

    # Drop raw hour/day (keep encoded versions)
    temporal = temporal.drop(columns=["preferred_hour", "preferred_day_of_week"])

    print(f"  Users with temporal features: {len(temporal):,}")
    print(f"  Temporal features: {len(temporal.columns) - 1} (excl. user_id)")

    return temporal


if __name__ == "__main__":
    from session_features import load_events

    events = load_events(months=["2019-Oct"])
    temporal = compute_temporal_features(events)

    print("\nSample output:")
    print(temporal.head())
    print("\nFeature stats:")
    print(temporal.describe().round(3))

    output = DATA_DIR / "features_temporal.parquet"
    temporal.to_parquet(output, index=False, engine="pyarrow")
    print(f"\nSaved → {output}")
