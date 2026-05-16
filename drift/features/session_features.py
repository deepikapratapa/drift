"""
drift/features/session_features.py

Computes user-level session features from raw event logs.
These form the core input to both the churn classifier and
the HDBSCAN behavioral clustering.

Features computed:
    - Recency: days since last session
    - Frequency: total number of sessions
    - Monetary: total spend
    - Session depth: avg events per session
    - Cart abandonment rate: cart adds with no purchase / total cart adds
    - Purchase conversion rate: purchases / views
    - Velocity: sessions per week
    - Avg session duration in minutes
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def load_events(months: list[str] = ["2019-Oct"]) -> pd.DataFrame:
    """
    Load one or more monthly Parquet files from the data directory.

    Args:
        months: list of month slugs matching Parquet filenames
                e.g. ["2019-Oct"] or ["2019-Oct", "2019-Nov"]

    Returns:
        Combined DataFrame of all events.
    """
    frames = []
    for month in months:
        path = DATA_DIR / f"{month}.parquet"
        if not path.exists():
            raise FileNotFoundError(
                f"Parquet file not found: {path}\n"
                "Run drift/ingestion/upload_to_s3.py first."
            )
        print(f"Loading {path.name} ...")
        df = pd.read_parquet(path, engine="pyarrow")
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    print(f"  Loaded {len(combined):,} events total.")
    return combined


def compute_session_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate raw events into one row per user with session-level features.

    Args:
        df: raw events DataFrame with columns:
            event_time, event_type, product_id, category_id,
            category_code, brand, price, user_id, user_session

    Returns:
        DataFrame with one row per user_id and engineered features.
    """
    print("Computing session features ...")

    # Ensure datetime
    df["event_time"] = pd.to_datetime(df["event_time"], utc=True)

    # ------------------------------------------------------------------ #
    # 1. Session-level aggregation (per user_id + user_session)
    # ------------------------------------------------------------------ #
    session_agg = (
        df.groupby(["user_id", "user_session"])
        .agg(
            session_start=("event_time", "min"),
            session_end=("event_time", "max"),
            total_events=("event_type", "count"),
            views=("event_type", lambda x: (x == "view").sum()),
            cart_adds=("event_type", lambda x: (x == "cart").sum()),
            purchases=("event_type", lambda x: (x == "purchase").sum()),
            session_revenue=("price", lambda x: x[df.loc[x.index, "event_type"] == "purchase"].sum()),
        )
        .reset_index()
    )

    # Session duration in minutes
    session_agg["duration_min"] = (
        (session_agg["session_end"] - session_agg["session_start"])
        .dt.total_seconds()
        .div(60)
        .clip(lower=0)
    )

    # ------------------------------------------------------------------ #
    # 2. User-level aggregation (per user_id)
    # ------------------------------------------------------------------ #
    reference_date = df["event_time"].max()

    user_agg = (
        session_agg.groupby("user_id")
        .agg(
            # Recency
            last_session=("session_end", "max"),
            # Frequency
            total_sessions=("user_session", "count"),
            # Monetary
            total_revenue=("session_revenue", "sum"),
            avg_session_revenue=("session_revenue", "mean"),
            # Depth
            avg_events_per_session=("total_events", "mean"),
            avg_duration_min=("duration_min", "mean"),
            # Raw counts for derived features
            total_views=("views", "sum"),
            total_cart_adds=("cart_adds", "sum"),
            total_purchases=("purchases", "sum"),
        )
        .reset_index()
    )

    # ------------------------------------------------------------------ #
    # 3. Derived features
    # ------------------------------------------------------------------ #

    # Recency in days
    user_agg["recency_days"] = (
        (reference_date - user_agg["last_session"])
        .dt.total_seconds()
        .div(86400)
        .round(1)
    )

    # Velocity: sessions per week
    # (approximate — assumes data spans ~6 weeks for Oct+Nov)
    obs_days = (
        df["event_time"].max() - df["event_time"].min()
    ).total_seconds() / 86400
    weeks = max(obs_days / 7, 1)
    user_agg["sessions_per_week"] = (
        user_agg["total_sessions"] / weeks
    ).round(4)

    # Cart abandonment rate: sessions with cart but no purchase / sessions with cart
    cart_sessions = session_agg[session_agg["cart_adds"] > 0].copy()
    cart_sessions["abandoned"] = cart_sessions["purchases"] == 0

    abandonment = (
        cart_sessions.groupby("user_id")
        .agg(
            cart_sessions_count=("user_session", "count"),
            abandoned_count=("abandoned", "sum"),
        )
        .reset_index()
    )
    abandonment["cart_abandonment_rate"] = (
        abandonment["abandoned_count"] / abandonment["cart_sessions_count"]
    ).round(4)

    user_agg = user_agg.merge(
        abandonment[["user_id", "cart_abandonment_rate"]],
        on="user_id",
        how="left",
    )
    user_agg["cart_abandonment_rate"] = user_agg["cart_abandonment_rate"].fillna(0.0)

    # Purchase conversion rate: purchases / views
    user_agg["purchase_conversion_rate"] = (
        user_agg["total_purchases"] / user_agg["total_views"].replace(0, np.nan)
    ).fillna(0.0).round(4)

    # ------------------------------------------------------------------ #
    # 4. Churn label (for supervised learning)
    #    Definition: user has not made a purchase in the last 14 days
    #    of the observation window.
    # ------------------------------------------------------------------ #
    churn_cutoff = reference_date - pd.Timedelta(days=14)

    last_purchase = (
        df[df["event_type"] == "purchase"]
        .groupby("user_id")["event_time"]
        .max()
        .reset_index()
        .rename(columns={"event_time": "last_purchase_time"})
    )

    user_agg = user_agg.merge(last_purchase, on="user_id", how="left")
    user_agg["churned"] = (
        user_agg["last_purchase_time"].isna()
        | (user_agg["last_purchase_time"] < churn_cutoff)
    ).astype(int)

    # ------------------------------------------------------------------ #
    # 5. Final feature selection
    # ------------------------------------------------------------------ #
    feature_cols = [
        "user_id",
        "recency_days",
        "total_sessions",
        "sessions_per_week",
        "total_revenue",
        "avg_session_revenue",
        "avg_events_per_session",
        "avg_duration_min",
        "total_views",
        "total_cart_adds",
        "total_purchases",
        "cart_abandonment_rate",
        "purchase_conversion_rate",
        "churned",  # label
    ]

    result = user_agg[feature_cols].copy()

    print(f"  Users: {len(result):,}")
    print(f"  Churn rate: {result['churned'].mean():.1%}")
    print(f"  Features: {len(feature_cols) - 2} (excl. user_id and label)")

    return result


def save_features(df: pd.DataFrame, output_path: Path) -> None:
    """Save feature DataFrame to Parquet."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False, engine="pyarrow")
    print(f"Saved features → {output_path} ({output_path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    # Run with October data only for dev speed
    events = load_events(months=["2019-Oct"])
    features = compute_session_features(events)

    print("\nSample output:")
    print(features.head())
    print("\nFeature stats:")
    print(features.describe().round(2))

    output = DATA_DIR / "features_session.parquet"
    save_features(features, output)
