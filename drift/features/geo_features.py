"""
drift/features/geo_features.py

Computes category and brand affinity features per user.
"Geospatial" here refers to behavioral space — where users
spend their attention across the product category landscape.

Features computed:
    - Top category affinity (which category they view most)
    - Category diversity score (Shannon entropy across categories)
    - Brand loyalty score (repeat views of same brand / total views)
    - Avg price point (what price tier they engage with)
    - Price sensitivity (std dev of prices they interact with)
    - Electronics affinity, fashion affinity (key category flags)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

# Top-level categories in REES46 dataset
TOP_CATEGORIES = [
    "electronics",
    "apparel",
    "appliances",
    "furniture",
    "sport",
    "kids",
    "auto",
]


def shannon_entropy(series: pd.Series) -> float:
    """Compute Shannon entropy of a categorical series — higher = more diverse."""
    counts = series.value_counts(normalize=True)
    return float(-(counts * np.log2(counts + 1e-10)).sum())


def extract_top_category(category_code: pd.Series) -> pd.Series:
    """Extract top-level category from dot-separated category_code."""
    return category_code.str.split(".").str[0].fillna("unknown")


def compute_geo_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute category and brand affinity features per user.

    Args:
        df: raw events DataFrame

    Returns:
        DataFrame with one row per user_id and category/brand features.
    """
    print("Computing category and brand features ...")

    df = df.copy()
    df["event_time"] = pd.to_datetime(df["event_time"], utc=True)
    df["top_category"] = extract_top_category(df["category_code"])

    # Work with view events for affinity (what they browse, not just buy)
    views = df[df["event_type"] == "view"].copy()
    purchases = df[df["event_type"] == "purchase"].copy()

    # ------------------------------------------------------------------ #
    # 1. Top category per user (most viewed)
    # ------------------------------------------------------------------ #
    top_cat = (
        views.groupby("user_id")["top_category"]
        .agg(lambda x: x.mode().iloc[0] if len(x) > 0 else "unknown")
        .rename("top_category")
    )

    # ------------------------------------------------------------------ #
    # 2. Category diversity score (entropy of category views)
    # ------------------------------------------------------------------ #
    cat_diversity = (
        views.groupby("user_id")["top_category"]
        .apply(shannon_entropy)
        .rename("category_diversity")
        .round(4)
    )

    # ------------------------------------------------------------------ #
    # 3. Per-category affinity flags (fraction of views in that category)
    # ------------------------------------------------------------------ #
    total_views_per_user = views.groupby("user_id").size().rename("total_views")

    category_affinities = {}
    for cat in TOP_CATEGORIES:
        cat_views = (
            views[views["top_category"] == cat]
            .groupby("user_id")
            .size()
            .rename(f"{cat}_views")
        )
        affinity = (cat_views / total_views_per_user).fillna(0).round(4)
        category_affinities[f"{cat}_affinity"] = affinity

    affinity_df = pd.DataFrame(category_affinities).fillna(0)

    # ------------------------------------------------------------------ #
    # 4. Brand loyalty score
    #    = (views of top brand) / (total views with brand info)
    # ------------------------------------------------------------------ #
    brand_views = views[views["brand"].notna()].copy()

    top_brand_views = (
        brand_views.groupby("user_id")["brand"]
        .apply(lambda x: x.value_counts().iloc[0] if len(x) > 0 else 0)
        .rename("top_brand_views")
    )
    total_brand_views = (
        brand_views.groupby("user_id").size().rename("total_brand_views")
    )
    brand_loyalty = (top_brand_views / total_brand_views).fillna(0).round(4).rename("brand_loyalty_score")

    # ------------------------------------------------------------------ #
    # 5. Price point features
    # ------------------------------------------------------------------ #
    price_stats = (
        purchases.groupby("user_id")["price"]
        .agg(
            avg_price_point="mean",
            price_sensitivity="std",
        )
        .round(2)
    )
    price_stats["price_sensitivity"] = price_stats["price_sensitivity"].fillna(0)

    # Also get avg browsed price (from views)
    avg_browse_price = (
        views.groupby("user_id")["price"]
        .mean()
        .rename("avg_browse_price")
        .round(2)
    )

    # ------------------------------------------------------------------ #
    # 6. Combine all features
    # ------------------------------------------------------------------ #
    geo = (
        pd.concat([
            top_cat,
            cat_diversity,
            affinity_df,
            brand_loyalty,
            price_stats,
            avg_browse_price,
        ], axis=1)
        .reset_index()
        .rename(columns={"index": "user_id"})
    )

    # One-hot encode top_category (low cardinality)
    top_cat_dummies = pd.get_dummies(
        geo["top_category"], prefix="topcategory", dtype=float
    )
    geo = pd.concat([geo.drop(columns=["top_category"]), top_cat_dummies], axis=1)

    # Fill any remaining NaNs
    geo = geo.fillna(0)

    print(f"  Users with geo features: {len(geo):,}")
    print(f"  Geo features: {len(geo.columns) - 1} (excl. user_id)")

    return geo


def merge_all_features(
    session_path: Path,
    temporal_path: Path,
    geo_path: Path,
) -> pd.DataFrame:
    """
    Merge all three feature sets into a single modeling DataFrame.

    Args:
        session_path: path to session features Parquet
        temporal_path: path to temporal features Parquet
        geo_path: path to geo features Parquet

    Returns:
        Merged DataFrame ready for model training.
    """
    print("Merging all feature sets ...")

    session = pd.read_parquet(session_path)
    temporal = pd.read_parquet(temporal_path)
    geo = pd.read_parquet(geo_path)

    merged = (
        session
        .merge(temporal, on="user_id", how="inner")
        .merge(geo, on="user_id", how="inner")
    )

    print(f"  Final dataset: {len(merged):,} users × {len(merged.columns)} features")
    return merged


if __name__ == "__main__":
    from session_features import load_events

    events = load_events(months=["2019-Oct"])
    geo = compute_geo_features(events)

    print("\nSample output:")
    print(geo.head())
    print("\nFeature stats:")
    print(geo.describe().round(3))

    output = DATA_DIR / "features_geo.parquet"
    geo.to_parquet(output, index=False, engine="pyarrow")
    print(f"\nSaved → {output}")

    # Merge all feature sets if others exist
    session_path = DATA_DIR / "features_session.parquet"
    temporal_path = DATA_DIR / "features_temporal.parquet"

    if session_path.exists() and temporal_path.exists():
        merged = merge_all_features(session_path, temporal_path, output)
        merged_output = DATA_DIR / "features_merged.parquet"
        merged.to_parquet(merged_output, index=False, engine="pyarrow")
        print(f"Merged features saved → {merged_output}")
