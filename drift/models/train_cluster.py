"""
drift/models/train_cluster.py

Clusters users into behavioral archetypes using HDBSCAN.
Unlike K-Means, HDBSCAN finds clusters of arbitrary shape
and labels outliers as noise — more realistic for user behavior.

Usage:
    python drift/models/train_cluster.py

Outputs:
    - data/cluster_labels.parquet — user_id + cluster label + archetype name
    - data/cluster_model.pkl — fitted HDBSCAN model
    - data/cluster_profiles.json — archetype summary statistics
"""

import json
import pickle
import warnings
import pandas as pd
import numpy as np
from pathlib import Path
from dotenv import load_dotenv

import hdbscan
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

warnings.filterwarnings("ignore")
load_dotenv()

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

# ------------------------------------------------------------------ #
# Archetype name mapping
# Hand-crafted names based on behavioral cluster profiles.
# Updated after inspecting cluster_profiles.json post-training.
# ------------------------------------------------------------------ #
ARCHETYPE_NAMES = {
    -1: "Outlier",           # HDBSCAN noise points
    0: "The Window Shopper",     # High views, low conversion
    1: "The Decisive Buyer",     # Low browse, high purchase rate
    2: "The Cart Abandoner",     # High cart adds, low purchases
    3: "The Weekend Binge",      # High weekend activity ratio
    4: "The Deal Hunter",        # High payday activity, price sensitive
    5: "The Loyal Regular",      # High brand loyalty, frequent sessions
    6: "The Casual Browser",     # Low frequency, low depth
}

# Features to use for clustering — behavioral signals only, not the label
CLUSTER_FEATURES = [
    "recency_days",
    "total_sessions",
    "sessions_per_week",
    "avg_events_per_session",
    "avg_duration_min",
    "cart_abandonment_rate",
    "purchase_conversion_rate",
    "weekend_activity_ratio",
    "night_owl_score",
    "payday_activity_ratio",
    "activity_trend",
    "category_diversity",
    "brand_loyalty_score",
    "avg_price_point",
]

HDBSCAN_PARAMS = {
    "min_cluster_size": 2000,
    "min_samples": 50,
    "metric": "euclidean",
    "cluster_selection_method": "eom",
    "prediction_data": True,
}


def load_and_prepare(path: Path, sample_size: int = 200_000) -> tuple:
    """
    Load merged features, select clustering features, scale, reduce dims.

    We sample for HDBSCAN speed — 200K users is representative of 3M.

    Returns:
        X_scaled, X_pca, user_ids, scaler, pca
    """
    print(f"Loading features ...")
    df = pd.read_parquet(path)

    # Keep only users with purchase history for meaningful clustering
    df = df[df["total_sessions"] >= 2].copy()
    print(f"  Users with ≥2 sessions: {len(df):,}")

    # Sample for clustering speed
    if len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42)
        print(f"  Sampled: {sample_size:,}")

    # Select and clean features
    available = [f for f in CLUSTER_FEATURES if f in df.columns]
    missing = set(CLUSTER_FEATURES) - set(available)
    if missing:
        print(f"  Missing features (skipped): {missing}")

    X = df[available].fillna(0)
    user_ids = df["user_id"].values

    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # PCA to 8 components — reduces noise, speeds up HDBSCAN
    pca = PCA(n_components=8, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    explained = pca.explained_variance_ratio_.sum()
    print(f"  PCA: 8 components explain {explained:.1%} of variance")

    return X_scaled, X_pca, user_ids, df, scaler, pca, available


def train_clusters(X_pca: np.ndarray) -> hdbscan.HDBSCAN:
    """Fit HDBSCAN on PCA-reduced features."""
    print(f"\nFitting HDBSCAN ...")
    clusterer = hdbscan.HDBSCAN(**HDBSCAN_PARAMS)
    clusterer.fit(X_pca)

    labels = clusterer.labels_
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    noise_pct = (labels == -1).mean()

    print(f"  Clusters found: {n_clusters}")
    print(f"  Noise points: {noise_pct:.1%}")
    print(f"  Cluster distribution:")
    for label, count in sorted(
        pd.Series(labels).value_counts().items()
    ):
        name = ARCHETYPE_NAMES.get(label, f"Cluster {label}")
        print(f"    {label} ({name}): {count:,}")

    return clusterer


def compute_silhouette(X_pca: np.ndarray, labels: np.ndarray) -> float:
    """Compute silhouette score excluding noise points."""
    mask = labels != -1
    if mask.sum() < 2:
        return 0.0
    score = silhouette_score(
        X_pca[mask],
        labels[mask],
        sample_size=10_000,
        random_state=42,
    )
    print(f"  Silhouette score (excl. noise): {score:.4f}")
    return round(float(score), 4)


def build_profiles(df: pd.DataFrame, labels: np.ndarray, feature_cols: list) -> dict:
    """
    Build cluster profile summaries — mean feature values per cluster.
    These are used to name archetypes and feed the GenAI persona layer.
    """
    df = df.copy()
    df["cluster"] = labels
    df["archetype"] = df["cluster"].map(
        lambda x: ARCHETYPE_NAMES.get(x, f"Cluster {x}")
    )

    profiles = {}
    for cluster_id in sorted(df["cluster"].unique()):
        cluster_df = df[df["cluster"] == cluster_id]
        archetype = ARCHETYPE_NAMES.get(cluster_id, f"Cluster {cluster_id}")
        profile = {
            "archetype": archetype,
            "cluster_id": int(cluster_id),
            "user_count": int(len(cluster_df)),
            "pct_of_sample": round(len(cluster_df) / len(df) * 100, 1),
            "features": {
                col: round(float(cluster_df[col].mean()), 4)
                for col in feature_cols
                if col in cluster_df.columns
            },
            "churn_rate": round(float(cluster_df["churned"].mean()), 4)
            if "churned" in cluster_df.columns else None,
        }
        profiles[str(cluster_id)] = profile

    return profiles


def main():
    features_path = DATA_DIR / "features_merged.parquet"
    if not features_path.exists():
        raise FileNotFoundError(
            "features_merged.parquet not found. "
            "Run all three feature scripts first."
        )

    X_scaled, X_pca, user_ids, df, scaler, pca, feature_cols = \
        load_and_prepare(features_path)

    # Train
    clusterer = train_clusters(X_pca)
    labels = clusterer.labels_

    # Evaluate
    silhouette = compute_silhouette(X_pca, labels)

    # Build profiles
    print("\nBuilding archetype profiles ...")
    profiles = build_profiles(df, labels, feature_cols)

    # Save profiles
    profiles_path = DATA_DIR / "cluster_profiles.json"
    with open(profiles_path, "w") as f:
        json.dump(profiles, f, indent=2)
    print(f"Profiles saved → {profiles_path}")

    # Save cluster labels
    labels_df = pd.DataFrame({
        "user_id": user_ids,
        "cluster": labels,
        "archetype": [ARCHETYPE_NAMES.get(l, f"Cluster {l}") for l in labels],
    })
    labels_path = DATA_DIR / "cluster_labels.parquet"
    labels_df.to_parquet(labels_path, index=False)
    print(f"Labels saved → {labels_path}")

    # Save model + preprocessors
    model_bundle = {
        "clusterer": clusterer,
        "scaler": scaler,
        "pca": pca,
        "feature_cols": feature_cols,
        "archetype_names": ARCHETYPE_NAMES,
        "silhouette_score": silhouette,
    }
    model_path = DATA_DIR / "cluster_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model_bundle, f)
    print(f"Model bundle saved → {model_path}")

    # Print archetype summary
    print("\n" + "=" * 60)
    print("BEHAVIORAL ARCHETYPES")
    print("=" * 60)
    for cluster_id, profile in profiles.items():
        if int(cluster_id) == -1:
            continue
        print(f"\n{profile['archetype']} (Cluster {cluster_id})")
        print(f"  Users: {profile['user_count']:,} ({profile['pct_of_sample']}%)")
        if profile['churn_rate'] is not None:
            print(f"  Churn rate: {profile['churn_rate']:.1%}")
        print("  Key features:")
        # Show top 5 most distinctive features
        for feat, val in list(profile["features"].items())[:5]:
            print(f"    {feat}: {val}")


if __name__ == "__main__":
    main()
