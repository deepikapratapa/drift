"""
drift/monitoring/drift_report.py

Data drift monitoring for Drift.
Uses KS test to detect feature distribution shift
between training baseline and current data.

Usage:
    python drift/monitoring/drift_report.py

Outputs:
    - data/drift_summary.json       — drift summary per feature
    - data/plots/drift_summary.png  — KS statistic bar chart
    - data/plots/drift_distributions.png — distribution overlays
"""

import json
import warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from scipy import stats
from dotenv import load_dotenv

warnings.filterwarnings("ignore")
load_dotenv()

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
PLOTS_DIR = DATA_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

MONITOR_FEATURES = [
    "recency_days", "total_sessions", "sessions_per_week",
    "total_revenue", "avg_session_revenue", "avg_events_per_session",
    "avg_duration_min", "total_views", "total_cart_adds", "total_purchases",
    "cart_abandonment_rate", "purchase_conversion_rate",
    "weekend_activity_ratio", "night_owl_score", "payday_activity_ratio",
    "activity_trend", "category_diversity", "brand_loyalty_score",
]

STYLE = {
    "bg": "#0f0f0f", "accent": "#7c6af7", "accent2": "#f76a8c",
    "text": "#e0e0e0", "grid": "#2a2a2a",
}


def set_style():
    plt.rcParams.update({
        "figure.facecolor": STYLE["bg"], "axes.facecolor": STYLE["bg"],
        "axes.edgecolor": STYLE["grid"], "axes.labelcolor": STYLE["text"],
        "text.color": STYLE["text"], "xtick.color": STYLE["text"],
        "ytick.color": STYLE["text"], "grid.color": STYLE["grid"],
        "font.family": "monospace",
    })


def load_reference_and_current(features_path: Path):
    print("Loading features ...")
    df = pd.read_parquet(features_path)
    available = [f for f in MONITOR_FEATURES if f in df.columns]
    df = df[available].fillna(0)

    split = int(len(df) * 0.8)
    reference = df.iloc[:split].sample(n=min(10_000, split), random_state=42)
    current = df.iloc[split:].sample(n=min(5_000, len(df) - split), random_state=42)

    print(f"  Reference: {len(reference):,} users")
    print(f"  Current:   {len(current):,} users")
    return reference, current


def detect_drift(reference: pd.DataFrame, current: pd.DataFrame, alpha: float = 0.05) -> dict:
    print("Running KS drift detection ...")
    results = {}
    drifted = []
    stable = []

    for col in reference.columns:
        ref_vals = reference[col].dropna().values
        cur_vals = current[col].dropna().values
        if len(ref_vals) == 0 or len(cur_vals) == 0:
            continue

        ks_stat, p_value = stats.ks_2samp(ref_vals, cur_vals)
        drift_detected = p_value < alpha

        results[col] = {
            "ks_statistic": round(float(ks_stat), 4),
            "p_value": round(float(p_value), 6),
            "drift_detected": drift_detected,
            "ref_mean": round(float(ref_vals.mean()), 4),
            "cur_mean": round(float(cur_vals.mean()), 4),
            "mean_shift_pct": round(float(
                abs(cur_vals.mean() - ref_vals.mean()) / (abs(ref_vals.mean()) + 1e-10) * 100), 2),
        }

        if drift_detected:
            drifted.append(col)
            print(f"  ⚠ DRIFT: {col} — KS={ks_stat:.4f}, p={p_value:.6f}")
        else:
            stable.append(col)

    drift_share = len(drifted) / len(results) if results else 0.0
    return {
        "dataset_drift_detected": drift_share > 0.5,
        "drift_share": round(drift_share, 4),
        "n_drifted_features": len(drifted),
        "n_stable_features": len(stable),
        "n_total_features": len(results),
        "drifted_features": drifted,
        "stable_features": stable,
        "feature_details": results,
        "alpha": alpha,
        "test": "Kolmogorov-Smirnov",
    }


def plot_drift_summary(summary: dict) -> None:
    set_style()
    details = summary["feature_details"]
    features = list(details.keys())
    ks_stats = [details[f]["ks_statistic"] for f in features]
    colors = [STYLE["accent2"] if details[f]["drift_detected"] else STYLE["accent"] for f in features]

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(features, ks_stats, color=colors, alpha=0.85)
    ax.axvline(x=0.05, color="#666", linestyle="--", linewidth=1, label="α=0.05")
    ax.set_xlabel("KS Statistic")
    ax.set_title("Feature Drift Summary (KS Test)", pad=15)
    ax.grid(True, axis="x", alpha=0.3)
    legend = [mpatches.Patch(color=STYLE["accent2"], label="Drift detected"),
              mpatches.Patch(color=STYLE["accent"], label="Stable")]
    ax.legend(handles=legend, loc="lower right")
    plt.tight_layout()
    path = PLOTS_DIR / "drift_summary.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=STYLE["bg"])
    plt.close()
    print(f"Drift summary plot → {path}")


def plot_feature_distributions(reference: pd.DataFrame, current: pd.DataFrame,
                                summary: dict, n_features: int = 6) -> None:
    set_style()
    details = summary["feature_details"]
    top_features = sorted(details.keys(), key=lambda f: details[f]["ks_statistic"], reverse=True)[:n_features]

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()

    for i, feat in enumerate(top_features):
        ax = axes[i]
        ref_vals = reference[feat].dropna()
        cur_vals = current[feat].dropna()
        bins = np.linspace(min(ref_vals.min(), cur_vals.min()),
                           max(ref_vals.max(), cur_vals.max()), 40)
        ax.hist(ref_vals, bins=bins, alpha=0.6, color=STYLE["accent"], label="Reference", density=True)
        ax.hist(cur_vals, bins=bins, alpha=0.6, color=STYLE["accent2"], label="Current", density=True)
        drift = details[feat]["drift_detected"]
        ks = details[feat]["ks_statistic"]
        ax.set_title(f"{feat}\nKS={ks:.4f} {'⚠' if drift else '✓'}",
                     color=STYLE["accent2"] if drift else STYLE["text"], fontsize=9)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.2)

    for j in range(len(top_features), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Feature Distribution: Reference vs Current", fontsize=13)
    plt.tight_layout()
    path = PLOTS_DIR / "drift_distributions.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=STYLE["bg"])
    plt.close()
    print(f"Distribution plots → {path}")


def print_summary(summary: dict) -> None:
    print("\n" + "=" * 55)
    print("DRIFT MONITORING SUMMARY")
    print("=" * 55)
    status = "⚠  DRIFT DETECTED" if summary["dataset_drift_detected"] else "✓  NO SIGNIFICANT DRIFT"
    print(f"Status: {status}")
    print(f"Drift share: {summary['drift_share']:.1%} of features")
    print(f"Drifted: {summary['n_drifted_features']} / {summary['n_total_features']} features")
    if summary["drifted_features"]:
        print("\nDrifted features:")
        for feat in summary["drifted_features"]:
            d = summary["feature_details"][feat]
            print(f"  {feat}: KS={d['ks_statistic']}, p={d['p_value']}, mean shift={d['mean_shift_pct']}%")
    else:
        print("\nAll features stable.")


if __name__ == "__main__":
    features_path = DATA_DIR / "features_merged.parquet"
    if not features_path.exists():
        raise FileNotFoundError("features_merged.parquet not found.")

    reference, current = load_reference_and_current(features_path)
    summary = detect_drift(reference, current)

    json_path = DATA_DIR / "drift_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2, default=lambda x: bool(x) if hasattr(x, "item") else str(x))
    print(f"\nJSON summary → {json_path}")

    plot_drift_summary(summary)
    plot_feature_distributions(reference, current, summary)
    print_summary(summary)
