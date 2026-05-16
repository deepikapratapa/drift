"""
drift/models/evaluate.py

Generates evaluation artifacts for the churn classifier:
    - ROC curve
    - Precision-recall curve
    - SHAP summary plot
    - SHAP feature importance bar chart
    - Confusion matrix at optimal threshold

All plots saved to data/plots/ for the Streamlit dashboard.

Usage:
    python drift/models/evaluate.py
"""

import warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import shap
import xgboost as xgb
from pathlib import Path
from sklearn.metrics import (
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

warnings.filterwarnings("ignore")

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
PLOTS_DIR = DATA_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

STYLE = {
    "bg": "#0f0f0f",
    "accent": "#7c6af7",
    "accent2": "#f76a8c",
    "text": "#e0e0e0",
    "grid": "#2a2a2a",
}


def set_style():
    """Apply dark theme consistent with Drift's aesthetic."""
    plt.rcParams.update({
        "figure.facecolor": STYLE["bg"],
        "axes.facecolor": STYLE["bg"],
        "axes.edgecolor": STYLE["grid"],
        "axes.labelcolor": STYLE["text"],
        "text.color": STYLE["text"],
        "xtick.color": STYLE["text"],
        "ytick.color": STYLE["text"],
        "grid.color": STYLE["grid"],
        "font.family": "monospace",
    })


def plot_roc_curve(y_true, y_proba) -> str:
    """Plot and save ROC curve. Returns path."""
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)

    set_style()
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color=STYLE["accent"], lw=2,
            label=f"ROC curve (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], color=STYLE["grid"], lw=1, linestyle="--")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve — Churn Classifier", pad=15)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    path = PLOTS_DIR / "roc_curve.png"
    fig.savefig(path, dpi=150, bbox_inches="tight",
                facecolor=STYLE["bg"])
    plt.close()
    print(f"Saved → {path}")
    return str(path)


def plot_precision_recall(y_true, y_proba) -> str:
    """Plot and save precision-recall curve. Returns path."""
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    ap = average_precision_score(y_true, y_proba)
    baseline = y_true.mean()

    set_style()
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(recall, precision, color=STYLE["accent2"], lw=2,
            label=f"PR curve (AP = {ap:.4f})")
    ax.axhline(y=baseline, color=STYLE["grid"], lw=1, linestyle="--",
               label=f"Baseline (churn rate = {baseline:.2f})")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve — Churn Classifier", pad=15)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    path = PLOTS_DIR / "pr_curve.png"
    fig.savefig(path, dpi=150, bbox_inches="tight",
                facecolor=STYLE["bg"])
    plt.close()
    print(f"Saved → {path}")
    return str(path)


def plot_shap_summary(shap_values: pd.DataFrame) -> str:
    """Plot SHAP feature importance bar chart. Returns path."""
    mean_abs_shap = shap_values.drop(
        columns=["churn_proba"], errors="ignore"
    ).abs().mean().sort_values(ascending=True).tail(15)

    set_style()
    fig, ax = plt.subplots(figsize=(8, 7))
    bars = ax.barh(
        mean_abs_shap.index,
        mean_abs_shap.values,
        color=STYLE["accent"],
        alpha=0.85,
    )
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title("Feature Importance (SHAP) — Top 15", pad=15)
    ax.grid(True, axis="x", alpha=0.3)

    # Value labels
    for bar, val in zip(bars, mean_abs_shap.values):
        ax.text(
            val + 0.0005, bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}", va="center", fontsize=8,
            color=STYLE["text"]
        )

    plt.tight_layout()
    path = PLOTS_DIR / "shap_importance.png"
    fig.savefig(path, dpi=150, bbox_inches="tight",
                facecolor=STYLE["bg"])
    plt.close()
    print(f"Saved → {path}")
    return str(path)


def plot_confusion_matrix(y_true, y_proba, threshold=0.5) -> str:
    """Plot confusion matrix at given threshold. Returns path."""
    y_pred = (y_proba >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)

    set_style()
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Not Churned", "Churned"]
    )
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"Confusion Matrix (threshold={threshold})", pad=15)
    plt.tight_layout()

    path = PLOTS_DIR / "confusion_matrix.png"
    fig.savefig(path, dpi=150, bbox_inches="tight",
                facecolor=STYLE["bg"])
    plt.close()
    print(f"Saved → {path}")
    return str(path)


def plot_churn_score_distribution(y_true, y_proba) -> str:
    """Plot distribution of churn scores by true label. Returns path."""
    set_style()
    fig, ax = plt.subplots(figsize=(8, 5))

    bins = np.linspace(0, 1, 50)
    ax.hist(y_proba[y_true == 0], bins=bins, alpha=0.6,
            color=STYLE["accent"], label="Not Churned", density=True)
    ax.hist(y_proba[y_true == 1], bins=bins, alpha=0.6,
            color=STYLE["accent2"], label="Churned", density=True)

    ax.set_xlabel("Predicted Churn Probability")
    ax.set_ylabel("Density")
    ax.set_title("Churn Score Distribution by True Label", pad=15)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    path = PLOTS_DIR / "score_distribution.png"
    fig.savefig(path, dpi=150, bbox_inches="tight",
                facecolor=STYLE["bg"])
    plt.close()
    print(f"Saved → {path}")
    return str(path)


def main():
    # Load SHAP values (includes churn_proba column)
    shap_path = DATA_DIR / "shap_values.parquet"
    if not shap_path.exists():
        raise FileNotFoundError(
            "shap_values.parquet not found. Run train_churn.py first."
        )

    print("Loading SHAP values ...")
    shap_df = pd.read_parquet(shap_path)
    churn_proba = shap_df["churn_proba"].values

    # Load model to get predictions on full test set
    model_path = DATA_DIR / "churn_model.json"
    if not model_path.exists():
        raise FileNotFoundError(
            "churn_model.json not found. Run train_churn.py first."
        )

    print("Loading model ...")
    model = xgb.XGBClassifier()
    model.load_model(str(model_path))

    # Load features for full evaluation
    features_path = DATA_DIR / "features_merged.parquet"
    df = pd.read_parquet(features_path)
    y = df["churned"]
    X = df.drop(columns=["user_id", "churned"]).select_dtypes(include=[np.number])

    print(f"Generating predictions on {len(X):,} users ...")
    y_proba = model.predict_proba(X)[:, 1]

    # Generate all plots
    print("\nGenerating evaluation plots ...")
    plot_roc_curve(y, y_proba)
    plot_precision_recall(y, y_proba)
    plot_shap_summary(shap_df)
    plot_confusion_matrix(y, y_proba)
    plot_churn_score_distribution(y.values, y_proba)

    print(f"\nAll plots saved to {PLOTS_DIR}")
    print("Run `streamlit run app/streamlit_app.py` to view in dashboard.")


if __name__ == "__main__":
    main()
