"""
drift/models/train_churn.py

Trains an XGBoost churn classifier on the merged feature set.
Tracks all experiments with MLflow — parameters, metrics, and
the trained model artifact.

Usage:
    python drift/models/train_churn.py

Outputs:
    - MLflow run logged to mlruns/
    - Model saved to data/churn_model.json
    - SHAP values saved to data/shap_values.parquet
"""

import os
import json
import warnings
import pandas as pd
import numpy as np
from pathlib import Path
from dotenv import load_dotenv

import mlflow
import mlflow.xgboost
import xgboost as xgb
import shap

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)
from sklearn.utils.class_weight import compute_class_weight

warnings.filterwarnings("ignore")
load_dotenv()

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")

# ------------------------------------------------------------------ #
# Config
# ------------------------------------------------------------------ #
LABEL_COL = "churned"
DROP_COLS = ["user_id", "churned"]
TEST_SIZE = 0.2
RANDOM_STATE = 42

XGB_PARAMS = {
    "n_estimators": 500,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 10,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "eval_metric": "auc",
    "early_stopping_rounds": 30,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
    "tree_method": "hist",
}


def load_features(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    """Load merged features and split into X, y."""
    print(f"Loading features from {path.name} ...")
    df = pd.read_parquet(path)
    print(f"  Shape: {df.shape}")
    print(f"  Churn rate: {df[LABEL_COL].mean():.1%}")

    y = df[LABEL_COL]
    X = df.drop(columns=DROP_COLS)

    # Drop any remaining non-numeric columns
    non_numeric = X.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric:
        print(f"  Dropping non-numeric cols: {non_numeric}")
        X = X.drop(columns=non_numeric)

    return X, y


def compute_metrics(y_true, y_pred_proba, threshold=0.5) -> dict:
    """Compute classification metrics at a given threshold."""
    y_pred = (y_pred_proba >= threshold).astype(int)
    return {
        "roc_auc": round(roc_auc_score(y_true, y_pred_proba), 4),
        "avg_precision": round(average_precision_score(y_true, y_pred_proba), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y_true, y_pred, zero_division=0), 4),
    }


def train(features_path: Path) -> None:
    X, y = load_features(features_path)

    # Train/test split — stratified to preserve churn ratio
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    print(f"\nTrain: {len(X_train):,} | Test: {len(X_test):,}")

    # Handle class imbalance — scale_pos_weight upweights the minority class
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    scale_pos_weight = neg / pos
    print(f"  scale_pos_weight: {scale_pos_weight:.2f}")

    # MLflow experiment
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment("drift-churn-classifier")

    with mlflow.start_run(run_name="xgboost-baseline"):

        # Log parameters
        params = {**XGB_PARAMS, "scale_pos_weight": round(float(scale_pos_weight), 2)}
        mlflow.log_params(params)
        mlflow.log_param("train_size", len(X_train))
        mlflow.log_param("test_size", len(X_test))
        mlflow.log_param("n_features", X_train.shape[1])
        mlflow.log_param("churn_rate", round(float(y.mean()), 4))

        # Train model
        print("\nTraining XGBoost ...")
        model = xgb.XGBClassifier(
            **{k: v for k, v in XGB_PARAMS.items()
               if k != "early_stopping_rounds"},
            scale_pos_weight=scale_pos_weight,
            early_stopping_rounds=XGB_PARAMS["early_stopping_rounds"],
        )

        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=50,
        )

        # Evaluate
        print("\nEvaluating ...")
        train_proba = model.predict_proba(X_train)[:, 1]
        test_proba = model.predict_proba(X_test)[:, 1]

        train_metrics = compute_metrics(y_train, train_proba)
        test_metrics = compute_metrics(y_test, test_proba)

        print("\nTrain metrics:")
        for k, v in train_metrics.items():
            print(f"  {k}: {v}")

        print("\nTest metrics:")
        for k, v in test_metrics.items():
            print(f"  {k}: {v}")

        print("\nClassification report (test):")
        print(classification_report(
            y_test,
            (test_proba >= 0.5).astype(int),
            target_names=["not churned", "churned"]
        ))

        # Log metrics to MLflow
        for k, v in train_metrics.items():
            mlflow.log_metric(f"train_{k}", v)
        for k, v in test_metrics.items():
            mlflow.log_metric(f"test_{k}", v)

        # Log model
        mlflow.xgboost.log_model(model, "churn_model")

        # Save model locally
        model_path = DATA_DIR / "churn_model.json"
        model.save_model(str(model_path))
        print(f"\nModel saved → {model_path}")

        # ------------------------------------------------------------------ #
        # SHAP values — on a sample for speed
        # ------------------------------------------------------------------ #
        print("\nComputing SHAP values (sample of 5,000) ...")
        sample_idx = np.random.RandomState(RANDOM_STATE).choice(
            len(X_test), size=min(5000, len(X_test)), replace=False
        )
        X_sample = X_test.iloc[sample_idx].reset_index(drop=True)

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)

        # Save SHAP values
        shap_df = pd.DataFrame(shap_values, columns=X_sample.columns)
        shap_df["churn_proba"] = test_proba[sample_idx]
        shap_path = DATA_DIR / "shap_values.parquet"
        shap_df.to_parquet(shap_path, index=False)
        print(f"SHAP values saved → {shap_path}")

        # Log feature importance summary
        feature_importance = pd.Series(
            model.feature_importances_,
            index=X_train.columns
        ).sort_values(ascending=False)

        top_features = feature_importance.head(10).to_dict()
        mlflow.log_dict(top_features, "top_features.json")

        print("\nTop 10 features by importance:")
        for feat, imp in top_features.items():
            print(f"  {feat}: {imp:.4f}")

        run_id = mlflow.active_run().info.run_id
        print(f"\nMLflow run ID: {run_id}")
        print("Done. View results: mlflow ui")


if __name__ == "__main__":
    features_path = DATA_DIR / "features_merged.parquet"
    if not features_path.exists():
        raise FileNotFoundError(
            "features_merged.parquet not found. "
            "Run all three feature scripts first."
        )
    train(features_path)
