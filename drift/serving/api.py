"""
drift/serving/api.py

FastAPI REST endpoint for Drift.
Serves churn predictions and behavioral archetype assignments.
The Streamlit dashboard calls this API — it never loads the model directly.

Endpoints:
    GET  /health          — liveness check
    POST /predict         — single user churn prediction + archetype
    POST /predict/batch   — batch of users
    GET  /model/info      — model metadata and feature importance

Usage:
    uvicorn drift.serving.api:app --reload --port 8000
"""

import os
import json
import pickle
import hdbscan
import numpy as np
import pandas as pd
import xgboost as xgb
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

load_dotenv()

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

app = FastAPI(
    title="Drift API",
    description="Behavioral intelligence — churn prediction and user archetype assignment",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------------------------ #
# Model loading — done once at startup
# ------------------------------------------------------------------ #
churn_model: Optional[xgb.XGBClassifier] = None
cluster_bundle: Optional[dict] = None
feature_importance: Optional[dict] = None
model_feature_names: Optional[list] = None


@app.on_event("startup")
async def load_models():
    global churn_model, cluster_bundle, feature_importance, model_feature_names

    churn_model_path = DATA_DIR / "churn_model.json"
    cluster_model_path = DATA_DIR / "cluster_model.pkl"

    if not churn_model_path.exists():
        raise RuntimeError(
            f"Churn model not found at {churn_model_path}. "
            "Run drift/models/train_churn.py first."
        )

    print("Loading churn model ...")
    churn_model = xgb.XGBClassifier()
    churn_model.load_model(str(churn_model_path))
    model_feature_names = churn_model.get_booster().feature_names
    print(f"  Features: {len(model_feature_names)}")

    if cluster_model_path.exists():
        print("Loading cluster model ...")
        with open(cluster_model_path, "rb") as f:
            cluster_bundle = pickle.load(f)
        print("  Cluster model loaded.")

    # Load SHAP-based feature importance
    shap_path = DATA_DIR / "shap_values.parquet"
    if shap_path.exists():
        shap_df = pd.read_parquet(shap_path)
        importance = (
            shap_df.drop(columns=["churn_proba"], errors="ignore")
            .abs().mean()
            .sort_values(ascending=False)
            .head(10)
            .round(4)
            .to_dict()
        )
        feature_importance = importance

    print("Drift API ready.")


# ------------------------------------------------------------------ #
# Request / Response schemas
# ------------------------------------------------------------------ #
class UserFeatures(BaseModel):
    """Feature vector for a single user."""
    recency_days: float = Field(..., ge=0, description="Days since last session")
    total_sessions: int = Field(..., ge=1)
    sessions_per_week: float = Field(..., ge=0)
    total_revenue: float = Field(..., ge=0)
    avg_session_revenue: float = Field(..., ge=0)
    avg_events_per_session: float = Field(..., ge=0)
    avg_duration_min: float = Field(..., ge=0)
    total_views: int = Field(..., ge=0)
    total_cart_adds: int = Field(..., ge=0)
    total_purchases: int = Field(..., ge=0)
    cart_abandonment_rate: float = Field(..., ge=0, le=1)
    purchase_conversion_rate: float = Field(..., ge=0)
    weekend_activity_ratio: float = Field(..., ge=0, le=1)
    night_owl_score: float = Field(..., ge=0, le=1)
    payday_activity_ratio: float = Field(..., ge=0, le=1)
    activity_trend: float = Field(..., ge=-1, le=1)
    category_diversity: float = Field(..., ge=0)
    brand_loyalty_score: float = Field(..., ge=0, le=1)
    avg_price_point: float = Field(0.0, ge=0)
    price_sensitivity: float = Field(0.0, ge=0)

    class Config:
        json_schema_extra = {
            "example": {
                "recency_days": 5.0,
                "total_sessions": 12,
                "sessions_per_week": 2.8,
                "total_revenue": 340.50,
                "avg_session_revenue": 28.37,
                "avg_events_per_session": 8.2,
                "avg_duration_min": 12.4,
                "total_views": 98,
                "total_cart_adds": 14,
                "total_purchases": 3,
                "cart_abandonment_rate": 0.78,
                "purchase_conversion_rate": 0.03,
                "weekend_activity_ratio": 0.6,
                "night_owl_score": 0.25,
                "payday_activity_ratio": 0.18,
                "activity_trend": -0.12,
                "category_diversity": 1.42,
                "brand_loyalty_score": 0.45,
                "avg_price_point": 113.5,
                "price_sensitivity": 42.3,
            }
        }


class PredictionResponse(BaseModel):
    churn_probability: float
    churn_prediction: bool
    risk_level: str
    archetype: str
    top_risk_factors: list[dict]
    recommendation: str


class BatchPredictionRequest(BaseModel):
    users: list[UserFeatures]


class BatchPredictionResponse(BaseModel):
    predictions: list[PredictionResponse]
    summary: dict


# ------------------------------------------------------------------ #
# Helper functions
# ------------------------------------------------------------------ #
def features_to_df(user: UserFeatures) -> pd.DataFrame:
    """Convert a UserFeatures pydantic model to a DataFrame row."""
    data = user.model_dump()
    df = pd.DataFrame([data])

    # Align to model feature names — fill missing with 0
    if model_feature_names:
        for col in model_feature_names:
            if col not in df.columns:
                df[col] = 0.0
        df = df[model_feature_names]

    return df


def assign_archetype(user: UserFeatures) -> str:
    """Assign behavioral archetype using rule-based fallback."""
    if cluster_bundle is None:
        # Rule-based fallback if cluster model not loaded
        if user.purchase_conversion_rate > 0.05:
            return "The Decisive Buyer"
        elif user.cart_abandonment_rate > 0.7:
            return "The Cart Abandoner"
        elif user.weekend_activity_ratio > 0.6:
            return "The Weekend Binge"
        elif user.total_sessions <= 2 and user.total_purchases == 0:
            return "The Window Shopper"
        elif user.payday_activity_ratio > 0.25:
            return "The Deal Hunter"
        elif user.brand_loyalty_score > 0.6:
            return "The Loyal Regular"
        else:
            return "The Casual Browser"

    # Use trained cluster model
    try:
        feature_cols = cluster_bundle["feature_cols"]
        scaler = cluster_bundle["scaler"]
        pca = cluster_bundle["pca"]
        archetype_names = cluster_bundle["archetype_names"]

        data = user.model_dump()
        X = np.array([[data.get(f, 0.0) for f in feature_cols]])
        X_scaled = scaler.transform(X)
        X_pca = pca.transform(X_scaled)

        labels, _ = hdbscan.approximate_predict(
            cluster_bundle["clusterer"], X_pca
        )
        label = int(labels[0])
        return archetype_names.get(label, "The Casual Browser")
    except Exception:
        return "The Casual Browser"


def get_risk_factors(user: UserFeatures, churn_proba: float) -> list[dict]:
    """Return top risk factors based on feature values."""
    factors = []

    if user.recency_days > 20:
        factors.append({
            "factor": "High recency",
            "detail": f"Last seen {user.recency_days:.0f} days ago",
            "impact": "high"
        })
    if user.cart_abandonment_rate > 0.7:
        factors.append({
            "factor": "Cart abandonment",
            "detail": f"{user.cart_abandonment_rate:.0%} of cart sessions not purchased",
            "impact": "high"
        })
    if user.activity_trend < -0.1:
        factors.append({
            "factor": "Declining activity",
            "detail": "Session frequency dropping over time",
            "impact": "medium"
        })
    if user.total_purchases == 0:
        factors.append({
            "factor": "No purchase history",
            "detail": "User has never completed a purchase",
            "impact": "high"
        })
    if user.sessions_per_week < 0.5:
        factors.append({
            "factor": "Low engagement",
            "detail": f"Only {user.sessions_per_week:.1f} sessions per week",
            "impact": "medium"
        })

    return factors[:3] if factors else [
        {"factor": "Healthy engagement", "detail": "No significant risk signals", "impact": "low"}
    ]


def get_recommendation(archetype: str, churn_proba: float) -> str:
    """Return intervention recommendation based on archetype and risk."""
    recommendations = {
        "The Window Shopper": "Send personalized product recommendations with first-purchase discount.",
        "The Decisive Buyer": "Offer loyalty rewards — this user converts when motivated.",
        "The Cart Abandoner": "Trigger abandoned cart email within 2 hours with limited-time offer.",
        "The Weekend Binge": "Schedule promotional emails for Friday evening.",
        "The Deal Hunter": "Target with payday promotions on the 1st and 15th.",
        "The Loyal Regular": "Enroll in VIP program — high retention value.",
        "The Casual Browser": "Re-engagement campaign with bestsellers in their top category.",
        "Outlier": "Manual review recommended — atypical behavior pattern.",
    }
    return recommendations.get(archetype, "Monitor engagement over next 7 days.")


def get_risk_level(churn_proba: float) -> str:
    if churn_proba >= 0.85:
        return "critical"
    elif churn_proba >= 0.65:
        return "high"
    elif churn_proba >= 0.40:
        return "medium"
    else:
        return "low"


# ------------------------------------------------------------------ #
# Endpoints
# ------------------------------------------------------------------ #
@app.get("/health")
async def health():
    return {
        "status": "ok",
        "churn_model_loaded": churn_model is not None,
        "cluster_model_loaded": cluster_bundle is not None,
    }


@app.get("/model/info")
async def model_info():
    if churn_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {
        "model_type": "XGBoostClassifier",
        "n_features": len(model_feature_names) if model_feature_names else 0,
        "top_features_shap": feature_importance or {},
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(user: UserFeatures):
    if churn_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    df = features_to_df(user)
    churn_proba = float(churn_model.predict_proba(df)[0, 1])
    churn_prediction = churn_proba >= 0.5
    risk_level = get_risk_level(churn_proba)
    archetype = assign_archetype(user)
    risk_factors = get_risk_factors(user, churn_proba)
    recommendation = get_recommendation(archetype, churn_proba)

    return PredictionResponse(
        churn_probability=round(churn_proba, 4),
        churn_prediction=churn_prediction,
        risk_level=risk_level,
        archetype=archetype,
        top_risk_factors=risk_factors,
        recommendation=recommendation,
    )


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest):
    if churn_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    predictions = []
    for user in request.users:
        df = features_to_df(user)
        churn_proba = float(churn_model.predict_proba(df)[0, 1])
        churn_prediction = churn_proba >= 0.5
        risk_level = get_risk_level(churn_proba)
        archetype = assign_archetype(user)
        risk_factors = get_risk_factors(user, churn_proba)
        recommendation = get_recommendation(archetype, churn_proba)

        predictions.append(PredictionResponse(
            churn_probability=round(churn_proba, 4),
            churn_prediction=churn_prediction,
            risk_level=risk_level,
            archetype=archetype,
            top_risk_factors=risk_factors,
            recommendation=recommendation,
        ))

    churn_probas = [p.churn_probability for p in predictions]
    archetypes = [p.archetype for p in predictions]

    summary = {
        "total_users": len(predictions),
        "predicted_churners": sum(p.churn_prediction for p in predictions),
        "avg_churn_probability": round(float(np.mean(churn_probas)), 4),
        "high_risk_users": sum(p.risk_level in ["high", "critical"] for p in predictions),
        "archetype_distribution": {
            a: archetypes.count(a) for a in set(archetypes)
        },
    }

    return BatchPredictionResponse(predictions=predictions, summary=summary)
