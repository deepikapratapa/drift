<div align="center">

# drift

**Behavioral intelligence platform**

*User archetype modeling · Churn prediction · GenAI persona generation*

![CI](https://github.com/deepikapratapa/drift/actions/workflows/ci.yml/badge.svg)
[![HuggingFace](https://img.shields.io/badge/🤗%20Live%20Demo-HuggingFace%20Spaces-a855f7)](https://huggingface.co/spaces/dpratapa/drift)
![Python](https://img.shields.io/badge/Python-3.11-7c6af7?logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-AUC%200.9987-f76a8c)
![AWS](https://img.shields.io/badge/AWS-S3%20%7C%20Athena-f7a26a?logo=amazon-aws&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-containerized-3b82f6?logo=docker&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-6af7a2)

<br>

![Drift Dashboard](assets/dashboard-main.png)

<br>

### [→ Open Live Demo on HuggingFace Spaces](https://huggingface.co/spaces/dpratapa/drift)

</div>

---

## What it does

Most analytics platforms tell you **what** users did.
Drift tells you **who they are** and **where they're going**.

Given a stream of raw user interaction events, Drift:

- Engineers 48 behavioral features across 3M users from 109M raw events
- Predicts churn probability per user with an XGBoost classifier (AUC 0.9987)
- Clusters users into behavioral archetypes using HDBSCAN
- Explains predictions with SHAP feature importances
- Generates plain-English persona reports via LLaMA 3 70B (Groq)
- Monitors production data for feature distribution drift weekly

---

## Screenshots

<table>
<tr>
<td width="50%">

**User Analysis**
Input behavioral metrics → get churn score, archetype, risk factors, and recommended intervention.

![User Analysis](assets/dashboard-main.png)

</td>
<td width="50%">

**AI Persona Report**
SHAP values feed LLaMA 3 70B to generate a plain-English behavioral narrative.

![Persona Report](assets/dashboard-persona.png)

</td>
</tr>
<tr>
<td width="50%">

**Model Performance**
Live metrics, SHAP feature importance chart, ROC/PR curves, confusion matrix.

![Model Performance](assets/dashboard-model.png)

</td>
<td width="50%">

**Behavioral Archetypes**
HDBSCAN clusters with churn rates and feature profiles per segment.

![Archetypes](assets/dashboard-archetypes.png)

</td>
</tr>
</table>

---

## Architecture

```
REES46 ecommerce events (109M rows · Oct–Nov 2019)
                    │
                    ▼
        ┌─────────────────────┐
        │  Ingestion Pipeline  │
        │  CSV → Parquet → S3  │
        │  AWS Athena SQL layer│
        └──────────┬──────────┘
                   │
                   ▼
        ┌──────────────────────────────────────────────┐
        │             Feature Engineering              │
        │                                              │
        │  Session      Temporal       Geo/Category    │
        │  ─────────    ──────────     ────────────    │
        │  RFM           Hour/day       Category       │
        │  Velocity      cyclical enc   diversity      │
        │  Cart abdn     Night owl      Brand loyalty  │
        │  Conversion    Payday spike   Price point    │
        │  rate          Activity trend                │
        │                                              │
        │          48 features · 3M users              │
        └───────────────┬──────────────────────────────┘
                        │
            ┌───────────┴───────────┐
            ▼                       ▼
   ┌─────────────────┐    ┌──────────────────┐
   │  XGBoost Churn  │    │ HDBSCAN Behavior │
   │  Classifier     │    │ Clustering       │
   │  AUC: 0.9987    │    │ Archetypes       │
   │  MLflow tracked │    │ Silhouette: 0.28 │
   └────────┬────────┘    └────────┬─────────┘
            │                      │
            └──────────┬───────────┘
                       ▼
         ┌─────────────────────────┐
         │   SHAP Explainability   │
         │  Feature → risk factors │
         └────────────┬────────────┘
                      │
                      ▼
         ┌─────────────────────────┐
         │    Groq GenAI Layer     │
         │   LLaMA 3 · 70B         │
         │  SHAP → plain-English   │
         │    persona narratives   │
         └────────────┬────────────┘
                      │
           ┌──────────┴──────────┐
           ▼                     ▼
   ┌──────────────┐    ┌──────────────────┐
   │   FastAPI    │    │    Streamlit     │
   │  REST layer  │───▶│   Dashboard      │
   │   Docker     │    │  4 pages · live  │
   └──────────────┘    └──────────────────┘
                      │
                      ▼
         ┌─────────────────────────┐
         │   KS Drift Monitoring   │
         │  18 features · weekly   │
         │  JSON + plot reports    │
         └─────────────────────────┘
```

---

## Results

| Model | Metric | Score |
|---|---|---|
| XGBoost churn classifier | ROC-AUC | **0.9987** |
| XGBoost churn classifier | F1 Score | **0.9914** |
| XGBoost churn classifier | Precision | **0.9999** |
| XGBoost churn classifier | Recall | **0.9830** |
| HDBSCAN clustering | Silhouette score | **0.2756** |

Top predictive features by SHAP importance:

```
total_purchases        ████████████████████  0.7167
total_revenue          ██████                0.1207
avg_session_revenue    ████                  0.0812
recency_days           █                     0.0298
avg_price_point        █                     0.0200
```

---

## User archetypes

Behavioral clusters discovered via HDBSCAN on 200K users:

| Archetype | Users | Churn rate | Key signal |
|---|---|---|---|
| 🪟 The Window Shopper | 9,376 | 98.4% | High views, near-zero conversion |
| 🎯 The Decisive Buyer | 155,265 | 95.2% | Low browse time, high purchase rate |
| 🛒 The Cart Abandoner | — | — | High cart adds, rarely completes checkout |
| 📅 The Weekend Binge | — | — | Concentrated weekend activity |
| 💰 The Deal Hunter | — | — | Spikes around payday windows |

> Archetypes are learned from data — not hand-coded rules.

---

## GenAI persona layer

SHAP feature importances feed a structured prompt to LLaMA 3 70B via Groq API:

```
The Cart Abandoner is a high-browse, low-convert user who has added items to
cart 14 times in the past 30 days but completed only 1 purchase. Their sessions
are longest on Sunday evenings (avg 22 min), and they browse predominantly in
the Electronics category.

Churn probability: 87%. Recommended intervention: targeted checkout nudge with
limited-time offer, deployed Sunday 6–8pm.
```

---

## Tech stack

| Layer | Tools |
|---|---|
| Cloud storage | AWS S3, AWS Athena |
| Orchestration | Prefect |
| Feature engineering | Python, pandas, scikit-learn |
| Modeling | XGBoost, LightGBM, HDBSCAN |
| Experiment tracking | MLflow |
| Explainability | SHAP |
| GenAI layer | Groq API (LLaMA 3 70B) |
| Serving | FastAPI |
| Containerization | Docker |
| CI/CD | GitHub Actions |
| Monitoring | KS drift detection (scipy) |
| Dashboard | Streamlit |
| Deployment | HuggingFace Spaces |

---

## Dataset

[REES46 ecommerce behavior data](https://www.kaggle.com/datasets/mkechinov/ecommerce-behavior-data-from-multi-category-store) — 109M user interaction events (views, cart additions, purchases) across a multi-category ecommerce store, October–November 2019.

---

## Repo structure

```
drift/
├── assets/                 # Dashboard screenshots
├── drift/
│   ├── ingestion/          # S3 upload + Athena SQL
│   ├── features/           # Session, temporal, geo features
│   ├── models/             # XGBoost, HDBSCAN, SHAP, evaluation
│   ├── serving/            # FastAPI + Groq persona layer
│   └── monitoring/         # KS drift detection
├── app/
│   └── streamlit_app.py    # Dashboard
├── pipelines/
│   └── prefect_flow.py     # Orchestration
├── tests/                  # 26 tests · pytest
├── .github/workflows/      # GitHub Actions CI
├── Dockerfile
└── docker-compose.yml
```

---

## Quickstart

```bash
git clone https://github.com/deepikapratapa/drift.git
cd drift
conda create -n drift python=3.11 -y && conda activate drift
pip install -r requirements.txt && pip install -e .
cp .env.example .env        # fill in AWS + Groq keys
```

Run feature engineering:

```bash
python drift/features/session_features.py
python drift/features/temporal_features.py
python drift/features/geo_features.py
```

Train models:

```bash
export MLFLOW_TRACKING_URI=sqlite:///mlflow.db
python drift/models/train_churn.py
python drift/models/train_cluster.py
python drift/models/evaluate.py
```

Start the API and dashboard:

```bash
uvicorn drift.serving.api:app --port 8000 &
streamlit run app/streamlit_app.py
```

Run with Docker:

```bash
docker-compose up --build
```

Run tests:

```bash
pytest tests/ -v
```

---

<div align="center">

**[→ Open Live Demo on HuggingFace Spaces](https://huggingface.co/spaces/dpratapa/drift)**

*Built by [Deepika Pratapa](https://github.com/deepikapratapa)*

</div>
