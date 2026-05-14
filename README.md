# drift

> Behavioral intelligence platform — user archetype modeling, churn prediction, and GenAI persona generation on 285M ecommerce events.

---

## What it does

Most analytics platforms tell you *what* users did. Drift tells you *who they are* and *where they're going*.

Given a stream of user interaction events, Drift:

1. Engineers session, temporal, and geospatial features from raw event logs
2. Clusters users into behavioral archetypes using HDBSCAN
3. Predicts churn probability per user with an XGBoost classifier
4. Generates plain-English persona reports by passing SHAP explanations through an LLM
5. Monitors production data for distribution drift with Evidently AI

Built end-to-end on AWS — ingestion through S3 and Athena, training via SageMaker, serving via a containerized FastAPI endpoint, orchestrated with Prefect.

---

## Architecture

```
REES46 ecommerce events (285M rows)
            │
            ▼
    ┌───────────────┐     ┌─────────────┐     ┌──────────────┐
    │    AWS S3     │────▶│   Prefect   │────▶│ AWS Athena   │
    │  Raw Parquet  │     │ Orchestrate │     │  SQL layer   │
    └───────────────┘     └─────────────┘     └──────────────┘
            │
            ▼
    ┌─────────────────────────────────────────────────────────┐
    │                  Feature Engineering                    │
    │  Session (RFM · velocity · depth)                       │
    │  Temporal (time-of-day · day-of-week · payday cycle)    │
    │  Geospatial (merchant category · location clusters)     │
    └─────────────────────────────────────────────────────────┘
            │
            ▼
    ┌──────────────┐     ┌──────────────────┐     ┌──────────────────┐
    │  SageMaker   │     │    HDBSCAN       │     │    MLflow        │
    │  XGBoost     │     │  Archetype       │     │  Experiment      │
    │  Churn model │     │  Clustering      │     │  Registry        │
    └──────────────┘     └──────────────────┘     └──────────────────┘
            │
            ▼
    ┌──────────────┐     ┌──────────────────┐     ┌──────────────────┐
    │   FastAPI    │     │  GenAI Persona   │     │  Docker +        │
    │  REST layer  │────▶│  SHAP → LLM      │     │  GitHub Actions  │
    │   endpoint   │     │  narrative       │     │  CI/CD           │
    └──────────────┘     └──────────────────┘     └──────────────────┘
            │
            ▼
    ┌──────────────┐     ┌──────────────────┐
    │ Evidently AI │     │    Streamlit     │
    │ Drift report │     │   Dashboard      │
    └──────────────┘     └──────────────────┘
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
| Cloud training | AWS SageMaker |
| Explainability | SHAP |
| GenAI layer | Claude API (Anthropic) |
| Serving | FastAPI |
| Containerization | Docker |
| CI/CD | GitHub Actions |
| Monitoring | Evidently AI |
| Dashboard | Streamlit |

---

## Dataset

[REES46 ecommerce behavior data](https://www.kaggle.com/datasets/mkechinov/ecommerce-behavior-data-from-multi-category-store) — 285M user interaction events (views, cart additions, purchases) across a multi-category ecommerce store.

> Download instructions in `data/README.md`

---

## Repo structure

```
drift/
├── README.md
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .github/
│   └── workflows/
│       └── ci.yml
├── data/
│   └── README.md
├── drift/
│   ├── ingestion/
│   │   ├── upload_to_s3.py
│   │   └── athena_queries.sql
│   ├── features/
│   │   ├── session_features.py
│   │   ├── temporal_features.py
│   │   └── geo_features.py
│   ├── models/
│   │   ├── train_churn.py
│   │   ├── train_cluster.py
│   │   └── evaluate.py
│   ├── serving/
│   │   ├── api.py
│   │   └── persona.py
│   └── monitoring/
│       └── drift_report.py
├── pipelines/
│   └── prefect_flow.py
├── app/
│   └── streamlit_app.py
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_model_experiments.ipynb
└── tests/
    ├── test_features.py
    └── test_api.py
```

---

## Quickstart

### Prerequisites

- Python 3.11+
- Docker
- AWS account with S3 and SageMaker access
- Anthropic API key

### Setup

```bash
git clone https://github.com/deepikapratapa/drift.git
cd drift
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # fill in your keys
```

### Run locally

```bash
# Start the API
uvicorn drift.serving.api:app --reload

# Run the dashboard
streamlit run app/streamlit_app.py
```

### Run with Docker

```bash
docker-compose up --build
```

---

## Key results

| Model | Metric | Score |
|---|---|---|
| XGBoost churn classifier | ROC-AUC | — |
| XGBoost churn classifier | Precision@K | — |
| HDBSCAN clustering | Silhouette score | — |

> Results will be updated as training runs complete.

---

## User archetypes

Drift identifies behavioral clusters from session data. Named archetypes emerging from the REES46 dataset:

| Archetype | Behavior pattern |
|---|---|
| The Window Shopper | High view rate, near-zero conversion, long sessions |
| The Decisive Buyer | Low browse time, high purchase rate, short sessions |
| The Cart Abandoner | Consistent add-to-cart, rarely completes checkout |
| The Weekend Binge | Concentrated activity on weekends, dormant weekdays |
| The Deal Hunter | Spikes during sale events, price-sensitive category focus |

> Archetypes are learned from data — final names and descriptions updated post-training.

---

## GenAI persona layer

SHAP feature importances from the churn classifier are passed to the Claude API with a structured prompt. The model returns a plain-English user persona report:

```
User #4820917 — "The Cart Abandoner"

This user has added items to cart 14 times in the past 30 days but completed
only 1 purchase. Their sessions are longest on Sunday evenings (avg 22 min),
and they browse predominantly in the Electronics category. Churn probability:
0.81. Recommended intervention: targeted checkout nudge with limited-time
offer, deployed Sunday 6–8pm.
```

---

## Monitoring

Evidently AI generates weekly data drift reports comparing the current week's feature distribution against the training baseline. Reports are saved to S3 and surfaced in the Streamlit dashboard.

---

## Development

```bash
# Run tests
pytest tests/

# Lint
ruff check drift/

# Type check
mypy drift/
```

---

## License

MIT
