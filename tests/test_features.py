# tests/test_features.py
# Unit tests for feature engineering modules

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from drift.features.temporal_features import cyclical_encode, compute_temporal_features
from drift.features.geo_features import shannon_entropy, extract_top_category


# ------------------------------------------------------------------ #
# Fixtures
# ------------------------------------------------------------------ #
@pytest.fixture
def sample_events():
    """Minimal synthetic event DataFrame for testing."""
    np.random.seed(42)
    n = 1000
    users = np.random.choice([1001, 1002, 1003, 1004, 1005], size=n)
    sessions = [f"sess_{u}_{i%3}" for u,i in zip(users, range(n))]

    df = pd.DataFrame({
        "event_time": pd.date_range("2019-10-01", periods=n, freq="1h", tz="UTC"),
        "event_type": np.random.choice(["view","cart","purchase"], size=n, p=[0.7,0.2,0.1]),
        "product_id": np.random.randint(1000, 9999, n),
        "category_id": np.random.randint(100, 999, n),
        "category_code": np.random.choice(
            ["electronics.smartphone","apparel.shoes","appliances.kitchen",None],
            size=n, p=[0.4,0.3,0.2,0.1]
        ),
        "brand": np.random.choice(["apple","samsung","nike",None], size=n, p=[0.3,0.3,0.3,0.1]),
        "price": np.random.uniform(10, 500, n).round(2),
        "user_id": users,
        "user_session": sessions,
    })
    return df


# ------------------------------------------------------------------ #
# Cyclical encoding tests
# ------------------------------------------------------------------ #
def test_cyclical_encode_hour():
    """Hour 0 and hour 24 should encode to the same value."""
    series = pd.Series([0, 6, 12, 18, 23])
    sin_enc, cos_enc = cyclical_encode(series, 24)
    assert len(sin_enc) == 5
    assert len(cos_enc) == 5
    # sin(0) should be 0
    assert abs(sin_enc.iloc[0]) < 1e-10
    # cos(0) should be 1
    assert abs(cos_enc.iloc[0] - 1.0) < 1e-10


def test_cyclical_encode_range():
    """All encoded values should be in [-1, 1]."""
    series = pd.Series(range(24))
    sin_enc, cos_enc = cyclical_encode(series, 24)
    assert sin_enc.between(-1, 1).all()
    assert cos_enc.between(-1, 1).all()


def test_cyclical_encode_day():
    """Day 0 and day 7 should be equivalent."""
    s0 = pd.Series([0])
    s7 = pd.Series([7])
    sin0, cos0 = cyclical_encode(s0, 7)
    sin7, cos7 = cyclical_encode(s7, 7)
    assert abs(float(sin0.iloc[0]) - float(sin7.iloc[0])) < 1e-10
    assert abs(float(cos0.iloc[0]) - float(cos7.iloc[0])) < 1e-10


# ------------------------------------------------------------------ #
# Shannon entropy tests
# ------------------------------------------------------------------ #
def test_shannon_entropy_uniform():
    """Uniform distribution should have max entropy."""
    series = pd.Series(["a","b","c","d"])
    entropy = shannon_entropy(series)
    assert entropy > 0


def test_shannon_entropy_single():
    """Single category should have near-zero entropy."""
    series = pd.Series(["a","a","a","a"])
    entropy = shannon_entropy(series)
    assert entropy < 0.01


def test_shannon_entropy_non_negative():
    """Entropy should always be non-negative."""
    for _ in range(10):
        series = pd.Series(np.random.choice(["x","y","z"], size=50))
        assert shannon_entropy(series) >= 0


# ------------------------------------------------------------------ #
# Top category extraction
# ------------------------------------------------------------------ #
def test_extract_top_category_normal():
    series = pd.Series(["electronics.smartphone", "apparel.shoes", "appliances.kitchen"])
    result = extract_top_category(series)
    assert result.tolist() == ["electronics", "apparel", "appliances"]


def test_extract_top_category_none():
    series = pd.Series([None, "electronics.phone"])
    result = extract_top_category(series)
    assert result.iloc[0] == "unknown"
    assert result.iloc[1] == "electronics"


def test_extract_top_category_no_dot():
    series = pd.Series(["electronics"])
    result = extract_top_category(series)
    assert result.iloc[0] == "electronics"


# ------------------------------------------------------------------ #
# Temporal features integration test
# ------------------------------------------------------------------ #
def test_temporal_features_shape(sample_events):
    """Should return one row per unique user."""
    result = compute_temporal_features(sample_events)
    n_users = sample_events["user_id"].nunique()
    assert len(result) == n_users


def test_temporal_features_columns(sample_events):
    """Should contain all expected feature columns."""
    result = compute_temporal_features(sample_events)
    expected = [
        "user_id", "weekend_activity_ratio", "night_owl_score",
        "payday_activity_ratio", "activity_trend",
        "preferred_hour_sin", "preferred_hour_cos",
        "preferred_day_sin", "preferred_day_cos",
    ]
    for col in expected:
        assert col in result.columns, f"Missing column: {col}"


def test_temporal_features_ranges(sample_events):
    """Ratio features should be in [0, 1]."""
    result = compute_temporal_features(sample_events)
    for col in ["weekend_activity_ratio", "night_owl_score", "payday_activity_ratio"]:
        assert result[col].between(0, 1).all(), f"{col} out of range"


def test_temporal_features_no_nulls(sample_events):
    """Result should have no null values."""
    result = compute_temporal_features(sample_events)
    assert result.isnull().sum().sum() == 0, "Found null values in temporal features"
