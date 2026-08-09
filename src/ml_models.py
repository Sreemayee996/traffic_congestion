"""
ml_models.py
------------
The "AI automation" layer. Two supervised models sit on top of the
rule-based scores in scoring.py:

1. CongestionRegressor  -- predicts Congestion Score from contextual
   features (area, road, weather, roadwork, calendar features) so the
   API can answer "what congestion should I expect on <road> on a
   <weekday> in <weather> with roadwork?" without needing a live
   sensor reading for that exact moment.

2. AccidentRiskClassifier -- predicts the probability a given
   context falls into the High/Critical accident-risk band, trained
   on the same contextual features.

Both are RandomForest models (robust to mixed categorical/numeric
features, minimal tuning needed, resistant to overfitting on a
modest-size dataset) wrapped in an sklearn Pipeline with a
ColumnTransformer so raw categorical strings can be passed straight
in from the API layer.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_DIR = BASE_DIR / "models"

FEATURE_NUMERIC = [
    "Public Transport Usage",
    "Parking Usage",
    "Pedestrian and Cyclist Count",
    "Month",
    "DayOfWeek",
    "IsWeekend",
]
FEATURE_CATEGORICAL = [
    "Area Name",
    "Road/Intersection Name",
    "Weather Conditions",
    "Roadwork and Construction Activity",
]
ALL_FEATURES = FEATURE_NUMERIC + FEATURE_CATEGORICAL

CONGESTION_TARGET = "Congestion Score"
RISK_TARGET_BINARY = "high_risk_flag"

def _build_pipeline(model) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), FEATURE_CATEGORICAL),
        ],
        remainder="passthrough",
    )
    return Pipeline(steps=[("preprocess", preprocessor), ("model", model)])

def train_congestion_model(scored_df: pd.DataFrame, random_state: int = 42) -> tuple[Pipeline, dict]:
    df = scored_df.dropna(subset=[CONGESTION_TARGET])
    X = df[ALL_FEATURES]
    y = df[CONGESTION_TARGET]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

    pipeline = _build_pipeline(
        RandomForestRegressor(n_estimators=300, max_depth=14, min_samples_leaf=3,
                               random_state=random_state, n_jobs=-1)
    )
    pipeline.fit(X_train, y_train)

    preds = pipeline.predict(X_test)
    metrics = {
        "mae": round(float(mean_absolute_error(y_test, preds)), 3),
        "r2": round(float(r2_score(y_test, preds)), 3),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
    }
    return pipeline, metrics

def train_risk_model(scored_df: pd.DataFrame, random_state: int = 42) -> tuple[Pipeline, dict]:
    df = scored_df.copy()
    df[RISK_TARGET_BINARY] = df["Accident Risk Level"].isin(["High", "Critical"]).astype(int)

    X = df[ALL_FEATURES]
    y = df[RISK_TARGET_BINARY]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y if y.nunique() > 1 else None
    )

    pipeline = _build_pipeline(
        RandomForestClassifier(n_estimators=300, max_depth=12, min_samples_leaf=3,
                                random_state=random_state, n_jobs=-1, class_weight="balanced")
    )
    pipeline.fit(X_train, y_train)

    proba = pipeline.predict_proba(X_test)[:, 1]
    metrics = {
        "roc_auc": round(float(roc_auc_score(y_test, proba)), 3) if y_test.nunique() > 1 else None,
        "positive_rate_train": round(float(y_train.mean()), 3),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
    }
    return pipeline, metrics

def save_models(congestion_pipeline: Pipeline, risk_pipeline: Pipeline,
                 metrics: dict, model_dir: str | Path = DEFAULT_MODEL_DIR) -> None:
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(congestion_pipeline, model_dir / "congestion_model.pkl")
    joblib.dump(risk_pipeline, model_dir / "risk_model.pkl")
    with open(model_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

def load_models(model_dir: str | Path = DEFAULT_MODEL_DIR) -> tuple[Pipeline, Pipeline]:
    model_dir = Path(model_dir)
    congestion_pipeline = joblib.load(model_dir / "congestion_model.pkl")
    risk_pipeline = joblib.load(model_dir / "risk_model.pkl")
    return congestion_pipeline, risk_pipeline

def build_feature_row(
    area: str,
    road: str,
    weather: str = "Clear",
    roadwork: str = "No",
    month: int = 6,
    day_of_week: int = 2,
    public_transport_usage: float = 45.0,
    parking_usage: float = 70.0,
    pedestrian_cyclist_count: float = 110.0,
) -> pd.DataFrame:
    """Builds a single-row feature frame in the exact schema the models expect,
    for ad-hoc predictions coming from the API."""
    return pd.DataFrame([{
        "Public Transport Usage": public_transport_usage,
        "Parking Usage": parking_usage,
        "Pedestrian and Cyclist Count": pedestrian_cyclist_count,
        "Month": month,
        "DayOfWeek": day_of_week,
        "IsWeekend": int(day_of_week in (5, 6)),
        "Area Name": area,
        "Road/Intersection Name": road,
        "Weather Conditions": weather,
        "Roadwork and Construction Activity": roadwork,
    }])

def predict_congestion(pipeline: Pipeline, feature_row: pd.DataFrame) -> float:
    return float(np.clip(pipeline.predict(feature_row)[0], 0, 100))

def predict_risk_probability(pipeline: Pipeline, feature_row: pd.DataFrame) -> float:
    return float(pipeline.predict_proba(feature_row)[0, 1])

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(BASE_DIR))
    from src.data_preprocessing import load_clean_data
    from src.scoring import score_all

    print("Loading and cleaning data...")
    df = load_clean_data()

    print("Scoring...")
    scored = score_all(df)

    print("Training congestion regressor...")
    congestion_pipeline, congestion_metrics = train_congestion_model(scored)
    print("Congestion model metrics:", congestion_metrics)

    print("Training accident risk classifier...")
    risk_pipeline, risk_metrics = train_risk_model(scored)
    print("Risk model metrics:", risk_metrics)

    print(f"Saving models to {DEFAULT_MODEL_DIR}...")
    save_models(
        congestion_pipeline, risk_pipeline,
        {"congestion": congestion_metrics, "risk": risk_metrics}
    )

    sample = build_feature_row(area=df["Area Name"].iloc[0], road=df["Road/Intersection Name"].iloc[0])
    print("Sample prediction - Congestion Score:", predict_congestion(congestion_pipeline, sample))
    print("Sample prediction - Risk Probability:", predict_risk_probability(risk_pipeline, sample))

    print("SUCCESS! Models saved and verified.")