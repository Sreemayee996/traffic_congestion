import pandas as pd
import numpy as np
import joblib
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = BASE_DIR / "data" / "webster_results_full.csv"
MODEL_DIR = BASE_DIR / "models_webster"

CONGESTION_FEATURES_NUMERIC = [
    "Traffic_Volume", "Average_Speed", "Travel_Time_Index",
    "Road_Capacity_Utilization_Pct", "Incident_Level",
]
CONGESTION_FEATURES_CATEGORICAL = ["Area_ID", "Intersection_ID", "Peak_Traffic_Status"]
CONGESTION_TARGET = "Congestion_Score"

RISK_FEATURES_NUMERIC = [
    "Traffic_Volume", "Average_Speed", "Travel_Time_Index",
    "Road_Capacity_Utilization_Pct", "Congestion_Score",
]
RISK_FEATURES_CATEGORICAL = ["Area_ID", "Intersection_ID", "Peak_Traffic_Status", "Bottleneck_Status"]
RISK_TARGET_RAW = "Accident_Flag"

def load_data():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"\nInput file not found:\n{INPUT_FILE}\n"
            "Please run webster_preprocessing.py, then webster_batch.py first."
        )
    return pd.read_csv(INPUT_FILE)

def _build_pipeline(model, categorical_columns):
    preprocessor = ColumnTransformer(
        transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), categorical_columns)],
        remainder="passthrough",
    )
    return Pipeline(steps=[("preprocess", preprocessor), ("model", model)])

def train_congestion_model(df, random_state=42):
    features = CONGESTION_FEATURES_CATEGORICAL + CONGESTION_FEATURES_NUMERIC
    X = df[features]
    y = df[CONGESTION_TARGET]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

    pipeline = _build_pipeline(
        RandomForestRegressor(n_estimators=300, max_depth=14, min_samples_leaf=3,
                               random_state=random_state, n_jobs=-1),
        CONGESTION_FEATURES_CATEGORICAL,
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

def train_risk_model(df, random_state=42):
    df = df.copy()
    df["high_risk_flag"] = (df[RISK_TARGET_RAW] == "YES").astype(int)

    features = RISK_FEATURES_CATEGORICAL + RISK_FEATURES_NUMERIC
    X = df[features]
    y = df["high_risk_flag"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y if y.nunique() > 1 else None
    )

    pipeline = _build_pipeline(
        RandomForestClassifier(n_estimators=300, max_depth=12, min_samples_leaf=3,
                                random_state=random_state, n_jobs=-1, class_weight="balanced"),
        RISK_FEATURES_CATEGORICAL,
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

def save_models(congestion_pipeline, risk_pipeline, metrics):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(congestion_pipeline, MODEL_DIR / "congestion_model.pkl")
    joblib.dump(risk_pipeline, MODEL_DIR / "risk_model.pkl")
    import json
    with open(MODEL_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

def load_models():
    congestion_pipeline = joblib.load(MODEL_DIR / "congestion_model.pkl")
    risk_pipeline = joblib.load(MODEL_DIR / "risk_model.pkl")
    return congestion_pipeline, risk_pipeline

def build_congestion_feature_row(area_id, intersection_id, traffic_volume, average_speed,
                                  travel_time_index, road_capacity_utilization_pct, incident_level,
                                  peak_traffic_status="NORMAL_TRAFFIC"):
    return pd.DataFrame([{
        "Area_ID": area_id, "Intersection_ID": intersection_id,
        "Peak_Traffic_Status": peak_traffic_status, "Traffic_Volume": traffic_volume,
        "Average_Speed": average_speed, "Travel_Time_Index": travel_time_index,
        "Road_Capacity_Utilization_Pct": road_capacity_utilization_pct, "Incident_Level": incident_level,
    }])

def build_risk_feature_row(area_id, intersection_id, traffic_volume, average_speed,
                            travel_time_index, road_capacity_utilization_pct, congestion_score,
                            peak_traffic_status="NORMAL_TRAFFIC", bottleneck_status="NORMAL"):
    return pd.DataFrame([{
        "Area_ID": area_id, "Intersection_ID": intersection_id,
        "Peak_Traffic_Status": peak_traffic_status, "Bottleneck_Status": bottleneck_status,
        "Traffic_Volume": traffic_volume, "Average_Speed": average_speed,
        "Travel_Time_Index": travel_time_index,
        "Road_Capacity_Utilization_Pct": road_capacity_utilization_pct,
        "Congestion_Score": congestion_score,
    }])

def predict_congestion(pipeline, feature_row):
    return float(np.clip(pipeline.predict(feature_row)[0], 0, 100))

def predict_risk_probability(pipeline, feature_row):
    return float(pipeline.predict_proba(feature_row)[0, 1])