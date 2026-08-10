import json
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent / "src"))

import webster_ml_models as ml_models

if __name__ == "__main__":

    print("Loading data...")
    df = ml_models.load_data()
    print(f"  {len(df):,} rows loaded from {ml_models.INPUT_FILE.name}")

    print("\nTraining Congestion Score model (RandomForestRegressor)...")
    congestion_pipeline, congestion_metrics = ml_models.train_congestion_model(df)
    print("  Metrics:", json.dumps(congestion_metrics, indent=2))

    print("\nTraining Accident Risk model (RandomForestClassifier)...")
    risk_pipeline, risk_metrics = ml_models.train_risk_model(df)
    print("  Metrics:", json.dumps(risk_metrics, indent=2))

    ml_models.save_models(
        congestion_pipeline,
        risk_pipeline,
        {"congestion_model": congestion_metrics, "risk_model": risk_metrics},
    )

    print("\nSUCCESS!")
    print(f"Saved models to: {ml_models.MODEL_DIR}")