"""
train_models.py
----------------
One-off (or scheduled) training script. Run this whenever the dataset
is refreshed:

    python train_models.py

It trains the congestion regressor + accident-risk classifier and
saves them to models/, along with metrics.json for a quick sanity
check on model quality. app.py loads these pickles at startup -- it
does not retrain on every request.
"""

import json

from src.simulation import TrafficAnalyzer

DATA_PATH = "data/Banglore_traffic_Dataset Working.csv"

if __name__ == "__main__":
    print("Loading + scoring data...")
    analyzer = TrafficAnalyzer(DATA_PATH, load_ml=False)
    print(f"  {len(analyzer.data):,} rows scored across {len(analyzer.location_catalog)} locations")

    print("Training models (RandomForest congestion regressor + accident-risk classifier)...")
    metrics = analyzer.train_ml_models()

    print("\nDone. Metrics:")
    print(json.dumps(metrics, indent=2))
    print("\nSaved to models/congestion_model.pkl and models/risk_model.pkl")