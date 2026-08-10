"""
app.py
------
Flask REST API for the Bangalore Traffic Congestion backend.
"""

from __future__ import annotations

import os

from flask import Flask, jsonify, request

from src.simulation import TrafficAnalyzer

DATA_PATH = os.environ.get("TRAFFIC_DATA_PATH", "data/Banglore_traffic_Dataset Working.csv")

app = Flask(__name__)
analyzer = TrafficAnalyzer(DATA_PATH)

def _require_area_road():
    area = request.args.get("area")
    road = request.args.get("road")
    if not area or not road:
        return None, None, (jsonify({"error": "Both 'area' and 'road' query params are required. "
                                                "See /api/locations for valid values."}), 400)
    return area, road, None

def _handle_lookup_error(exc: Exception):
    return jsonify({"error": str(exc)}), 404

@app.get("/api/health")
def health():
    return jsonify({
        "status": "ok",
        "rows_loaded": int(len(analyzer.data)),
        "locations": int(len(analyzer.location_catalog)),
        "ml_models_ready": analyzer.ml_ready,
    })

@app.get("/api/locations")
def locations():
    return jsonify(analyzer.list_locations())

@app.get("/api/congestion-score")
def congestion_score():
    area, road, err = _require_area_road()
    if err:
        return err
    try:
        return jsonify(analyzer.congestion_score(area, road))
    except ValueError as e:
        return _handle_lookup_error(e)

@app.get("/api/bottlenecks")
def bottlenecks():
    top_n = request.args.get("top_n", default=10, type=int)
    top_n = max(1, min(top_n, 50))
    return jsonify(analyzer.bottlenecks(top_n=top_n))

@app.get("/api/signal-timing")
def signal_timing():
    area, road, err = _require_area_road()
    if err:
        return err
    num_phases = request.args.get("num_phases", default=4, type=int)
    lanes_per_phase = request.args.get("lanes_per_phase", default=2, type=int)
    try:
        return jsonify(analyzer.signal_timing(area, road, num_phases=num_phases, lanes_per_phase=lanes_per_phase))
    except ValueError as e:
        return _handle_lookup_error(e)

@app.get("/api/accident-risk")
def accident_risk():
    area, road, err = _require_area_road()
    if err:
        return err
    try:
        return jsonify(analyzer.accident_risk(area, road))
    except ValueError as e:
        return _handle_lookup_error(e)

@app.get("/api/report")
def report():
    area, road, err = _require_area_road()
    if err:
        return err
    try:
        return jsonify(analyzer.full_report(area, road))
    except ValueError as e:
        return _handle_lookup_error(e)

@app.get("/api/predict")
def predict():
    area, road, err = _require_area_road()
    if err:
        return err
    weather = request.args.get("weather", default="Clear")
    roadwork = request.args.get("roadwork", default="No")
    month = request.args.get("month", default=6, type=int)
    day_of_week = request.args.get("day_of_week", default=2, type=int)
    try:
        return jsonify(analyzer.predict(area, road, weather=weather, roadwork=roadwork,
                                         month=month, day_of_week=day_of_week))
    except ValueError as e:
        return _handle_lookup_error(e)
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 409

@app.post("/api/train")
def train():
    metrics = analyzer.train_ml_models()
    return jsonify({"status": "trained", "metrics": metrics})

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)