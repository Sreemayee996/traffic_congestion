"""
app.py
------
Flask REST API for the Bangalore Traffic Congestion backend.

Includes both:
- Pipeline B (live, on-the-fly calculation) endpoints — original.
- Pipeline A (batch/Webster CSV results) endpoints — new, so the
  pre-calculated batch recommendations and full location detail
  are actually served, instead of sitting unused in CSV files.
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


def _require_area_intersection_id():
    area_id = request.args.get("area_id")
    intersection_id = request.args.get("intersection_id")
    if not area_id or not intersection_id:
        return None, None, (jsonify({"error": "Both 'area_id' and 'intersection_id' query params "
                                                "are required."}), 400)
    return area_id, intersection_id, None


def _handle_lookup_error(exc: Exception):
    return jsonify({"error": str(exc)}), 404


def _handle_not_ready_error(exc: Exception):
    return jsonify({"error": str(exc)}), 409


# ----------------------------------------------------------------------
# Pipeline B — live, on-the-fly endpoints (original)
# ----------------------------------------------------------------------

@app.get("/api/health")
def health():
    return jsonify({
        "status": "ok",
        "rows_loaded": int(len(analyzer.data)),
        "locations": int(len(analyzer.location_catalog)),
        "ml_models_ready": analyzer.ml_ready,
        "batch_data_ready": analyzer.batch_data_ready,
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


# ----------------------------------------------------------------------
# Pipeline A — batch/Webster CSV results, now actually served
# ----------------------------------------------------------------------

@app.get("/api/location-detail")
def location_detail():
    """Full raw detail (every column, every row) for one intersection,
    sourced from webster_results_full.csv (Pipeline A)."""
    area_id, intersection_id, err = _require_area_intersection_id()
    if err:
        return err
    try:
        return jsonify(analyzer.location_full_detail(area_id, intersection_id))
    except ValueError as e:
        return _handle_lookup_error(e)
    except RuntimeError as e:
        return _handle_not_ready_error(e)


@app.get("/api/improved-signal-timing")
def improved_signal_timing():
    """Signal timing recommendation using Pipeline A's properly
    calculated lanes/saturation flow, more accurate than the live
    on-the-fly /api/signal-timing calculation."""
    area_id, intersection_id, err = _require_area_intersection_id()
    if err:
        return err
    try:
        return jsonify(analyzer.improved_signal_timing(area_id, intersection_id))
    except ValueError as e:
        return _handle_lookup_error(e)
    except RuntimeError as e:
        return _handle_not_ready_error(e)


@app.get("/api/batch-recommendation")
def batch_recommendation():
    """Before/after comparison (current congestion vs recommended
    signal timing) for one intersection, from batch_report.py's
    simulation_summary.csv (Pipeline A)."""
    area_id, intersection_id, err = _require_area_intersection_id()
    if err:
        return err
    try:
        return jsonify(analyzer.batch_recommendation(area_id, intersection_id))
    except ValueError as e:
        return _handle_lookup_error(e)
    except RuntimeError as e:
        return _handle_not_ready_error(e)


@app.get("/api/bottlenecks-batch")
def bottlenecks_batch():
    """Bottleneck ranking sourced from the full historical batch
    summary (Pipeline A), as an alternative to /api/bottlenecks."""
    top_n = request.args.get("top_n", default=10, type=int)
    try:
        return jsonify(analyzer.bottleneck_ranking_batch(top_n=top_n))
    except RuntimeError as e:
        return _handle_not_ready_error(e)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)