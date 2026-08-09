"""
simulation.py
--------------
High-level facade used by app.py. Loads + scores the data once,
holds the trained ML models in memory, and exposes simple methods
that map 1:1 onto the API endpoints. Keeping this orchestration out
of app.py keeps the Flask layer thin (routing + HTTP concerns only).
"""

from __future__ import annotations

import os

import pandas as pd

from . import ml_models, scoring, webster
from .data_preprocessing import get_location_catalog, load_clean

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")


class TrafficAnalyzer:
    def __init__(self, csv_path: str, model_dir: str = MODEL_DIR, load_ml: bool = True):
        self.csv_path = csv_path
        self.model_dir = model_dir

        raw = load_clean(csv_path)
        self.data = scoring.score_all(raw)
        self.location_catalog = get_location_catalog(self.data)

        self.congestion_model = None
        self.risk_model = None
        if load_ml:
            self._try_load_ml()

    # ------------------------------------------------------------------
    # ML lifecycle
    # ------------------------------------------------------------------
    def _try_load_ml(self) -> None:
        try:
            self.congestion_model, self.risk_model = ml_models.load_models(self.model_dir)
        except FileNotFoundError:
            self.congestion_model, self.risk_model = None, None

    def train_ml_models(self) -> dict:
        congestion_pipeline, congestion_metrics = ml_models.train_congestion_model(self.data)
        risk_pipeline, risk_metrics = ml_models.train_risk_model(self.data)
        ml_models.save_models(
            congestion_pipeline, risk_pipeline,
            {"congestion_model": congestion_metrics, "risk_model": risk_metrics},
            self.model_dir,
        )
        self.congestion_model, self.risk_model = congestion_pipeline, risk_pipeline
        return {"congestion_model": congestion_metrics, "risk_model": risk_metrics}

    @property
    def ml_ready(self) -> bool:
        return self.congestion_model is not None and self.risk_model is not None

    # ------------------------------------------------------------------
    # Locations
    # ------------------------------------------------------------------
    def list_locations(self) -> list[dict]:
        return self.location_catalog.to_dict(orient="records")

    def _validate_location(self, area: str, road: str) -> None:
        match = self.location_catalog[
            (self.location_catalog["Area Name"].str.lower() == area.lower())
            & (self.location_catalog["Road/Intersection Name"].str.lower() == road.lower())
        ]
        if match.empty:
            raise ValueError(
                f"Unknown (area, road) combination: ({area!r}, {road!r}). "
                f"Call /api/locations for valid values."
            )

    # ------------------------------------------------------------------
    # Congestion score
    # ------------------------------------------------------------------
    def congestion_score(self, area: str, road: str) -> dict:
        self._validate_location(area, road)
        return scoring.location_summary(self.data, area, road)

    # ------------------------------------------------------------------
    # Bottlenecks
    # ------------------------------------------------------------------
    def bottlenecks(self, top_n: int = 10) -> list[dict]:
        ranking = scoring.bottleneck_ranking(self.data, top_n=top_n)
        return ranking.to_dict(orient="records")

    # ------------------------------------------------------------------
    # Signal timing (Webster's method)
    # ------------------------------------------------------------------
    def signal_timing(self, area: str, road: str, num_phases: int = 4, lanes_per_phase: int = 2) -> dict:
        self._validate_location(area, road)
        subset = self.data[
            (self.data["Area Name"].str.lower() == area.lower())
            & (self.data["Road/Intersection Name"].str.lower() == road.lower())
        ]
        avg_volume = float(subset["Traffic Volume"].mean())
        avg_utilization = float(subset["Road Capacity Utilization"].mean())

        result = webster.estimate_signal_timing(
            traffic_volume_vph=avg_volume,
            road_capacity_utilization_pct=avg_utilization,
            num_phases=num_phases,
            lanes_per_phase=lanes_per_phase,
        )
        out = result.to_dict()
        out.update({
            "area": area,
            "road": road,
            "avg_daily_traffic_volume": round(avg_volume, 1),
            "avg_road_capacity_utilization_pct": round(avg_utilization, 1),
        })
        return out

    # ------------------------------------------------------------------
    # Accident risk
    # ------------------------------------------------------------------
    def accident_risk(self, area: str, road: str) -> dict:
        self._validate_location(area, road)
        summary = scoring.location_summary(self.data, area, road)
        return {
            "area": area,
            "road": road,
            "accident_risk_score": summary["latest_accident_risk_score"],
            "accident_risk_level": summary["latest_accident_risk_level"],
            "historical_avg_accident_risk_score": summary["historical_avg_accident_risk_score"],
            "historical_avg_incident_reports": summary["historical_avg_incident_reports"],
        }

    # ------------------------------------------------------------------
    # Combined dashboard payload
    # ------------------------------------------------------------------
    def full_report(self, area: str, road: str) -> dict:
        self._validate_location(area, road)
        report = {
            "congestion": self.congestion_score(area, road),
            "accident_risk": self.accident_risk(area, road),
            "signal_timing": self.signal_timing(area, road),
        }
        return report

    # ------------------------------------------------------------------
    # AI predictions (forward-looking, no live sensor needed)
    # ------------------------------------------------------------------
    def predict(self, area: str, road: str, weather: str = "Clear", roadwork: str = "No",
                month: int = 6, day_of_week: int = 2) -> dict:
        self._validate_location(area, road)
        if not self.ml_ready:
            raise RuntimeError("ML models are not trained yet. POST /api/train first.")

        row = ml_models.build_feature_row(
            area=area, road=road, weather=weather, roadwork=roadwork,
            month=month, day_of_week=day_of_week,
        )
        congestion = ml_models.predict_congestion(self.congestion_model, row)
        risk_proba = ml_models.predict_risk_probability(self.risk_model, row)

        return {
            "area": area,
            "road": road,
            "scenario": {
                "weather": weather, "roadwork": roadwork,
                "month": month, "day_of_week": day_of_week,
            },
            "predicted_congestion_score": round(congestion, 2),
            "predicted_high_accident_risk_probability": round(risk_proba, 3),
        }