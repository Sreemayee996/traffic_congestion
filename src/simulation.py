"""
simulation.py
--------------
High-level facade used by app.py. Loads + scores the data once,
holds the trained ML models in memory, and exposes simple methods
that map 1:1 onto the API endpoints. Keeping this orchestration out
of app.py keeps the Flask layer thin (routing + HTTP concerns only).

This version also loads Pipeline A's batch outputs (webster_results_full.csv
and simulation_summary.csv) so the live API can serve the more accurate,
pre-calculated Webster recommendations and full per-location detail,
instead of only the simplified on-the-fly calculation in webster.py.
"""

from __future__ import annotations

import os

import pandas as pd

from . import ml_models, scoring, webster
from .data_preprocessing import get_location_catalog, load_clean

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
WEBSTER_RESULTS_FILE = os.path.join(DATA_DIR, "webster_results_full.csv")
BATCH_SUMMARY_FILE = os.path.join(DATA_DIR, "simulation_summary.csv")


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

        # Pipeline A outputs (batch CSVs) — loaded once at startup, optional.
        self.webster_results = self._load_csv_safely(WEBSTER_RESULTS_FILE)
        self.batch_summary = self._load_csv_safely(BATCH_SUMMARY_FILE)

    def _load_csv_safely(self, path: str):
        try:
            return pd.read_csv(path)
        except FileNotFoundError:
            return None

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

    @property
    def batch_data_ready(self) -> bool:
        return self.webster_results is not None and self.batch_summary is not None

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

    def congestion_score(self, area: str, road: str) -> dict:
        self._validate_location(area, road)
        return scoring.location_summary(self.data, area, road)

    def bottlenecks(self, top_n: int = 10) -> list[dict]:
        ranking = scoring.bottleneck_ranking(self.data, top_n=top_n)
        return ranking.to_dict(orient="records")

    def signal_timing(self, area: str, road: str, num_phases: int = 4, lanes_per_phase: int = 2,
                       peak_hour_factor: float = 0.09) -> dict:
        self._validate_location(area, road)
        subset = self.data[
            (self.data["Area Name"].str.lower() == area.lower())
            & (self.data["Road/Intersection Name"].str.lower() == road.lower())
        ]
        avg_daily_volume = float(subset["Traffic Volume"].mean())
        avg_peak_hour_volume = avg_daily_volume * peak_hour_factor
        avg_utilization = float(subset["Road Capacity Utilization"].mean())

        result = webster.estimate_signal_timing(
            traffic_volume_vph=avg_peak_hour_volume,
            road_capacity_utilization_pct=avg_utilization,
            num_phases=num_phases,
            lanes_per_phase=lanes_per_phase,
        )
        out = result.to_dict()
        out.update({
            "area": area,
            "road": road,
            "avg_daily_traffic_volume": round(avg_daily_volume, 1),
            "peak_hour_factor": peak_hour_factor,
            "estimated_peak_hour_volume": round(avg_peak_hour_volume, 1),
            "avg_road_capacity_utilization_pct": round(avg_utilization, 1),
        })
        return out

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

    def full_report(self, area: str, road: str) -> dict:
        self._validate_location(area, road)
        report = {
            "congestion": self.congestion_score(area, road),
            "accident_risk": self.accident_risk(area, road),
            "signal_timing": self.signal_timing(area, road),
        }
        return report

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

    # ------------------------------------------------------------------
    # Pipeline A integration — batch/Webster results served through the
    # live API, instead of sitting unused as CSV files.
    # ------------------------------------------------------------------

    def _match_webster_results(self, area_id: str, intersection_id: str) -> pd.DataFrame:
        if self.webster_results is None:
            raise RuntimeError(
                "webster_results_full.csv not found. Run webster_preprocessing.py "
                "then webster_batch.py first."
            )
        match = self.webster_results[
            (self.webster_results["Area_ID"].astype(str).str.upper() == area_id.upper())
            & (self.webster_results["Intersection_ID"].astype(str).str.upper() == intersection_id.upper())
        ]
        if match.empty:
            raise ValueError(f"No batch Webster data for ({area_id}, {intersection_id}).")
        return match

    def location_full_detail(self, area_id: str, intersection_id: str) -> list[dict]:
        """Returns every calculated column, every row (e.g. every day/record)
        for one intersection from the Pipeline A batch results — full raw
        detail, not a summary."""
        match = self._match_webster_results(area_id, intersection_id)
        return match.to_dict(orient="records")

    def improved_signal_timing(self, area_id: str, intersection_id: str) -> dict:
        """Signal timing recommendation using Pipeline A's properly
        calculated lanes/saturation flow, instead of webster.py's
        guessed defaults (num_phases=4, lanes_per_phase=2)."""
        match = self._match_webster_results(area_id, intersection_id)
        row = match.iloc[0]
        return {
            "area_id": row["Area_ID"],
            "intersection_id": row["Intersection_ID"],
            "estimated_lanes": row.get("Estimated_Lanes"),
            "saturation_flow": row.get("Saturation_Flow"),
            "webster_status": row.get("Webster_Status"),
            "recommended_cycle_sec": row.get("Webster_Optimal_Cycle_Practical"),
            "recommended_phase_1_green_sec": row.get("Webster_Phase_1_Green"),
            "recommended_phase_2_green_sec": row.get("Webster_Phase_2_Green"),
            "baseline_cycle_sec": row.get("Baseline_Cycle"),
            "cycle_change_seconds": row.get("Cycle_Change_Seconds"),
            "decision_priority": row.get("Decision_Priority"),
        }

    def batch_recommendation(self, area_id: str, intersection_id: str) -> dict:
        """Before/after comparison for one intersection, using the
        pre-computed summary from batch_report.py (simulation_summary.csv)."""
        if self.batch_summary is None:
            raise RuntimeError(
                "simulation_summary.csv not found. Run webster_preprocessing.py, "
                "webster_batch.py, then batch_report.py first."
            )
        match = self.batch_summary[
            (self.batch_summary["Area_ID"].astype(str).str.upper() == area_id.upper())
            & (self.batch_summary["Intersection_ID"].astype(str).str.upper() == intersection_id.upper())
        ]
        if match.empty:
            raise ValueError(f"No batch summary for ({area_id}, {intersection_id}).")

        row = match.iloc[0]
        return {
            "area_id": row["Area_ID"],
            "intersection_id": row["Intersection_ID"],
            "before": {
                "congestion_score": row["Congestion_Score"],
                "congestion_level": row["Most_Common_Congestion_Level"],
                "bottleneck_rank": int(row["Bottleneck_Rank"]),
                "bottleneck_index": row["Bottleneck_Index"],
                "pct_severe_bottleneck_days": row["Pct_Severe_Bottleneck_Days"],
            },
            "after_recommendation": {
                "cycle_length_sec": row["Recommended_Cycle_Length_Sec"],
                "phase_1_green_sec": row["Recommended_Phase_1_Green_Sec"],
                "phase_2_green_sec": row["Recommended_Phase_2_Green_Sec"],
            },
            "accident_risk_score": row["Accident_Risk_Score"],
            "accident_risk_level": row["Accident_Risk_Level"],
            "decision_priority": row["Most_Common_Decision_Priority"],
        }

    def bottleneck_ranking_batch(self, top_n: int = 10) -> list[dict]:
        """Bottleneck ranking sourced from Pipeline A's batch summary
        (based on the full historical dataset), as an alternative to
        the live-computed scoring.bottleneck_ranking()."""
        if self.batch_summary is None:
            raise RuntimeError(
                "simulation_summary.csv not found. Run webster_preprocessing.py, "
                "webster_batch.py, then batch_report.py first."
            )
        top_n = max(1, min(top_n, len(self.batch_summary)))
        ranked = self.batch_summary.sort_values("Bottleneck_Index", ascending=False).head(top_n)
        return ranked.to_dict(orient="records")