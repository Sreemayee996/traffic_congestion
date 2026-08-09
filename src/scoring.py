"""
scoring.py
----------
Rule-based (transparent, explainable) scoring engine built on top of
the cleaned traffic DataFrame. Everything here is a deterministic
formula over the raw features -- no black box -- so the numbers can
be defended to a stakeholder. The ML layer (ml_models.py) sits on top
of this for forward-looking predictions.

Three things are produced:

1. Congestion Score       (0-100, higher = worse)
2. Bottleneck ranking     (locations ordered by how bad + how often)
3. Accident-Prone Risk    (0-100 + Low/Medium/High/Critical label)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

try:
    from .data_preprocessing import WEATHER_RISK_WEIGHT
except ImportError:
    from data_preprocessing import WEATHER_RISK_WEIGHT

CONGESTION_WEIGHTS = {
    "congestion_level": 0.35,
    "capacity_utilization": 0.25,
    "travel_time_index": 0.20,
    "speed_inverse": 0.20,
}

RISK_WEIGHTS = {
    "incident_reports": 0.35,
    "weather": 0.20,
    "signal_compliance": 0.20,
    "pedestrian_exposure": 0.15,
    "roadwork": 0.10,
}

def _normalize(series: pd.Series, lo: float | None = None, hi: float | None = None) -> pd.Series:
    """Min-max normalize to 0-100, clipped. lo/hi override the data range
    so scores stay comparable across different slices of the data."""
    lo = series.min() if lo is None else lo
    hi = series.max() if hi is None else hi
    if hi == lo:
        return pd.Series(np.zeros(len(series)), index=series.index)
    out = (series - lo) / (hi - lo) * 100
    return out.clip(0, 100)

def add_congestion_score(df: pd.DataFrame) -> pd.DataFrame:
    """Adds a Congestion Score column (0-100, higher = more congested)."""
    df = df.copy()

    tti_norm = _normalize(df["Travel Time Index"], lo=1.0, hi=df["Travel Time Index"].quantile(0.99))
    speed_norm = 100 - _normalize(df["Average Speed"])

    df["Congestion Score"] = (
        CONGESTION_WEIGHTS["congestion_level"] * df["Congestion Level"]
        + CONGESTION_WEIGHTS["capacity_utilization"] * df["Road Capacity Utilization"]
        + CONGESTION_WEIGHTS["travel_time_index"] * tti_norm
        + CONGESTION_WEIGHTS["speed_inverse"] * speed_norm
    ).round(2)

    df["Congestion Category"] = pd.cut(
        df["Congestion Score"],
        bins=[-1, 40, 60, 80, 101],
        labels=["Low", "Moderate", "High", "Severe"],
    )
    return df

def add_accident_risk(df: pd.DataFrame) -> pd.DataFrame:
    """Adds Accident Risk Score (0-100) and Accident Risk Level columns."""
    df = df.copy()

    incident_norm = _normalize(df["Incident Reports"], lo=0, hi=df["Incident Reports"].quantile(0.99))
    weather_norm = df["Weather Conditions"].map(WEATHER_RISK_WEIGHT).fillna(0.25) * 100
    compliance_risk = 100 - df["Traffic Signal Compliance"]
    pedestrian_norm = _normalize(df["Pedestrian and Cyclist Count"])
    roadwork_risk = np.where(df["Roadwork and Construction Activity"].str.lower() == "yes", 100, 0)

    df["Accident Risk Score"] = (
        RISK_WEIGHTS["incident_reports"] * incident_norm
        + RISK_WEIGHTS["weather"] * weather_norm
        + RISK_WEIGHTS["signal_compliance"] * compliance_risk
        + RISK_WEIGHTS["pedestrian_exposure"] * pedestrian_norm
        + RISK_WEIGHTS["roadwork"] * roadwork_risk
    ).round(2)

    df["Accident Risk Level"] = pd.cut(
        df["Accident Risk Score"],
        bins=[-1, 30, 50, 70, 101],
        labels=["Low", "Medium", "High", "Critical"],
    )
    return df

def score_all(df: pd.DataFrame) -> pd.DataFrame:
    """Run every scoring pass in one call."""
    df = add_congestion_score(df)
    df = add_accident_risk(df)
    return df

def bottleneck_ranking(scored_df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """
    Rank (Area, Road) locations by a Bottleneck Index that rewards both
    severity (average congestion) and frequency (how often it tips
    into High/Severe congestion) -- a road that's occasionally bad is a
    smaller bottleneck than one that's consistently bad.
    """
    grouped = scored_df.groupby(["Area Name", "Road/Intersection Name"]).agg(
        avg_congestion_score=("Congestion Score", "mean"),
        p90_congestion_score=("Congestion Score", lambda s: np.percentile(s, 90)),
        pct_severe=("Congestion Category", lambda s: (s == "Severe").mean() * 100),
        avg_speed=("Average Speed", "mean"),
        avg_incident_reports=("Incident Reports", "mean"),
        avg_accident_risk=("Accident Risk Score", "mean"),
        observations=("Congestion Score", "size"),
    ).reset_index()

    grouped["Bottleneck Index"] = (
        0.55 * grouped["avg_congestion_score"]
        + 0.25 * grouped["p90_congestion_score"]
        + 0.20 * grouped["pct_severe"]
    ).round(2)

    grouped = grouped.sort_values("Bottleneck Index", ascending=False).reset_index(drop=True)
    grouped.insert(0, "Rank", grouped.index + 1)
    for col in ["avg_congestion_score", "p90_congestion_score", "pct_severe", "avg_speed",
                "avg_incident_reports", "avg_accident_risk"]:
        grouped[col] = grouped[col].round(2)

    return grouped.head(top_n)

def location_summary(scored_df: pd.DataFrame, area: str, road: str) -> dict:
    """Single-location summary combining congestion + risk, latest + historical."""
    subset = scored_df[
        (scored_df["Area Name"].str.lower() == area.lower())
        & (scored_df["Road/Intersection Name"].str.lower() == road.lower())
    ]
    if subset.empty:
        return {}

    latest = subset.sort_values("Date").iloc[-1]
    return {
        "area": area,
        "road": road,
        "latest_date": str(latest["Date"].date()),
        "latest_congestion_score": float(latest["Congestion Score"]),
        "latest_congestion_category": str(latest["Congestion Category"]),
        "latest_accident_risk_score": float(latest["Accident Risk Score"]),
        "latest_accident_risk_level": str(latest["Accident Risk Level"]),
        "historical_avg_congestion_score": round(float(subset["Congestion Score"].mean()), 2),
        "historical_avg_accident_risk_score": round(float(subset["Accident Risk Score"].mean()), 2),
        "historical_avg_speed_kmph": round(float(subset["Average Speed"].mean()), 2),
        "historical_avg_incident_reports": round(float(subset["Incident Reports"].mean()), 2),
        "observations": int(len(subset)),
    }