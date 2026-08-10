"""
data_preprocessing.py
----------------------
General cleaning module for the FULL Bangalore traffic dataset.
Keeps original column names as-is (scoring.py and ml_models.py both
expect the raw headers, e.g. "Area Name", "Road/Intersection Name",
"Congestion Level"). This is NOT the same as webster_preprocessing.py,
which builds a separate dataset with renamed columns (Area_ID,
Intersection_ID, etc.) specifically for the Webster signal-timing
calculation.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_FILE = BASE_DIR / "data" / "Banglore_traffic_Dataset Working.csv"

WEATHER_RISK_WEIGHT = {
    "Clear": 0.10,
    "Overcast": 0.30,
    "Windy": 0.35,
    "Rain": 0.60,
    "Fog": 0.70,
}

def load_clean_data(input_file: Path = INPUT_FILE) -> pd.DataFrame:
    df = pd.read_csv(input_file)

    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")

    numeric_columns = [
        "Traffic Volume", "Average Speed", "Travel Time Index",
        "Congestion Level", "Road Capacity Utilization", "Incident Reports",
        "Environmental Impact", "Public Transport Usage",
        "Traffic Signal Compliance", "Parking Usage",
        "Pedestrian and Cyclist Count",
    ]

    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(subset=["Date", "Area Name", "Road/Intersection Name"])

    for column in numeric_columns:
        if column in df.columns:
            df[column] = df[column].fillna(df[column].median())

    df["Weather Conditions"] = df["Weather Conditions"].fillna("Clear")
    df["Roadwork and Construction Activity"] = (
        df["Roadwork and Construction Activity"].fillna("No")
    )

    df["Month"] = df["Date"].dt.month
    df["DayOfWeek"] = df["Date"].dt.dayofweek
    df["IsWeekend"] = df["DayOfWeek"].isin([5, 6]).astype(int)

    return df.reset_index(drop=True)

load_clean = load_clean_data

def get_location_catalog(df: pd.DataFrame) -> pd.DataFrame:
    catalog = (
        df[["Area Name", "Road/Intersection Name"]]
        .drop_duplicates()
        .sort_values(["Area Name", "Road/Intersection Name"])
        .reset_index(drop=True)
    )
    return catalog

if __name__ == "__main__":
    cleaned = load_clean_data()
    print("Rows after cleaning:", len(cleaned))
    print("Date range:", cleaned["Date"].min(), "to", cleaned["Date"].max())
    catalog = get_location_catalog(cleaned)
    print("Unique locations:", len(catalog))