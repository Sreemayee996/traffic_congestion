import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = BASE_DIR / "data" / "webster_input_full.csv"
OUTPUT_FILE = BASE_DIR / "data" / "webster_results_full.csv"

MAX_PRACTICAL_CYCLE = 180

def main():

    try:
        df = pd.read_csv(INPUT_FILE)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"\nInput file not found:\n{INPUT_FILE}\n"
            "Please run webster_preprocessing.py first."
        )

    print("\n" + "=" * 70)
    print("WEBSTER SIGNAL TIMING CALCULATION")
    print("=" * 70)

    print("\nInput file loaded successfully.")
    print("Total rows:", len(df))

    required_columns = [
        "Area_ID", "Intersection_ID", "Traffic_Volume", "Phase_1_Flow",
        "Phase_2_Flow", "Phase_1_Flow_Ratio", "Phase_2_Flow_Ratio",
        "Saturation_Flow", "Lost_Time", "Baseline_Cycle"
    ]

    missing_columns = [c for c in required_columns if c not in df.columns]
    if missing_columns:
        print("\nERROR: Missing columns:", missing_columns)
        raise ValueError("Required Webster input columns are missing.")

    numeric_columns = [
        "Traffic_Volume", "Phase_1_Flow", "Phase_2_Flow", "Phase_1_Flow_Ratio",
        "Phase_2_Flow_Ratio", "Saturation_Flow", "Lost_Time", "Baseline_Cycle"
    ]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    if (df["Saturation_Flow"] <= 0).any():
        raise ValueError("Saturation_Flow must be greater than 0.")
    if (df["Lost_Time"] < 0).any():
        raise ValueError("Lost_Time cannot be negative.")

    df["Webster_Phase_1_Ratio"] = df["Phase_1_Flow_Ratio"]
    df["Webster_Phase_2_Ratio"] = df["Phase_2_Flow_Ratio"]
    df["Webster_Y"] = (df["Webster_Phase_1_Ratio"] + df["Webster_Phase_2_Ratio"]).round(4)

    def check_webster_status(y):
        if pd.isna(y):
            return "INVALID_INPUT"
        elif y <= 0:
            return "NO_VALID_DEMAND"
        elif y >= 1:
            return "OVERSATURATED"
        else:
            return "VALID"

    df["Webster_Status"] = df["Webster_Y"].apply(check_webster_status)

    df["Webster_Optimal_Cycle"] = np.round(np.where(
        df["Webster_Status"] == "VALID",
        (1.5 * df["Lost_Time"] + 5) / (1 - df["Webster_Y"]),
        np.nan
    ), 2)

    df["Webster_Cycle_Capped"] = df["Webster_Optimal_Cycle"] > MAX_PRACTICAL_CYCLE
    df["Webster_Optimal_Cycle_Practical"] = np.where(
        df["Webster_Cycle_Capped"], MAX_PRACTICAL_CYCLE, df["Webster_Optimal_Cycle"]
    )

    df["Webster_Available_Green"] = np.round(np.where(
        df["Webster_Status"] == "VALID",
        df["Webster_Optimal_Cycle_Practical"] - df["Lost_Time"], np.nan
    ), 2)

    valid_and_positive_y = (df["Webster_Status"] == "VALID") & (df["Webster_Y"] > 0)

    df["Webster_Phase_1_Green"] = np.round(np.where(
        valid_and_positive_y,
        df["Webster_Available_Green"] * (df["Webster_Phase_1_Ratio"] / df["Webster_Y"]),
        np.nan
    ), 2)

    df["Webster_Phase_2_Green"] = np.round(np.where(
        valid_and_positive_y,
        df["Webster_Available_Green"] * (df["Webster_Phase_2_Ratio"] / df["Webster_Y"]),
        np.nan
    ), 2)

    df["Cycle_Change_Seconds"] = np.round(np.where(
        df["Webster_Status"] == "VALID",
        df["Webster_Optimal_Cycle_Practical"] - df["Baseline_Cycle"], np.nan
    ), 2)

    if "Phase_1_Baseline_Green" in df.columns:
        df["Phase_1_Green_Change"] = np.round(np.where(
            df["Webster_Status"] == "VALID",
            df["Webster_Phase_1_Green"] - df["Phase_1_Baseline_Green"], np.nan
        ), 2)

    if "Phase_2_Baseline_Green" in df.columns:
        df["Phase_2_Green_Change"] = np.round(np.where(
            df["Webster_Status"] == "VALID",
            df["Webster_Phase_2_Green"] - df["Phase_2_Baseline_Green"], np.nan
        ), 2)

    def get_decision_priority(row):
        webster_status = row["Webster_Status"]
        congestion = str(row.get("Calculated_Congestion_Level", "")).upper()
        bottleneck = str(row.get("Bottleneck_Status", "")).upper()
        incident = str(row.get("Incident_Status", "")).upper()
        peak = str(row.get("Peak_Traffic_Status", "")).upper()

        if webster_status == "OVERSATURATED":
            if congestion == "SEVERE" or "SEVERE" in bottleneck:
                return "CRITICAL"
            return "HIGH"
        if webster_status in ["INVALID_INPUT", "NO_VALID_DEMAND"]:
            return "REVIEW_REQUIRED"
        if congestion == "SEVERE":
            return "HIGH"
        if congestion == "HIGH" and ("BOTTLENECK" in bottleneck or incident != "NO_INCIDENT" or peak == "PEAK_TRAFFIC"):
            return "HIGH"
        if congestion in ["HIGH", "MODERATE"]:
            return "MEDIUM"
        return "LOW"

    df["Decision_Priority"] = df.apply(get_decision_priority, axis=1)

    df["Decision_Status"] = df["Webster_Status"].map({
        "VALID": "WEBSTER_TIMING_AVAILABLE",
        "OVERSATURATED": "CAPACITY_OR_NETWORK_REVIEW_REQUIRED"
    }).fillna("INPUT_REVIEW_REQUIRED")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)

    print("\nSUCCESS! Rows:", len(df))
    print(df["Webster_Status"].value_counts(dropna=False))
    print("\nDecision Priority Summary:")
    print(df["Decision_Priority"].value_counts(dropna=False))
    print("\nwebster_results_full.csv is ready for batch_report.py")

    return df

if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print("\nSCRIPT FAILED:", error)
        raise