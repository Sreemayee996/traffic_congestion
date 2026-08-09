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
            "Please run data_preprocessing.py first."
        )

    print("\n" + "=" * 70)
    print("WEBSTER SIGNAL TIMING CALCULATION")
    print("=" * 70)

    print("\nInput file loaded successfully.")
    print("Total rows:", len(df))

    required_columns = [
        "Area_ID",
        "Intersection_ID",
        "Traffic_Volume",
        "Phase_1_Flow",
        "Phase_2_Flow",
        "Phase_1_Flow_Ratio",
        "Phase_2_Flow_Ratio",
        "Saturation_Flow",
        "Lost_Time",
        "Baseline_Cycle"
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        print("\nERROR: Missing columns:")
        print(missing_columns)

        print("\nAvailable columns:")
        print(df.columns.tolist())

        raise ValueError(
            "Required Webster input columns are missing."
        )

    numeric_columns = [
        "Traffic_Volume",
        "Phase_1_Flow",
        "Phase_2_Flow",
        "Phase_1_Flow_Ratio",
        "Phase_2_Flow_Ratio",
        "Saturation_Flow",
        "Lost_Time",
        "Baseline_Cycle"
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    nan_rows = df[numeric_columns].isna().any(axis=1).sum()
    if nan_rows > 0:
        print(
            f"\nWARNING: {nan_rows} row(s) have non-numeric or "
            "missing values in required columns. These rows "
            "will be marked INVALID_INPUT in the results."
        )

    if (df["Saturation_Flow"] <= 0).any():
        raise ValueError(
            "Saturation_Flow must be greater than 0."
        )

    if (df["Lost_Time"] < 0).any():
        raise ValueError(
            "Lost_Time cannot be negative."
        )

    df["Webster_Phase_1_Ratio"] = df["Phase_1_Flow_Ratio"]
    df["Webster_Phase_2_Ratio"] = df["Phase_2_Flow_Ratio"]

    df["Webster_Y"] = (
        df["Webster_Phase_1_Ratio"] +
        df["Webster_Phase_2_Ratio"]
    )

    df["Webster_Y"] = df["Webster_Y"].round(4)

    def check_webster_status(y):

        if pd.isna(y):
            return "INVALID_INPUT"

        elif y <= 0:
            return "NO_VALID_DEMAND"

        elif y >= 1:
            return "OVERSATURATED"

        else:
            return "VALID"

    df["Webster_Status"] = (
        df["Webster_Y"].apply(check_webster_status)
    )

    df["Webster_Optimal_Cycle"] = np.where(
        df["Webster_Status"] == "VALID",
        (
            (1.5 * df["Lost_Time"] + 5) /
            (1 - df["Webster_Y"])
        ),
        np.nan
    )

    df["Webster_Optimal_Cycle"] = (
        df["Webster_Optimal_Cycle"].round(2)
    )

    df["Webster_Cycle_Capped"] = (
        df["Webster_Optimal_Cycle"] > MAX_PRACTICAL_CYCLE
    )

    df["Webster_Optimal_Cycle_Practical"] = np.where(
        df["Webster_Cycle_Capped"],
        MAX_PRACTICAL_CYCLE,
        df["Webster_Optimal_Cycle"]
    )

    df["Webster_Available_Green"] = np.where(
        df["Webster_Status"] == "VALID",
        (
            df["Webster_Optimal_Cycle_Practical"] -
            df["Lost_Time"]
        ),
        np.nan
    )

    df["Webster_Available_Green"] = (
        df["Webster_Available_Green"].round(2)
    )

    valid_and_positive_y = (
        (df["Webster_Status"] == "VALID") &
        (df["Webster_Y"] > 0)
    )

    df["Webster_Phase_1_Green"] = np.where(
        valid_and_positive_y,
        (
            df["Webster_Available_Green"] *
            (df["Webster_Phase_1_Ratio"] / df["Webster_Y"])
        ),
        np.nan
    )

    df["Webster_Phase_2_Green"] = np.where(
        valid_and_positive_y,
        (
            df["Webster_Available_Green"] *
            (df["Webster_Phase_2_Ratio"] / df["Webster_Y"])
        ),
        np.nan
    )

    df["Webster_Phase_1_Green"] = (
        df["Webster_Phase_1_Green"].round(2)
    )

    df["Webster_Phase_2_Green"] = (
        df["Webster_Phase_2_Green"].round(2)
    )

    df["Cycle_Change_Seconds"] = np.where(
        df["Webster_Status"] == "VALID",
        (
            df["Webster_Optimal_Cycle_Practical"] -
            df["Baseline_Cycle"]
        ),
        np.nan
    )

    df["Cycle_Change_Seconds"] = (
        df["Cycle_Change_Seconds"].round(2)
    )

    if "Phase_1_Baseline_Green" in df.columns:
        df["Phase_1_Green_Change"] = np.where(
            df["Webster_Status"] == "VALID",
            (
                df["Webster_Phase_1_Green"] -
                df["Phase_1_Baseline_Green"]
            ),
            np.nan
        )
        df["Phase_1_Green_Change"] = (
            df["Phase_1_Green_Change"].round(2)
        )

    if "Phase_2_Baseline_Green" in df.columns:
        df["Phase_2_Green_Change"] = np.where(
            df["Webster_Status"] == "VALID",
            (
                df["Webster_Phase_2_Green"] -
                df["Phase_2_Baseline_Green"]
            ),
            np.nan
        )
        df["Phase_2_Green_Change"] = (
            df["Phase_2_Green_Change"].round(2)
        )

    def get_decision_priority(row):

        webster_status = row["Webster_Status"]

        congestion = str(
            row.get("Calculated_Congestion_Level", "")
        ).upper()

        bottleneck = str(
            row.get("Bottleneck_Status", "")
        ).upper()

        incident = str(
            row.get("Incident_Status", "")
        ).upper()

        peak = str(
            row.get("Peak_Traffic_Status", "")
        ).upper()

        if webster_status == "OVERSATURATED":
            if congestion == "SEVERE" or "SEVERE" in bottleneck:
                return "CRITICAL"
            return "HIGH"

        if webster_status in ["INVALID_INPUT", "NO_VALID_DEMAND"]:
            return "REVIEW_REQUIRED"

        if congestion == "SEVERE":
            return "HIGH"

        if (
            congestion == "HIGH" and
            (
                "BOTTLENECK" in bottleneck or
                incident != "NO_INCIDENT" or
                peak == "PEAK_TRAFFIC"
            )
        ):
            return "HIGH"

        if congestion in ["HIGH", "MODERATE"]:
            return "MEDIUM"

        return "LOW"

    df["Decision_Priority"] = df.apply(
        get_decision_priority, axis=1
    )

    def get_decision_status(row):

        if row["Webster_Status"] == "VALID":
            return "WEBSTER_TIMING_AVAILABLE"
        elif row["Webster_Status"] == "OVERSATURATED":
            return "CAPACITY_OR_NETWORK_REVIEW_REQUIRED"
        else:
            return "INPUT_REVIEW_REQUIRED"

    df["Decision_Status"] = df.apply(
        get_decision_status, axis=1
    )

    result_columns = []

    id_columns = ["Area_ID", "Intersection_ID"]
    for column in id_columns:
        if column in df.columns:
            result_columns.append(column)

    original_traffic_columns = [
        "Traffic_Volume", "Road_Capacity_Utilization_Pct",
        "Estimated_Road_Capacity", "Average_Speed",
        "Travel_Time_Index", "Congestion_Level", "Incident_Level"
    ]
    for column in original_traffic_columns:
        if column in df.columns:
            result_columns.append(column)

    congestion_columns = [
        "Capacity_Utilization", "Bottleneck_Status",
        "Speed_Condition", "Travel_Delay_Condition",
        "Incident_Status", "Accident_Flag",
        "Peak_Traffic_Status", "Congestion_Score",
        "Calculated_Congestion_Level", "Congestion_Reasons"
    ]
    for column in congestion_columns:
        if column in df.columns:
            result_columns.append(column)

    lane_columns = ["Estimated_Lanes", "Saturation_Flow"]
    for column in lane_columns:
        if column in df.columns:
            result_columns.append(column)

    phase_columns = [
        "Number_of_Phases", "Phase_1_Flow", "Phase_2_Flow",
        "Phase_1_Flow_Ratio", "Phase_2_Flow_Ratio"
    ]
    for column in phase_columns:
        if column in df.columns:
            result_columns.append(column)

    baseline_columns = [
        "Baseline_Cycle", "Lost_Time", "Available_Green_Time",
        "Phase_1_Baseline_Green", "Phase_2_Baseline_Green"
    ]
    for column in baseline_columns:
        if column in df.columns:
            result_columns.append(column)

    webster_result_columns = [
        "Webster_Y", "Webster_Status",
        "Webster_Optimal_Cycle", "Webster_Cycle_Capped",
        "Webster_Optimal_Cycle_Practical",
        "Webster_Available_Green",
        "Webster_Phase_1_Green", "Webster_Phase_2_Green",
        "Cycle_Change_Seconds"
    ]
    for column in webster_result_columns:
        if column in df.columns:
            result_columns.append(column)

    green_change_columns = [
        "Phase_1_Green_Change", "Phase_2_Green_Change"
    ]
    for column in green_change_columns:
        if column in df.columns:
            result_columns.append(column)

    decision_columns = ["Decision_Priority", "Decision_Status"]
    for column in decision_columns:
        if column in df.columns:
            result_columns.append(column)

    result_columns = list(dict.fromkeys(result_columns))

    webster_results = df[result_columns].copy()

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    webster_results.to_csv(OUTPUT_FILE, index=False)

    print("\n" + "=" * 70)
    print("WEBSTER RESULTS CREATED SUCCESSFULLY")
    print("=" * 70)

    print("\nOutput file:")
    print(OUTPUT_FILE)

    print("\nTotal rows:")
    print(len(webster_results))

    print("\nWebster Status Summary:")
    print(webster_results["Webster_Status"].value_counts(dropna=False))

    if "Webster_Cycle_Capped" in webster_results.columns:
        capped_count = webster_results["Webster_Cycle_Capped"].sum()
        print(
            f"\nRows with cycle capped at {MAX_PRACTICAL_CYCLE}s "
            f"(near-saturation, Y close to 1): {capped_count}"
        )

    print("\nDecision Priority Summary:")
    print(webster_results["Decision_Priority"].value_counts(dropna=False))

    print("\nFirst 5 Webster Results:")

    display_columns = [
        "Area_ID", "Intersection_ID", "Calculated_Congestion_Level",
        "Webster_Y", "Webster_Status", "Baseline_Cycle",
        "Webster_Optimal_Cycle_Practical", "Webster_Phase_1_Green",
        "Webster_Phase_2_Green", "Decision_Priority", "Decision_Status"
    ]

    display_columns = [
        column for column in display_columns
        if column in webster_results.columns
    ]

    print(webster_results[display_columns].head())

    print("\nSUCCESS!")
    print("webster_results_full.csv is ready for simulation.py")

if __name__ == "__main__":

    try:
        main()
    except Exception as error:
        print("\n" + "=" * 70)
        print("SCRIPT FAILED")
        print("=" * 70)
        print(f"\nError: {error}")
        raise