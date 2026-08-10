import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = BASE_DIR / "data" / "webster_results_full.csv"
OUTPUT_FILE = BASE_DIR / "data" / "simulation_summary.csv"

ACCIDENT_RISK_LOW_MAX = 25
ACCIDENT_RISK_MEDIUM_MAX = 50
ACCIDENT_RISK_HIGH_MAX = 75

def main():
    try:
        df = pd.read_csv(INPUT_FILE)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"\nInput file not found:\n{INPUT_FILE}\n"
            "Please run webster_preprocessing.py then webster_batch.py first."
        )

    print("\n" + "=" * 70)
    print("TRAFFIC SIMULATION SUMMARY")
    print("=" * 70)
    print("\nInput file loaded successfully.")
    print("Total rows:", len(df))

    required_columns = [
        "Area_ID", "Intersection_ID", "Traffic_Volume", "Average_Speed",
        "Congestion_Score", "Calculated_Congestion_Level", "Bottleneck_Status",
        "Incident_Level", "Accident_Flag", "Webster_Optimal_Cycle_Practical",
        "Webster_Phase_1_Green", "Webster_Phase_2_Green", "Decision_Priority",
    ]

    missing_columns = [c for c in required_columns if c not in df.columns]
    if missing_columns:
        print("\nERROR: Missing columns:", missing_columns)
        raise ValueError("Required simulation input columns are missing.")

    def pct_true(series, condition):
        return round((series == condition).mean() * 100, 2)

    def mode_or_first(series):
        modes = series.mode()
        return modes.iloc[0] if not modes.empty else series.iloc[0]

    summary_rows = []
    grouped = df.groupby(["Area_ID", "Intersection_ID"])

    for (area_id, intersection_id), group in grouped:
        avg_congestion_score = round(group["Congestion_Score"].mean(), 2)
        p90_congestion_score = round(np.percentile(group["Congestion_Score"], 90), 2)
        pct_severe_bottleneck = pct_true(group["Bottleneck_Status"], "SEVERE_BOTTLENECK")
        pct_bottleneck_any = round(
            group["Bottleneck_Status"].isin(["SEVERE_BOTTLENECK", "BOTTLENECK"]).mean() * 100, 2
        )

        bottleneck_index = round(
            0.5 * avg_congestion_score + 0.3 * p90_congestion_score + 0.2 * pct_severe_bottleneck, 2
        )

        avg_incident_level = round(group["Incident_Level"].mean(), 2)
        pct_accident_days = pct_true(group["Accident_Flag"], "YES")

        accident_risk_score = round(
            (avg_incident_level / 3.0) * 60 + (pct_accident_days / 100.0) * 40, 2
        )
        accident_risk_score = min(max(accident_risk_score, 0), 100)

        if accident_risk_score < ACCIDENT_RISK_LOW_MAX:
            accident_risk_level = "LOW"
        elif accident_risk_score < ACCIDENT_RISK_MEDIUM_MAX:
            accident_risk_level = "MEDIUM"
        elif accident_risk_score < ACCIDENT_RISK_HIGH_MAX:
            accident_risk_level = "HIGH"
        else:
            accident_risk_level = "CRITICAL"

        recommended_cycle = round(group["Webster_Optimal_Cycle_Practical"].mean(), 1)
        recommended_phase_1_green = round(group["Webster_Phase_1_Green"].mean(), 1)
        recommended_phase_2_green = round(group["Webster_Phase_2_Green"].mean(), 1)

        summary_rows.append({
            "Area_ID": area_id,
            "Intersection_ID": intersection_id,
            "Observations": len(group),
            "Avg_Traffic_Volume": round(group["Traffic_Volume"].mean(), 1),
            "Avg_Speed": round(group["Average_Speed"].mean(), 1),
            "Congestion_Score": avg_congestion_score,
            "Congestion_Score_P90": p90_congestion_score,
            "Most_Common_Congestion_Level": mode_or_first(group["Calculated_Congestion_Level"]),
            "Pct_Severe_Bottleneck_Days": pct_severe_bottleneck,
            "Pct_Any_Bottleneck_Days": pct_bottleneck_any,
            "Bottleneck_Index": bottleneck_index,
            "Avg_Incident_Level": avg_incident_level,
            "Pct_Accident_Days": pct_accident_days,
            "Accident_Risk_Score": accident_risk_score,
            "Accident_Risk_Level": accident_risk_level,
            "Recommended_Cycle_Length_Sec": recommended_cycle,
            "Recommended_Phase_1_Green_Sec": recommended_phase_1_green,
            "Recommended_Phase_2_Green_Sec": recommended_phase_2_green,
            "Most_Common_Decision_Priority": mode_or_first(group["Decision_Priority"]),
        })

    summary = pd.DataFrame(summary_rows)
    summary = summary.sort_values("Bottleneck_Index", ascending=False).reset_index(drop=True)
    summary.insert(0, "Bottleneck_Rank", summary.index + 1)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUTPUT_FILE, index=False)

    print("\nSUCCESS! Total intersections summarized:", len(summary))
    print("\nTop 5 Bottleneck Intersections:")
    print(summary[["Bottleneck_Rank", "Area_ID", "Intersection_ID", "Congestion_Score",
                    "Bottleneck_Index", "Most_Common_Congestion_Level"]].head(5).to_string(index=False))

    return summary

def get_report(summary, area_id, intersection_id):
    match = summary[
        (summary["Area_ID"].str.upper() == area_id.upper()) &
        (summary["Intersection_ID"].str.upper() == intersection_id.upper())
    ]
    if match.empty:
        return None
    row = match.iloc[0]
    return {
        "area_id": row["Area_ID"],
        "intersection_id": row["Intersection_ID"],
        "congestion_score": row["Congestion_Score"],
        "congestion_level": row["Most_Common_Congestion_Level"],
        "bottleneck_rank": int(row["Bottleneck_Rank"]),
        "bottleneck_index": row["Bottleneck_Index"],
        "pct_severe_bottleneck_days": row["Pct_Severe_Bottleneck_Days"],
        "signal_timing": {
            "cycle_length_sec": row["Recommended_Cycle_Length_Sec"],
            "phase_1_green_sec": row["Recommended_Phase_1_Green_Sec"],
            "phase_2_green_sec": row["Recommended_Phase_2_Green_Sec"],
        },
        "accident_risk_score": row["Accident_Risk_Score"],
        "accident_risk_level": row["Accident_Risk_Level"],
        "decision_priority": row["Most_Common_Decision_Priority"],
    }

def print_report(report):
    if report is None:
        print("No matching intersection found.")
        return
    print("\n" + "-" * 50)
    print(f"REPORT: {report['area_id']} / {report['intersection_id']}")
    print("-" * 50)
    print(f"Congestion Score      : {report['congestion_score']} ({report['congestion_level']})")
    print(f"Bottleneck Rank       : #{report['bottleneck_rank']} (index {report['bottleneck_index']}, "
          f"{report['pct_severe_bottleneck_days']}% severe days)")
    print(f"Signal Timing         : cycle {report['signal_timing']['cycle_length_sec']}s "
          f"| phase 1 green {report['signal_timing']['phase_1_green_sec']}s "
          f"| phase 2 green {report['signal_timing']['phase_2_green_sec']}s")
    print(f"Accident-Prone Risk   : {report['accident_risk_score']} ({report['accident_risk_level']})")
    print(f"Decision Priority     : {report['decision_priority']}")
    print("-" * 50)

def interactive_lookup(summary):
    print("\nType an Intersection_ID (e.g. RIN01) to get its report, or press Enter / type 'exit' to quit.")
    while True:
        try:
            user_input = input("\nIntersection_ID: ").strip()
        except EOFError:
            break
        if not user_input or user_input.lower() == "exit":
            break
        match = summary[summary["Intersection_ID"].str.upper() == user_input.upper()]
        if match.empty:
            print("Not found. Valid IDs:", sorted(summary["Intersection_ID"].unique().tolist()))
            continue
        area_id = match.iloc[0]["Area_ID"]
        report = get_report(summary, area_id, user_input)
        print_report(report)

if __name__ == "__main__":
    try:
        result_summary = main()
        interactive_lookup(result_summary)
    except Exception as error:
        print("\nSCRIPT FAILED:", error)
        raise