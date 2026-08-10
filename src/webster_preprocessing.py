import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = BASE_DIR / "data" / "Banglore_traffic_Dataset Working.csv"
OUTPUT_FILE = BASE_DIR / "data" / "webster_input_full.csv"

df = pd.read_csv(INPUT_FILE)

print("\nOriginal column names:")
print(df.columns.tolist())

print("\nOriginal number of rows:", len(df))

column_rename_map = {
    "Area Name ID": "Area_ID",
    "RIN ID": "Intersection_ID",
    "Traffic Volume": "Traffic_Volume",
    "Average Speed": "Average_Speed",
    "Travel Time Index": "Travel_Time_Index",
    "Congestion Level": "Congestion_Level",
    "Road Capacity Utilization": "Road_Capacity_Utilization_Pct",
    "Incident Reports": "Incident_Level",
}

df = df.rename(columns=column_rename_map)

df = df.reset_index(drop=True).copy()

print("Total rows used:", len(df))

required_columns = [
    "Area_ID",
    "Intersection_ID",
    "Traffic_Volume",
    "Road_Capacity_Utilization_Pct",
    "Average_Speed",
    "Travel_Time_Index",
    "Congestion_Level",
    "Incident_Level"
]

missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:
    print("\nERROR: These columns were not found:")
    print(missing_columns)
    print("\nAvailable columns:")
    print(df.columns.tolist())
    raise ValueError("Please update the column names in the code.")

# Keep the human-readable names too (if present in the raw data), so
# reports/dashboards can display "Indiranagar / 100ft Road" instead of
# just raw ID codes like "AREA01" / "RIN01".
NAME_COLUMNS_AVAILABLE = []
if "Area Name" in df.columns:
    NAME_COLUMNS_AVAILABLE.append("Area Name")
else:
    print("\nWARNING: 'Area Name' column not found in raw data — "
          "output will only have Area_ID, no readable area name.")

if "Road/Intersection Name" in df.columns:
    NAME_COLUMNS_AVAILABLE.append("Road/Intersection Name")
else:
    print("\nWARNING: 'Road/Intersection Name' column not found in raw data — "
          "output will only have Intersection_ID, no readable road name.")

if "Date" in df.columns:
    NAME_COLUMNS_AVAILABLE.append("Date")
else:
    print("\nWARNING: 'Date' column not found in raw data — "
          "trend charts over time won't be available in the dashboard.")

print("Unique intersections in dataset:", df["Intersection_ID"].nunique())

numeric_columns = [
    "Traffic_Volume",
    "Road_Capacity_Utilization_Pct",
    "Average_Speed",
    "Travel_Time_Index",
    "Incident_Level"
]

for column in numeric_columns:
    df[column] = pd.to_numeric(df[column], errors="coerce")

for column in numeric_columns:
    df[column] = df[column].fillna(df[column].median())

df["Capacity_Utilization"] = (df["Road_Capacity_Utilization_Pct"] / 100)

safe_utilization = df["Capacity_Utilization"].replace(0, np.nan)
df["Estimated_Road_Capacity"] = (df["Traffic_Volume"] / safe_utilization)
df["Estimated_Road_Capacity"] = df["Estimated_Road_Capacity"].fillna(df["Estimated_Road_Capacity"].median())

def detect_bottleneck(utilization):
    if utilization >= 1.00:
        return "SEVERE_BOTTLENECK"
    elif utilization >= 0.90:
        return "BOTTLENECK"
    elif utilization >= 0.75:
        return "AT_RISK"
    else:
        return "NORMAL"

df["Bottleneck_Status"] = df["Capacity_Utilization"].apply(detect_bottleneck)

speed_q25 = df["Average_Speed"].quantile(0.25)
speed_q50 = df["Average_Speed"].quantile(0.50)
speed_q75 = df["Average_Speed"].quantile(0.75)

def detect_speed_condition(speed):
    if speed <= speed_q25:
        return "VERY_LOW_SPEED"
    elif speed <= speed_q50:
        return "LOW_SPEED"
    elif speed <= speed_q75:
        return "NORMAL_SPEED"
    else:
        return "HIGH_SPEED"

df["Speed_Condition"] = df["Average_Speed"].apply(detect_speed_condition)

tti_q25 = df["Travel_Time_Index"].quantile(0.25)
tti_q50 = df["Travel_Time_Index"].quantile(0.50)
tti_q75 = df["Travel_Time_Index"].quantile(0.75)

def detect_travel_delay(tti):
    if tti >= tti_q75:
        return "SEVERE_DELAY"
    elif tti >= tti_q50:
        return "HIGH_DELAY"
    elif tti >= tti_q25:
        return "MODERATE_DELAY"
    else:
        return "LOW_DELAY"

df["Travel_Delay_Condition"] = df["Travel_Time_Index"].apply(detect_travel_delay)

def detect_incident(incident_level):
    if incident_level <= 0:
        return "NO_INCIDENT"
    elif incident_level == 1:
        return "LOW_INCIDENT"
    elif incident_level == 2:
        return "MODERATE_INCIDENT"
    else:
        return "HIGH_INCIDENT"

df["Incident_Status"] = df["Incident_Level"].apply(detect_incident)
df["Accident_Flag"] = np.where(df["Incident_Level"] > 0, "YES", "NO")

traffic_q75 = df["Traffic_Volume"].quantile(0.75)
traffic_q90 = df["Traffic_Volume"].quantile(0.90)

def detect_peak_traffic(volume):
    if volume >= traffic_q90:
        return "PEAK_TRAFFIC"
    elif volume >= traffic_q75:
        return "HIGH_TRAFFIC"
    else:
        return "NORMAL_TRAFFIC"

df["Peak_Traffic_Status"] = df["Traffic_Volume"].apply(detect_peak_traffic)

utilization_score = np.clip(df["Capacity_Utilization"], 0, 1)

speed_max = df["Average_Speed"].max()
speed_min = df["Average_Speed"].min()
if speed_max != speed_min:
    speed_congestion_score = (speed_max - df["Average_Speed"]) / (speed_max - speed_min)
else:
    speed_congestion_score = pd.Series(0.5, index=df.index)

tti_max = df["Travel_Time_Index"].max()
tti_min = df["Travel_Time_Index"].min()
if tti_max != tti_min:
    delay_score = (df["Travel_Time_Index"] - tti_min) / (tti_max - tti_min)
else:
    delay_score = pd.Series(0.5, index=df.index)

incident_max = df["Incident_Level"].max()
if incident_max > 0:
    incident_score = df["Incident_Level"] / incident_max
else:
    incident_score = pd.Series(0, index=df.index)

volume_max = df["Traffic_Volume"].max()
volume_min = df["Traffic_Volume"].min()
if volume_max != volume_min:
    traffic_score = (df["Traffic_Volume"] - volume_min) / (volume_max - volume_min)
else:
    traffic_score = pd.Series(0.5, index=df.index)

df["Congestion_Score"] = (
    utilization_score * 0.35 + speed_congestion_score * 0.20 +
    delay_score * 0.20 + incident_score * 0.10 + traffic_score * 0.15
) * 100
df["Congestion_Score"] = df["Congestion_Score"].round(2)

def calculate_congestion_level(score):
    if score >= 75:
        return "SEVERE"
    elif score >= 50:
        return "HIGH"
    elif score >= 25:
        return "MODERATE"
    else:
        return "LOW"

df["Calculated_Congestion_Level"] = df["Congestion_Score"].apply(calculate_congestion_level)

def create_congestion_summary(row):
    reasons = []
    if row["Capacity_Utilization"] >= 0.90:
        reasons.append("HIGH_CAPACITY_UTILIZATION")
    if row["Speed_Condition"] in ["VERY_LOW_SPEED", "LOW_SPEED"]:
        reasons.append("LOW_SPEED")
    if row["Travel_Delay_Condition"] in ["SEVERE_DELAY", "HIGH_DELAY"]:
        reasons.append("HIGH_TRAVEL_DELAY")
    if row["Incident_Level"] > 0:
        reasons.append("INCIDENT_PRESENT")
    if row["Peak_Traffic_Status"] == "PEAK_TRAFFIC":
        reasons.append("PEAK_TRAFFIC_VOLUME")
    if not reasons:
        reasons.append("NORMAL_TRAFFIC_CONDITION")
    return ", ".join(reasons)

df["Congestion_Reasons"] = df.apply(create_congestion_summary, axis=1)

CAPACITY_PER_LANE = 1500
df["Estimated_Lanes"] = np.ceil(df["Estimated_Road_Capacity"] / CAPACITY_PER_LANE).astype(int)
df["Estimated_Lanes"] = df["Estimated_Lanes"].clip(lower=1)

SATURATION_FLOW_PER_LANE = 1800
df["Saturation_Flow"] = (df["Estimated_Lanes"] * SATURATION_FLOW_PER_LANE)

PHASE_1_SHARE = 0.55
PHASE_2_SHARE = 0.45
df["Phase_1_Flow"] = (df["Traffic_Volume"] * PHASE_1_SHARE).round(2)
df["Phase_2_Flow"] = (df["Traffic_Volume"] * PHASE_2_SHARE).round(2)

def assign_baseline_cycle(congestion_level):
    if congestion_level == "LOW":
        return 60
    elif congestion_level == "MODERATE":
        return 80
    elif congestion_level == "HIGH":
        return 100
    else:
        return 120

df["Baseline_Cycle"] = df["Calculated_Congestion_Level"].apply(assign_baseline_cycle)

NUMBER_OF_PHASES = 2
LOST_TIME_PER_PHASE = 4
TOTAL_LOST_TIME = NUMBER_OF_PHASES * LOST_TIME_PER_PHASE

df["Number_of_Phases"] = NUMBER_OF_PHASES
df["Lost_Time"] = TOTAL_LOST_TIME
df["Available_Green_Time"] = df["Baseline_Cycle"] - df["Lost_Time"]

total_phase_flow = df["Phase_1_Flow"] + df["Phase_2_Flow"]

df["Phase_1_Baseline_Green"] = np.where(
    total_phase_flow > 0,
    (df["Available_Green_Time"] * df["Phase_1_Flow"] / total_phase_flow.replace(0, np.nan)),
    df["Available_Green_Time"] / 2
)
df["Phase_2_Baseline_Green"] = np.where(
    total_phase_flow > 0,
    (df["Available_Green_Time"] * df["Phase_2_Flow"] / total_phase_flow.replace(0, np.nan)),
    df["Available_Green_Time"] / 2
)
df["Phase_1_Baseline_Green"] = pd.Series(df["Phase_1_Baseline_Green"], index=df.index).round(2)
df["Phase_2_Baseline_Green"] = pd.Series(df["Phase_2_Baseline_Green"], index=df.index).round(2)

df["Phase_1_Flow_Ratio"] = (df["Phase_1_Flow"] / df["Saturation_Flow"]).round(4)
df["Phase_2_Flow_Ratio"] = (df["Phase_2_Flow"] / df["Saturation_Flow"]).round(4)

webster_columns = [
    "Area_ID", "Intersection_ID",
] + NAME_COLUMNS_AVAILABLE + [
    "Traffic_Volume", "Road_Capacity_Utilization_Pct", "Estimated_Road_Capacity",
    "Average_Speed", "Travel_Time_Index", "Congestion_Level", "Incident_Level",
    "Capacity_Utilization", "Bottleneck_Status", "Speed_Condition",
    "Travel_Delay_Condition", "Incident_Status", "Accident_Flag",
    "Peak_Traffic_Status", "Congestion_Score", "Calculated_Congestion_Level",
    "Congestion_Reasons", "Estimated_Lanes", "Saturation_Flow",
    "Number_of_Phases", "Phase_1_Flow", "Phase_2_Flow",
    "Phase_1_Flow_Ratio", "Phase_2_Flow_Ratio",
    "Baseline_Cycle", "Lost_Time", "Available_Green_Time",
    "Phase_1_Baseline_Green", "Phase_2_Baseline_Green"
]

webster_input = df[webster_columns].copy()

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
webster_input.to_csv(OUTPUT_FILE, index=False)

print("\nSUCCESS! Rows:", len(webster_input))
print("Columns included:", webster_input.columns.tolist())