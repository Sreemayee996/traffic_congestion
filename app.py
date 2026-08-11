"""
streamlit_app.py
-----------------
Standalone dashboard for the traffic congestion project. Reads the
Pipeline A batch output CSVs directly (no Flask server needed).

Includes:
- Dropdown filters using real Area Name / Road name
- Recommendation card with before/after bar chart for a chosen location
- Color-coded pie charts (red = severe/critical, green = normal/low)
  for Congestion Level and Decision Priority breakdowns
- A line chart of average congestion score over time (if Date is
  available), with a red threshold line marking the "severe" cutoff
- Full filterable data table + CSV download

Run with:
    streamlit run streamlit_app.py
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
WEBSTER_RESULTS_FILE = BASE_DIR / "data" / "webster_results_full.csv"
SUMMARY_FILE = BASE_DIR / "data" / "simulation_summary.csv"

st.set_page_config(page_title="Traffic Congestion Dashboard", layout="wide")

# ----------------------------------------------------------------------
# Color maps — consistent "alert" coloring across every chart.
# Red = worst / most urgent, green = best / normal.
# ----------------------------------------------------------------------

CONGESTION_COLOR_MAP = {
    "LOW": "#2ecc71",       # green
    "MODERATE": "#f1c40f",  # yellow
    "HIGH": "#e67e22",      # orange
    "SEVERE": "#e74c3c",    # red
}

PRIORITY_COLOR_MAP = {
    "LOW": "#2ecc71",             # green
    "MEDIUM": "#f1c40f",          # yellow
    "HIGH": "#e67e22",            # orange
    "CRITICAL": "#e74c3c",        # red
    "REVIEW_REQUIRED": "#9b59b6", # purple (data issue, not severity)
}

ACCIDENT_RISK_COLOR_MAP = {
    "LOW": "#2ecc71",
    "MEDIUM": "#f1c40f",
    "HIGH": "#e67e22",
    "CRITICAL": "#e74c3c",
}

SEVERE_THRESHOLD = 75  # matches calculate_congestion_level() in webster_preprocessing.py

# ----------------------------------------------------------------------
# Data loading (cached so it doesn't reload on every click)
# ----------------------------------------------------------------------

@st.cache_data
def load_webster_results():
    if not WEBSTER_RESULTS_FILE.exists():
        return None
    return pd.read_csv(WEBSTER_RESULTS_FILE)

@st.cache_data
def load_summary():
    if not SUMMARY_FILE.exists():
        return None
    return pd.read_csv(SUMMARY_FILE)

df = load_webster_results()
summary_df = load_summary()

st.title("Bangalore Traffic Congestion — Webster Results Dashboard")

if df is None:
    st.error(
        "webster_results_full.csv not found in the data/ folder.\n\n"
        "Run these first, from your project root:\n\n"
        "python src/webster_preprocessing.py\n"
        "python src/webster_batch.py\n"
        "python src/batch_report.py"
    )
    st.stop()

HAS_AREA_NAME = "Area Name" in df.columns
HAS_ROAD_NAME = "Road/Intersection Name" in df.columns
HAS_DATE = "Date" in df.columns

AREA_DISPLAY_COL = "Area Name" if HAS_AREA_NAME else "Area_ID"
ROAD_DISPLAY_COL = "Road/Intersection Name" if HAS_ROAD_NAME else "Intersection_ID"

if not HAS_AREA_NAME or not HAS_ROAD_NAME:
    st.warning(
        "Real area/road names not found — showing Area_ID / Intersection_ID "
        "instead. Re-run webster_preprocessing.py then webster_batch.py to get "
        "real names."
    )

if HAS_DATE:
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce", dayfirst=True)

st.caption(f"Loaded {len(df):,} rows from webster_results_full.csv"
           + (f" and {len(summary_df):,} summarized intersections from simulation_summary.csv"
              if summary_df is not None else ""))

# ----------------------------------------------------------------------
# Sidebar filters
# ----------------------------------------------------------------------

st.sidebar.header("Filters")

areas = ["All"] + sorted(df[AREA_DISPLAY_COL].dropna().unique().tolist())
selected_area = st.sidebar.selectbox("Area", areas)

if selected_area != "All":
    filtered_roads = df[df[AREA_DISPLAY_COL] == selected_area][ROAD_DISPLAY_COL].dropna().unique()
else:
    filtered_roads = df[ROAD_DISPLAY_COL].dropna().unique()

roads = ["All"] + sorted(filtered_roads.tolist())
selected_road = st.sidebar.selectbox("Road / Intersection", roads)

priority_options = ["All"] + sorted(df["Decision_Priority"].dropna().unique().tolist())
selected_priority = st.sidebar.selectbox("Decision Priority", priority_options)

search_text = st.sidebar.text_input("Search any column")

filtered = df.copy()
if selected_area != "All":
    filtered = filtered[filtered[AREA_DISPLAY_COL] == selected_area]
if selected_road != "All":
    filtered = filtered[filtered[ROAD_DISPLAY_COL] == selected_road]
if selected_priority != "All":
    filtered = filtered[filtered["Decision_Priority"] == selected_priority]
if search_text:
    mask = filtered.apply(
        lambda row: row.astype(str).str.contains(search_text, case=False).any(), axis=1
    )
    filtered = filtered[mask]

# ----------------------------------------------------------------------
# Recommendation summary card (only shown when one specific location picked)
# ----------------------------------------------------------------------

if selected_area != "All" and selected_road != "All" and not filtered.empty:
    row = filtered.iloc[0]
    st.subheader(f"{row[AREA_DISPLAY_COL]} / {row[ROAD_DISPLAY_COL]} — Recommendation")

    priority_val = str(row.get("Decision_Priority", ""))
    if priority_val == "CRITICAL":
        st.error(f"CRITICAL priority — immediate review recommended.")
    elif priority_val == "HIGH":
        st.warning(f"HIGH priority — significant congestion detected.")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Congestion Score", row.get("Congestion_Score", "N/A"),
                help=str(row.get("Calculated_Congestion_Level", "")))
    col2.metric("Recommended Cycle (sec)", row.get("Webster_Optimal_Cycle_Practical", "N/A"))
    col3.metric("Phase 1 Green (sec)", row.get("Webster_Phase_1_Green", "N/A"))
    col4.metric("Phase 2 Green (sec)", row.get("Webster_Phase_2_Green", "N/A"))

    col5, col6 = st.columns(2)
    col5.metric("Decision Priority", row.get("Decision_Priority", "N/A"))
    col6.metric("Baseline vs Recommended Cycle Change (sec)",
                row.get("Cycle_Change_Seconds", "N/A"))

    baseline = row.get("Baseline_Cycle")
    recommended = row.get("Webster_Optimal_Cycle_Practical")
    if pd.notna(baseline) and pd.notna(recommended):
        before_after_df = pd.DataFrame({
            "Stage": ["Current (Baseline)", "Recommended"],
            "Cycle Length (sec)": [baseline, recommended],
        })
        fig = px.bar(
            before_after_df, x="Stage", y="Cycle Length (sec)",
            color="Stage",
            color_discrete_map={"Current (Baseline)": "#95a5a6", "Recommended": "#2ecc71"},
            title="Signal Cycle: Before vs After Recommendation",
        )
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

# ----------------------------------------------------------------------
# City-wide overview — pie charts (color-coded) + line chart trend
# ----------------------------------------------------------------------

st.subheader("City-wide Overview")

pie_col1, pie_col2, pie_col3 = st.columns(3)

with pie_col1:
    st.caption("Congestion Level breakdown")
    congestion_counts = df["Calculated_Congestion_Level"].value_counts().reset_index()
    congestion_counts.columns = ["Level", "Count"]
    fig = px.pie(
        congestion_counts, names="Level", values="Count",
        color="Level", color_discrete_map=CONGESTION_COLOR_MAP,
        hole=0.4,
    )
    st.plotly_chart(fig, use_container_width=True)

with pie_col2:
    st.caption("Decision Priority breakdown")
    priority_counts = df["Decision_Priority"].value_counts().reset_index()
    priority_counts.columns = ["Priority", "Count"]
    fig = px.pie(
        priority_counts, names="Priority", values="Count",
        color="Priority", color_discrete_map=PRIORITY_COLOR_MAP,
        hole=0.4,
    )
    st.plotly_chart(fig, use_container_width=True)

with pie_col3:
    if "Accident_Flag" in df.columns:
        st.caption("Accident Flag breakdown")
        accident_counts = df["Accident_Flag"].value_counts().reset_index()
        accident_counts.columns = ["Flag", "Count"]
        fig = px.pie(
            accident_counts, names="Flag", values="Count",
            color="Flag", color_discrete_map={"YES": "#e74c3c", "NO": "#2ecc71"},
            hole=0.4,
        )
        st.plotly_chart(fig, use_container_width=True)

# Line chart — congestion score trend over time, with a red threshold line
if HAS_DATE and df["Date"].notna().any():
    st.caption("Average Congestion Score Over Time (red line = SEVERE threshold)")
    trend = df.dropna(subset=["Date"]).groupby(df["Date"].dt.date)["Congestion_Score"].mean().reset_index()
    trend.columns = ["Date", "Avg_Congestion_Score"]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=trend["Date"], y=trend["Avg_Congestion_Score"],
        mode="lines+markers", name="Avg Congestion Score",
        line=dict(color="#3498db"),
        marker=dict(
            color=["#e74c3c" if v >= SEVERE_THRESHOLD else "#3498db"
                   for v in trend["Avg_Congestion_Score"]],
            size=7,
        ),
    ))
    fig.add_hline(
        y=SEVERE_THRESHOLD, line_dash="dash", line_color="red",
        annotation_text="SEVERE threshold", annotation_position="top left",
    )
    fig.update_layout(xaxis_title="Date", yaxis_title="Avg Congestion Score")
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No usable Date column found — trend line chart unavailable. "
            "Re-run the updated webster_preprocessing.py to include Date.")

if summary_df is not None:
    summary_road_col = "Road/Intersection Name" if "Road/Intersection Name" in summary_df.columns else "Intersection_ID"

    st.caption("Top 10 Worst Bottlenecks (from simulation_summary.csv)")
    top10 = summary_df.sort_values("Bottleneck_Index", ascending=False).head(10).copy()
    label_col = top10[summary_road_col] if summary_road_col in top10.columns else top10["Intersection_ID"]

    fig = px.bar(
        top10, x=label_col, y="Bottleneck_Index",
        color="Bottleneck_Index",
        color_continuous_scale=["#2ecc71", "#f1c40f", "#e67e22", "#e74c3c"],
        labels={"x": "Location", "Bottleneck_Index": "Bottleneck Index"},
    )
    fig.update_layout(xaxis_title="Location", yaxis_title="Bottleneck Index")
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# ----------------------------------------------------------------------
# Full filtered table
# ----------------------------------------------------------------------

st.subheader(f"Full Data Table ({len(filtered):,} rows)")
st.dataframe(filtered, use_container_width=True, height=500)

st.download_button(
    "Download filtered data as CSV",
    filtered.to_csv(index=False),
    file_name="filtered_webster_results.csv",
    mime="text/csv",
)
