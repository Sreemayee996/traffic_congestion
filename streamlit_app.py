"""
streamlit_app.py
-----------------
A standalone dashboard for the traffic congestion project.
Reads the Pipeline A batch output CSVs directly (no Flask server
needed) and lets you browse, filter, and visualize the data with
dropdowns, a full table, and before/after charts.

Run with:
    streamlit run streamlit_app.py
"""

import pandas as pd
import streamlit as st
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
WEBSTER_RESULTS_FILE = BASE_DIR / "data" / "webster_results_full.csv"
SUMMARY_FILE = BASE_DIR / "data" / "simulation_summary.csv"

st.set_page_config(page_title="Traffic Congestion Dashboard", layout="wide")

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

st.caption(f"Loaded {len(df):,} rows from webster_results_full.csv"
           + (f" and {len(summary_df):,} summarized intersections from simulation_summary.csv"
              if summary_df is not None else ""))

# ----------------------------------------------------------------------
# Sidebar filters
# ----------------------------------------------------------------------

st.sidebar.header("Filters")

areas = ["All"] + sorted(df["Area_ID"].dropna().unique().tolist())
selected_area = st.sidebar.selectbox("Area", areas)

if selected_area != "All":
    filtered_intersections = df[df["Area_ID"] == selected_area]["Intersection_ID"].dropna().unique()
else:
    filtered_intersections = df["Intersection_ID"].dropna().unique()

intersections = ["All"] + sorted(filtered_intersections.tolist())
selected_intersection = st.sidebar.selectbox("Intersection", intersections)

priority_options = ["All"] + sorted(df["Decision_Priority"].dropna().unique().tolist())
selected_priority = st.sidebar.selectbox("Decision Priority", priority_options)

search_text = st.sidebar.text_input("Search any column")

# Apply filters
filtered = df.copy()
if selected_area != "All":
    filtered = filtered[filtered["Area_ID"] == selected_area]
if selected_intersection != "All":
    filtered = filtered[filtered["Intersection_ID"] == selected_intersection]
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

if selected_area != "All" and selected_intersection != "All" and not filtered.empty:
    row = filtered.iloc[0]
    st.subheader(f"{row['Area_ID']} / {row['Intersection_ID']} — Recommendation")

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

    # Before vs after bar chart
    baseline = row.get("Baseline_Cycle")
    recommended = row.get("Webster_Optimal_Cycle_Practical")
    if pd.notna(baseline) and pd.notna(recommended):
        chart_df = pd.DataFrame({
            "Cycle Length (sec)": [baseline, recommended]
        }, index=["Current (Baseline)", "Recommended"])
        st.bar_chart(chart_df)

    st.divider()

# ----------------------------------------------------------------------
# City-wide overview charts (shown regardless of filter)
# ----------------------------------------------------------------------

st.subheader("City-wide Overview")

overview_col1, overview_col2 = st.columns(2)

with overview_col1:
    st.caption("Decision Priority breakdown")
    priority_counts = df["Decision_Priority"].value_counts()
    st.bar_chart(priority_counts)

with overview_col2:
    st.caption("Congestion Level breakdown")
    congestion_counts = df["Calculated_Congestion_Level"].value_counts()
    st.bar_chart(congestion_counts)

if summary_df is not None:
    st.caption("Top 10 Worst Bottlenecks (from simulation_summary.csv)")
    top10 = summary_df.sort_values("Bottleneck_Index", ascending=False).head(10)
    st.bar_chart(top10.set_index("Intersection_ID")["Bottleneck_Index"])

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