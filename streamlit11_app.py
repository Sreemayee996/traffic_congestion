import os
import streamlit as st
from src.simulation import TrafficAnalyzer

DATA_PATH = os.environ.get("TRAFFIC_DATA_PATH", "data/Banglore_traffic_Dataset Working.csv")

@st.cache_resource
def load_data():
    return TrafficAnalyzer(DATA_PATH)

analyzer = load_data()

st.title("🚦 Bangalore Traffic Congestion Dashboard")

# Navigation menu
option = st.sidebar.selectbox(
    "Choose a View", 
    ["System Status", "Top Bottlenecks", "Congestion Prediction"]
)

if option == "System Status":
    st.header("API Health Status")
    col1, col2 = st.columns(2)
    col1.metric("Rows Loaded", len(analyzer.data))
    col2.metric("Locations Tracked", len(analyzer.location_catalog))

elif option == "Top Bottlenecks":
    st.header("Top Traffic Bottlenecks")
    top_n = st.slider("Number of Bottlenecks to View", 1, 50, 10)
    st.dataframe(analyzer.bottlenecks(top_n=top_n))

elif option == "Congestion Prediction":
    st.header("Predict Traffic Risk")
    area = st.text_input("Area", "Indiranagar")
    road = st.text_input("Road", "100 Feet Road")
    
    if st.button("Run Analysis"):
        try:
            score = analyzer.congestion_score(area, road)
            st.json(score)
        except Exception as e:
            st.error(f"Error: {e}")
