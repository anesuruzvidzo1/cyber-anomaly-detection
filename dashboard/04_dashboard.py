from pathlib import Path
import pandas as pd
import streamlit as st
import plotly.express as px

# 1) Resolve the project root (folder above 'notebooks')
BASE_DIR = Path(__file__).resolve().parent.parent

# 2) Absolute path to the CSV
CSV_PATH = BASE_DIR / "data" / "processed" / "anomaly_scores.csv"


@st.cache_data
def load_data():
    df = pd.read_csv(CSV_PATH)
    df = df.rename(columns={
        "Date first seen":   "timestamp",
        "Src IP Addr":       "host",
        "predicted_anomaly": "anomaly_score"
    })
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df

# 4) Load & confirm
df = load_data()
st.write(f"Loaded {len(df)} records from {CSV_PATH.name}")

# 5) Sidebar controls
st.sidebar.header("Filters & Threshold")
threshold = st.sidebar.slider(
    "Anomaly Score Threshold",
    float(df["anomaly_score"].min()),
    float(df["anomaly_score"].max()),
    float(df["anomaly_score"].quantile(0.95))
)
hosts = st.sidebar.multiselect(
    "Select Hosts (leave blank for all)",
    options=sorted(df["host"].unique()),
    default=[]
)

# 6) Filter
filtered = df[df["anomaly_score"] >= threshold]
if hosts:
    filtered = filtered[filtered["host"].isin(hosts)]
st.write(f"Showing {len(filtered)} records after filtering")

# 7) Time-series chart
ts = (
    filtered
    .groupby(pd.Grouper(key="timestamp", freq="h"))  # lowercase 'h' to silence future warning
    .size()
    .reset_index(name="anomaly_count")
)
fig_ts = px.line(ts, x="timestamp", y="anomaly_count", title="Hourly Anomaly Count")
st.plotly_chart(fig_ts, use_container_width=True)

# 8) Top 10 hosts table (exact two columns)
vc = filtered["host"].value_counts().head(10)
top_hosts = pd.DataFrame({
    "host": vc.index,
    "count": vc.values
})
st.subheader("Top 10 Anomalous Hosts")

st.table(top_hosts)

# 9) Distribution
fig_dist = px.histogram(df, x="anomaly_score", nbins=50, title="Anomaly Score Distribution")
st.plotly_chart(fig_dist, use_container_width=True)