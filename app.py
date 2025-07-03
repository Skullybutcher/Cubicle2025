import streamlit as st
import pandas as pd
import altair as alt
from datetime import datetime, timedelta, timezone

st.set_page_config(page_title="Retail Flash‑Insight Copilot", layout="wide")

# ---------- Utility ---------------------------------------------------------
@st.cache_data(ttl=10)
def load_csv(path):
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        return pd.DataFrame()

def filter_date(df, col, days):
    if df.empty or col not in df.columns:
        return df
    max_ts = df[col].max()
    df[col] = pd.to_datetime(df[col], errors="coerce")
    cutoff = datetime.now() - timedelta(days=days)
    #cutoff = max_ts - timedelta(days=days)
    return df[df[col] >= cutoff]

# ---------- Sidebar ---------------------------------------------------------
st.sidebar.header("📂 Data Monitor")
view = st.sidebar.radio("Choose a data view:", ["Forecast", "Anomalies", "Sentiment Trends"])
# Default window
default_days_back = 30

# State handling for dynamic jump
if "days_back" not in st.session_state:
    st.session_state["days_back"] = default_days_back

# Manual slider
days_back = st.sidebar.slider("Look‑back window (days)", 7, 180, st.session_state["days_back"], 1)

if st.sidebar.button("🔄 Jump to Latest Available Data"):
    latest_dates = []

    for file, col in [("forecast.csv", "timestamp"),
                      ("anomalies.csv", "timestamp"),
                      ("sentiments.csv", "window_start")]:
        df = load_csv(file)
        if col in df.columns:
            max_date = pd.to_datetime(df[col], errors="coerce").max()
            if not pd.isnull(max_date):
                latest_dates.append(max_date)

    if latest_dates:
        latest = max(latest_dates)
        today = datetime.now().date()
        delta_days = (today - latest.date()).days

        # Clamp value between 7 and 180
        new_days_back = min(180, max(7, delta_days + 1))
        st.session_state["days_back"] = new_days_back
        st.rerun()
    else:
        st.sidebar.warning("⚠️ No recent data found to jump to.")


# ---------- Shared datasets -------------------------------------------------
forecast_df  = filter_date(load_csv("forecast.csv"),      "timestamp",    days_back)
anomaly_df   = filter_date(load_csv("anomalies.csv"),     "timestamp",    days_back)
sentiment_df = filter_date(load_csv("sentiments.csv"),    "window_start", days_back)

# Make sure timestamps are proper dtype
for col in ["timestamp", "window_start"]:
    for df in [forecast_df, anomaly_df, sentiment_df]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

# ---------- Forecast View ---------------------------------------------------
if view == "Forecast":
    st.title("📈 Forecasted Demand")
    if forecast_df.empty:
        
        raw_forecast = load_csv("forecast.csv")
        if not raw_forecast.empty:
            latest = pd.to_datetime(raw_forecast["timestamp"], errors="coerce").max()
            st.warning(f"⚠️ No forecast data in selected window. "
                   f"Latest available forecast is from **{latest.strftime('%Y-%m-%d')}**. "
                   f"Try increasing the slider to include that date.")
        else:
            st.info("Waiting for forecast data …")
    else:
        sku_opts = sorted(forecast_df["sku"].astype(str).unique())
        selected = st.multiselect("Select SKU(s)", sku_opts, default=sku_opts)
        fdf = forecast_df[forecast_df["sku"].astype(str).isin(selected)]

        st.caption(f"Showing data from {fdf['timestamp'].min().date()} to {fdf['timestamp'].max().date()}")
        st.dataframe(fdf, use_container_width=True)

        chart = (
            alt.Chart(fdf)
            .mark_line(point=True)
            .encode(
                x=alt.X("timestamp:T", title="Timestamp"),
                y=alt.Y("forecast:Q", title="Forecast"),
                color=alt.Color("sku:N", legend=alt.Legend(title="SKU")),
                tooltip=["sku", "forecast", "timestamp"]
            )
            .interactive()
            .properties(height=350)
        )
        st.altair_chart(chart, use_container_width=True)

# ---------- Anomaly View ----------------------------------------------------
elif view == "Anomalies":
    st.title("🚨 Anomaly Events")
    
    if anomaly_df.empty:
        raw_anomaly = load_csv("anomalies.csv")
        if not raw_anomaly.empty:
            latest = pd.to_datetime(raw_anomaly["timestamp"], errors="coerce").max()
            st.warning(f"⚠️ No anomalies in selected window. Latest anomaly data is from **{latest.strftime('%Y-%m-%d')}**. "
             "Try increasing the look-back slider.")
        else:
            st.success("No anomalies detected in the selected window.")
    else:
        sku_opts = sorted(anomaly_df["sku"].astype(str).unique())
        selected = st.multiselect("Select SKU(s)", sku_opts, default=sku_opts)
        adf = anomaly_df[anomaly_df["sku"].astype(str).isin(selected)]

        st.caption(f"Showing data from {adf['timestamp'].min().date()} to {adf['timestamp'].max().date()}")
        st.dataframe(adf, use_container_width=True)

        bar = (
            alt.Chart(adf)
            .mark_bar(color="crimson")
            .encode(
                x=alt.X("timestamp:T", title="Timestamp"),
                y=alt.Y("z_score:Q", title="Z‑score"),
                tooltip=["sku", "store_id", "quantity", "z_score"]
            )
            .properties(height=350)
        )
        st.altair_chart(bar, use_container_width=True)

# ---------- Sentiment View --------------------------------------------------
else:
    st.title("💬 Mention Sentiments Over Time")
    if sentiment_df.empty:
        raw_sentiment = load_csv("sentiments.csv")
        if not raw_sentiment.empty:
            latest = pd.to_datetime(raw_sentiment["window_start"], errors="coerce").max()
            st.warning(f"⚠️ No sentiment data in current window. Latest data starts from **{latest.strftime('%Y-%m-%d')}**.")
        else:
            st.info("Waiting for sentiment data …")

    else:
        sentiment_df = sentiment_df[sentiment_df["window_start"] > "2000-01-01"]

        sent_opts = sentiment_df["sentiment"].unique().tolist()
        sel_sent  = st.multiselect("Show sentiments", sent_opts, default=sent_opts)
        sdf = sentiment_df[sentiment_df["sentiment"].isin(sel_sent)]

        st.caption(f"Showing data from {sdf['window_start'].min().date()} to {sdf['window_start'].max().date()}")
        st.dataframe(sdf, use_container_width=True)

        line = (
            alt.Chart(sdf)
            .mark_line(point=True)
            .encode(
                x=alt.X("window_start:T", title="Window start"),
                y=alt.Y("count:Q", title="Count"),
                color=alt.Color("sentiment:N", legend=alt.Legend(title="Sentiment")),
                tooltip=["sentiment", "count", "window_start"]
            )
            .interactive()
            .properties(height=350)
        )
        st.altair_chart(line, use_container_width=True)
