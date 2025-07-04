import streamlit as st
import pandas as pd
import altair as alt
from datetime import datetime, timedelta

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
    df[col] = pd.to_datetime(df[col], errors="coerce")
    cutoff = datetime.now() - timedelta(days=days)
    return df[df[col] >= cutoff]

def get_latest_ts(df, col):
    if col in df.columns and not df.empty:
        return pd.to_datetime(df[col], errors="coerce").max()
    return "No data yet"

# ---------- Sidebar ---------------------------------------------------------
st.sidebar.header("📂 Data Monitor")
view = st.sidebar.radio("Choose a data view:", ["Forecast", "Anomalies", "Sentiment Trends"])
# Shared list of store IDs
store_ids = pd.concat([
    load_csv("forecast.csv")[["store_id"]] if "store_id" in load_csv("forecast.csv").columns else pd.DataFrame(),
    load_csv("anomalies.csv")[["store_id"]] if "store_id" in load_csv("anomalies.csv").columns else pd.DataFrame()
], ignore_index=True)

store_ids = sorted(store_ids["store_id"].dropna().astype(int).unique()) if not store_ids.empty else []
selected_stores = st.sidebar.multiselect("Filter by Store ID", store_ids, default=store_ids)

default_days_back = 30

if "days_back" not in st.session_state:
    st.session_state["days_back"] = default_days_back

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
        new_days_back = min(180, max(7, delta_days + 1))
        st.session_state["days_back"] = new_days_back
        st.rerun()
    else:
        st.sidebar.warning("⚠️ No recent data found to jump to.")

# ---------- Shared Datasets -------------------------------------------------
forecast_df  = filter_date(load_csv("forecast.csv"),      "timestamp",    days_back)
anomaly_df   = filter_date(load_csv("anomalies.csv"),     "timestamp",    days_back)
sentiment_df = filter_date(load_csv("sentiments.csv"),    "window_start", days_back)

# Make sure timestamps are correct dtype
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
            st.warning(f"⚠️ No forecast data in selected window. Latest available is from **{latest.strftime('%Y-%m-%d')}**.")
        else:
            st.info("Waiting for forecast data …")
    else:
        sku_opts = sorted(forecast_df["sku"].astype(str).unique())
        selected = st.multiselect("Select SKU(s)", sku_opts, default=sku_opts)
        fdf = forecast_df.dropna(subset=["store_id"])
        fdf = fdf[
            fdf["sku"].astype(str).isin(selected) &
            fdf["store_id"].astype(int).isin(selected_stores)
        ]

        fdf = fdf.sort_values("timestamp")
        st.caption(f"Showing data from {fdf['timestamp'].min().date()} to {fdf['timestamp'].max().date()}")
        st.caption(f"📌 Latest timestamp seen: {get_latest_ts(fdf, 'timestamp')}")
        st.caption(f"Filtered by Store IDs: {', '.join(map(str, selected_stores))}")

        # Compute predicted revenue
        fdf["revenue"] = fdf["forecast"] * fdf["quantity"] * fdf["price"]

        sku_revenue = (
        fdf.groupby("sku")["revenue"]
        .sum()
        .reset_index()
        .sort_values("revenue", ascending=False)
    )

        if "store_id" in fdf.columns:
            store_revenue = (
                fdf.groupby("store_id")["revenue"]
                .sum()
                .reset_index()
                .sort_values("revenue", ascending=False)
            )

        revenue_trend = (
            fdf.groupby("timestamp")["revenue"]
            .sum()
            .reset_index()
        )



        

        # Total revenue summary
        total_revenue = fdf["revenue"].sum()
        # Show summary number
        st.metric("💰 Total Predicted Revenue", f"${total_revenue:,.2f}")

# Revenue per SKU
        st.subheader("📊 Revenue by SKU")
        st.dataframe(sku_revenue, use_container_width=True)

# Revenue per store (if applicable)
        if "store_id" in fdf.columns:
            st.subheader("🏪 Revenue by Store")
            st.dataframe(store_revenue, use_container_width=True)

# Revenue Trend Chart
        st.subheader("📈 Revenue Forecast Over Time")
        line_chart = (
            alt.Chart(revenue_trend)
            .mark_line(point=True)
            .encode(
                x=alt.X("timestamp:T", title="Timestamp"),
                y=alt.Y("revenue:Q", title="Predicted Revenue"),
                tooltip=["timestamp", "revenue"]
            )
            .interactive()
            .properties(height=350)
        )
        st.altair_chart(line_chart, use_container_width=True)


# Select and reorder columns if they exist
        display_cols = [col for col in ["sku", "forecast", "quantity", "price", "revenue", "timestamp"] if col in fdf.columns]
        st.dataframe(fdf[display_cols], use_container_width=True)



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
            st.warning(f"⚠️ No anomalies in selected window. Latest anomaly is from **{latest.strftime('%Y-%m-%d')}**.")
        else:
            st.success("No anomalies detected.")
    else:
        sku_opts = sorted(anomaly_df["sku"].astype(str).unique())
        selected = st.multiselect("Select SKU(s)", sku_opts, default=sku_opts)
        adf = anomaly_df.dropna(subset=["store_id"])
        adf = adf[
            adf["sku"].astype(str).isin(selected) &
            adf["store_id"].astype(int).isin(selected_stores)
        ]
        st.subheader("⚠️ Anomaly Summary")
        col1, col2 = st.columns(2)

        with col1:
            st.metric("Total Anomalies", len(adf))
            st.metric("Distinct SKUs", adf["sku"].nunique())

        with col2:
            last = adf["timestamp"].max()
            st.metric("Most Recent", last.strftime("%b %d, %Y") if pd.notnull(last) else "–")

        top_skus = adf["sku"].value_counts().head(3)
        st.markdown("**🔢 Top 3 Anomalous SKUs:**")
        st.write(top_skus)

        st.caption(f"Showing data from {adf['timestamp'].min().date()} to {adf['timestamp'].max().date()}")
        st.caption(f"📌 Latest timestamp seen: {get_latest_ts(adf, 'timestamp')}")
        st.caption(f"Filtered by Store IDs: {', '.join(map(str, selected_stores))}")

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
            st.warning(f"⚠️ No sentiment data in selected window. Latest data is from **{latest.strftime('%Y-%m-%d')}**.")
        else:
            st.info("Waiting for sentiment data …")
    else:
        sentiment_df = sentiment_df[sentiment_df["window_start"] > "2000-01-01"]
        sent_opts = sentiment_df["sentiment"].unique().tolist()
        sel_sent  = st.multiselect("Show sentiments", sent_opts, default=sent_opts)
        sdf = sentiment_df[sentiment_df["sentiment"].isin(sel_sent)]
    st.subheader("💬 Sentiment Summary")
    col1, col2 = st.columns(2)

    with col1:
        dist = sdf.groupby("sentiment")["count"].sum()
        st.write("**Distribution**")
        st.dataframe(dist.reset_index().rename(columns={"count": "Total"}), use_container_width=True)

    with col2:
        pie = alt.Chart(dist.reset_index()).mark_arc().encode(
            theta="count:Q",
            color="sentiment:N",
            tooltip=["sentiment", "count"]
        ).properties(height=250)
        st.altair_chart(pie, use_container_width=True)

        st.caption(f"Showing data from {sdf['window_start'].min().date()} to {sdf['window_start'].max().date()}")
        st.caption(f"📌 Latest timestamp seen: {get_latest_ts(sdf, 'window_start')}")
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
