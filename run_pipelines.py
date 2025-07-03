import datetime
import pandas as pd
import pathway as pw
from transformers import pipeline as hf_pipeline
from prophet import Prophet

import os
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

#comment the next 3 lines if you want to see the Pathway dashboard in the terminal
#leave them as they are if you want to see the logs in pipeline.log
import sys
sys.stdout = open("pipeline.log", "a")
sys.stderr = sys.stdout

# === Hugging Face Sentiment Pipeline ===
sentiment_fn = hf_pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

# === Define Schemas ===
class SalesSchema(pw.Schema):
    timestamp: str
    store_id: float
    sku: str
    quantity: int
    price: float

class MentionsSchema(pw.Schema):
    timestamp: str
    text: str
    platform: str

# === Read Streaming CSVs ===
raw_sales = pw.io.csv.read("sales.csv", schema=SalesSchema, mode="streaming")
raw_mentions = pw.io.csv.read("mentions.csv", schema=MentionsSchema, mode="streaming")

# === Clean & Convert ===
sales = raw_sales.select(
    timestamp=pw.apply_with_type(datetime.datetime.fromisoformat, pw.DateTimeNaive, raw_sales.timestamp),
    store_id=raw_sales.store_id,
    sku=raw_sales.sku,
    quantity=raw_sales.quantity,
    price=raw_sales.price
)

mentions = raw_mentions.select(
    timestamp=pw.apply_with_type(datetime.datetime.fromisoformat, pw.DateTimeNaive, raw_mentions.timestamp),
    text=raw_mentions.text,
    platform=raw_mentions.platform
)

# === Forecast Preprocessing ===
forecast_pre = (
    sales
    .windowby(time_expr=sales.timestamp, window=pw.temporal.tumbling(duration=datetime.timedelta(hours=1)), instance=sales.sku)
    .reduce(
        sku=pw.reducers.max(pw.this.sku),
        quantity=pw.reducers.sum(pw.this.quantity),
        window_start=pw.this._pw_window_start,
        window_end=pw.this._pw_window_end,
    )
)
pw.io.csv.write(forecast_pre, "forecast_pre.csv")

# === Anomaly Preprocessing ===
anomaly_stats = (
    sales
    .windowby(time_expr=sales.timestamp, window=pw.temporal.tumbling(duration=datetime.timedelta(minutes=30)), instance=sales.sku)
    .reduce(
        sku=pw.reducers.max(pw.this.sku),
        mean_qty=pw.reducers.avg(pw.this.quantity),
        mean_sq_qty=pw.reducers.avg(pw.this.quantity * pw.this.quantity),
        window_start=pw.this._pw_window_start,
    )
)
pw.io.csv.write(anomaly_stats, "anomaly_stats.csv")

# === Sentiment Classification ===
@pw.udf
def get_sentiment(text: str) -> str:
    try:
        return sentiment_fn(text[:512])[0]["label"]
    except Exception:
        return "UNKNOWN"

classified_mentions = mentions.select(
    timestamp=mentions.timestamp,
    platform=mentions.platform,
    text=mentions.text,
    sentiment=get_sentiment(mentions.text)
)

sentiments = (
    classified_mentions
    .windowby(time_expr=classified_mentions.timestamp, window=pw.temporal.tumbling(duration=datetime.timedelta(minutes=15)), instance=classified_mentions.sentiment)
    .reduce(
        sentiment=pw.reducers.max(pw.this.sentiment),
        count=pw.reducers.count(),
        window_start=pw.this._pw_window_start,
    )
)
pw.io.csv.write(sentiments, "sentiments.csv")

# === Shared Threshold Tracker for Watchdog ===
ROW_THRESHOLDS = {
    "forecast_pre.csv": {"last_rows": 0, "threshold": 10},
    "anomaly_stats.csv": {"last_rows": 0, "threshold": 10},
}

# === Postprocessing Logic ===
def run_forecast_and_anomalies():
    # Forecasting
    if os.path.exists("forecast_pre.csv"):
        df = pd.read_csv("forecast_pre.csv")
        forecast_result = []

        for sku in df["sku"].unique():
            sku_df = df[df["sku"] == sku].copy()
            sku_df = sku_df.rename(columns={"window_start": "ds", "quantity": "y"})

            m = Prophet()
            m.fit(sku_df[["ds", "y"]])

        # Forecast multiple periods
            future = m.make_future_dataframe(periods=7, freq="D")  # change as needed
            forecast = m.predict(future)[["ds", "yhat"]]

            forecast["sku"] = sku
            forecast_result.append(forecast)

        forecast_all = pd.concat(forecast_result)
        forecast_all.rename(columns={"ds": "timestamp", "yhat": "forecast"}, inplace=True)
        forecast_all.to_csv("forecast.csv", index=False)

        print(f"✅ forecast.csv updated ({len(forecast_all)} rows)")

    # Anomalies
    if os.path.exists("anomaly_stats.csv") and os.path.exists("sales.csv"):
        stats_df = pd.read_csv("anomaly_stats.csv", parse_dates=["window_start"])
        sales_df = pd.read_csv("sales.csv", parse_dates=["timestamp"])
        stats_df["std_qty"] = (stats_df["mean_sq_qty"] - stats_df["mean_qty"]**2) ** 0.5

        

        merged = pd.merge_asof(
            sales_df.sort_values("timestamp"),
            stats_df.sort_values("window_start"),
            by="sku",
            left_on="timestamp",
            right_on="window_start",
            direction="backward",
            tolerance=pd.Timedelta("30min")
        )

        

        def safe_z(row):
            try:
                return (row["quantity"] - row["mean_qty"]) / row["std_qty"] if row["std_qty"] > 0 else 0
            except:
                return 0

        merged["z_score"] = merged.apply(safe_z, axis=1)
        merged["is_anomaly"] = merged["z_score"].abs() > 1.5

        anomalies = merged[merged["is_anomaly"]]
        if not anomalies.empty:
            anomalies.to_csv("anomalies.csv", index=False)
            print(f"🚨 anomalies.csv updated ({len(anomalies)} rows)")
        else:
            print("⚠️ No anomalies found.")
            pd.DataFrame(columns=merged.columns).to_csv("anomalies.csv", index=False)

        print("std_qty stats:", stats_df["std_qty"].describe())
        print("Rows in merged:", len(merged))
        print("Number of anomalies:", len(anomalies))

# === Watchdog Handler ===
class CSVUpdateHandler(FileSystemEventHandler):
    def on_modified(self, event):
        if not event.is_directory:
            filename = Path(event.src_path).name
            if filename in ROW_THRESHOLDS:
                try:
                    current_rows = sum(1 for _ in open(filename)) - 1  # skip header
                    last_rows = ROW_THRESHOLDS[filename]["last_rows"]
                    threshold = ROW_THRESHOLDS[filename]["threshold"]
                    if current_rows - last_rows >= threshold:
                        print(f"🟢 {filename} grew by {current_rows - last_rows} rows, running postprocessing...")
                        ROW_THRESHOLDS[filename]["last_rows"] = current_rows
                        run_forecast_and_anomalies()
                except Exception as e:
                    print(f"⚠️ Error processing {filename}: {e}")

# === Launch ===
if __name__ == "__main__":
    observer = Observer()
    observer.schedule(CSVUpdateHandler(), path=".", recursive=False)
    observer.start()
    print("👀 Watching forecast_pre.csv and anomaly_stats.csv (streaming mode with thresholds)...")

    try:
        pw.run()
    finally:
        observer.stop()
        observer.join()
