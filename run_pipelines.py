import datetime
import pathway as pw
import pandas as pd
from transformers import pipeline as hf_pipeline
from prophet import Prophet

# === 1. Define Schema with timestamp as string ===
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

# === 2. Read CSVs using schema ===
raw_sales = pw.io.csv.read("sales.csv", schema=SalesSchema, mode="streaming")
raw_mentions = pw.io.csv.read("mentions.csv", schema=MentionsSchema, mode="streaming")

# === 3. Convert timestamp string to DateTimeNaive ===
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

# === 4. Forecasting Preparation: .windowby + .reduce + output ===
forecast_pre = (
    sales
    .windowby(
        time_expr=sales.timestamp,
        window=pw.temporal.tumbling(duration=datetime.timedelta(hours=1)),
        instance=sales.sku
    )
    .reduce(
        sku=pw.reducers.max(pw.this.sku),
        quantity=pw.reducers.sum(pw.this.quantity),
        window_start=pw.this._pw_window_start,
        window_end=pw.this._pw_window_end,
    )
)
pw.io.csv.write(forecast_pre, "forecast_pre.csv")  # Used post-run for ML

# === 5. Anomaly Detection (using reduce) ===
anomaly_stats = (
    sales
    .windowby(
        time_expr=sales.timestamp,
        window=pw.temporal.tumbling(duration=datetime.timedelta(minutes=30)),
        instance=sales.sku
    )
    .reduce(
        sku=pw.reducers.max(pw.this.sku),
        
        mean_qty=pw.reducers.avg(pw.this.quantity),
        #std_qty=pw.reducers.std(pw.this.quantity),
        window_start=pw.this._pw_window_start,
    )
)
pw.io.csv.write(anomaly_stats, "anomaly_stats.csv")  # Use later for manual join if needed

# === 6. Sentiment Analysis (map_batches + reduce) ===
sentiment_fn = hf_pipeline("sentiment-analysis")

def classify_sentiment(batch_df):
    batch_df["sentiment"] = batch_df["text"].map(lambda t: sentiment_fn(t)[0]["label"])
    return batch_df

# classify sentiment first
@pw.udf
def get_sentiment(text: str) -> str:
    return sentiment_fn(text)[0]["label"]

classified_mentions = mentions.select(
    timestamp=mentions.timestamp,
    platform=mentions.platform,
    text=mentions.text,
    sentiment=get_sentiment(mentions.text)
)


sentiments = (
    classified_mentions
    .windowby(
        time_expr=classified_mentions.timestamp,
        window=pw.temporal.tumbling(duration=datetime.timedelta(minutes=15)),
        instance=classified_mentions.sentiment
    )
    .reduce(
        sentiment=pw.reducers.max(pw.this.sentiment),
        count=pw.reducers.count(),
        window_start=pw.this._pw_window_start,
    )
)

pw.io.csv.write(sentiments, "sentiments.csv")  

# === 7. Run Pathway and Forecast Postprocessing ===
if __name__ == "__main__":
    pw.run()

    # Post-run ML Forecasting using Prophet on forecast_pre.csv
    df = pd.read_csv("forecast_pre.csv")

    result = []
    for sku in df["sku"].unique():
        sku_df = df[df["sku"] == sku].copy()
        sku_df = sku_df.rename(columns={"window_start": "ds", "quantity": "y"})
        m = Prophet()
        m.fit(sku_df[["ds", "y"]])
        future = m.make_future_dataframe(periods=1, freq="D")
        yhat = m.predict(future).iloc[-1]["yhat"]
        result.append({
            "sku": sku,
            "timestamp": future.iloc[-1]["ds"],
            "forecast": yhat
        })

    forecast_result = pd.DataFrame(result)
    forecast_table = pw.debug.table_from_pandas(forecast_result, name="forecast")
    
    pw.io.csv.write(forecast_table, "forecast.csv")  # Save final forecast
