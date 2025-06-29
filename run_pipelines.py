# run_pipelines.py

import pathway as pw
import pandas as pd
from transformers import pipeline as hf_pipeline
from prophet import Prophet

# === Streaming Tables ===
sales = pw.Table.read_csv("sales.csv", rate=1.0)
mentions = pw.Table.read_csv("mentions.csv", rate=0.5)

# === Forecasting Pipeline ===
def apply_forecast(batch_df):
    df_prophet = batch_df.rename(columns={"timestamp": "ds", "quantity": "y"})
    m = Prophet()
    m.fit(df_prophet[["ds", "y"]])
    future = m.make_future_dataframe(periods=1, freq="D")
    forecast = m.predict(future)
    yhat = forecast.iloc[-1]["yhat"]
    return pd.DataFrame({
        "timestamp": [future.iloc[-1]["ds"]],
        "sku": [batch_df["sku"].iloc[0]],
        "forecast": [yhat]
    })

forecast = (
    sales
    .window("1h", partition_by="sku")
    .aggregate(pw.agg.sum("quantity").alias("quantity"))
    .map_batches(apply_forecast)
)
forecast.output("forecast_table")

# === Anomaly Detection ===
stats = (
    sales
    .window("30m", partition_by="sku")
    .aggregate(
        pw.agg.mean("quantity").alias("mean_qty"),
        pw.agg.std("quantity").alias("std_qty")
    )
)

anomalies = (
    stats
    .join(sales, on=["sku"])
    .filter(lambda r: abs(r.quantity - r.mean_qty) > 3 * r.std_qty)
    .select("timestamp", "sku", "quantity")
)
anomalies.output("anomalies_table")

# === Sentiment Analysis ===
sentiment_fn = hf_pipeline("sentiment-analysis")

def classify_sentiment(batch_df):
    batch_df['sentiment'] = batch_df['text'].map(lambda t: sentiment_fn(t)[0]['label'])
    return batch_df

sentiments = (
    mentions
    .map_batches(classify_sentiment)
    .window("15m")
    .aggregate(pw.agg.count("sentiment").alias("count"), partition_by="sentiment")
)
sentiments.output("sentiments_table")

# === Run the Pipeline ===
if __name__ == "__main__":
    pw.run()
