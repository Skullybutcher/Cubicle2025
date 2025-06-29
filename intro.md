# Retail Flash‑Insight Copilot (Hackathon MVP) Design Document

---

## 1. Project Overview  

**Goal:** Build a 48‑hour prototype of a Flash‑Insight Copilot using Pathway for streaming analytics and Streamlit for UI, powered by simulated retail data.

**Key Analytics Pipelines:**  
1. Demand Forecasting  
2. Anomaly Detection  
3. Sentiment Analysis  

---

## 2. Data Collection & Simulation  

### 2.1 Datasets Selected  

| Component          | Dataset                                                                 | Notes                                                        |
|--------------------|-------------------------------------------------------------------------|--------------------------------------------------------------|
| Demand Forecasting  | [Kaggle: Demand Forecasting](https://www.kaggle.com/competitions/demand-forecasting-kernels-only/data) | 5–10 (store,item) pairs; replay at 1 row/sec                 |
|                    | [Kaggle: Rossmann Store Sales](https://www.kaggle.com/datasets/pratyushakar/rossmann-store-sales/data) | Adds `Promo`, `Holiday` features                             |
| Sentiment Analysis  | [Kaggle: Amazon Reviews](https://www.kaggle.com/datasets/arhamrumi/amazon-product-reviews)           | Sample ~2 K reviews as “social mentions”                     |
|                    | [Kaggle: Sentiment140](https://www.kaggle.com/datasets/kazanova/sentiment140)                         | Tweet‑style variety                                           |

### 2.2 Simulator Scripts  

- **Sales Stream** (`sales.csv` → Pathway):  
  - **Schema:** `timestamp,sku,quantity,price,store_id`  
  - **Logic:** Read CSV rows, `time.sleep(1/rate)`, emit to Pathway.

- **Social Mentions Stream** (`mentions.csv` → Pathway):  
  - **Schema:** `timestamp,text,platform`  
  - **Logic:** Shuffle rows, `time.sleep(1/rate)`, emit to Pathway.

- **(Optional) Cart Events** (`cart_events.json`):  
  - **Schema:**  
    ```json
    {
      "timestamp": "2025-06-29T14:32:05Z",
      "session_id": "uuid",
      "sku": "SKU_001",
      "event": "add_to_cart" | "checkout_start" | "checkout_abandon"
    }
    ```

- **SKU Catalog** (`sku_catalog.csv`):  
  - **Fields:** `sku,product_name,category`

---

## 3. High‑Level Architecture  

```text
┌─────────────────┐      ┌───────────────────┐      ┌───────────────────┐
│ Data Simulator  │──▶──▶│ Pathway Pipelines │──▶──▶│ Streamlit Frontend│
│ (CSV/JSON + PT) │      │ (Forecast, Anomaly,│      │ (Charts & Tables) │
└─────────────────┘      │  Sentiment)       │      └───────────────────┘
```

## 4. Pathway Pipeline Definitions

### 4.1 Ingestion

```python
import pathway as pw

sales    = pw.Table.read_csv("sales.csv", rate=1.0)
mentions = pw.Table.read_csv("mentions.csv", rate=0.5)
```

### 4.2 Demand Forecasting

```python
def apply_forecast(batch_df):
    # train Prophet on batch_df[['timestamp','qty']]
    # return DataFrame with ['timestamp','sku','qty','forecast']
    ...

forecast = (
    sales
    .window("1h", partition_by="sku")
    .aggregate(pw.agg.sum("quantity").alias("qty"))
    .map_batches(apply_forecast)
)
forecast.output("forecast_table")
```

### 4.3 Anomaly Detection

```python
stats = (
    sales
    .window("30m", partition_by="sku")
    .aggregate(
        pw.agg.mean("quantity").alias("mean_qty"),
        pw.agg.std("quantity").alias("std_qty"),
    )
)

anomalies = (
    stats
    .join(sales, on=["sku"])
    .filter(lambda r: abs(r.quantity - r.mean_qty) > 3 * r.std_qty)
    .select("timestamp", "sku", "quantity")
)
anomalies.output("anomalies_table")
```

### 4.4 Sentiment Analysis

```python
from transformers import pipeline
sentiment_fn = pipeline("sentiment-analysis")

def classify(df):
    df["sentiment"] = df["text"].map(lambda t: sentiment_fn(t)[0]["label"])
    return df

sentiments = (
    mentions
    .map_batches(classify)
    .window("15m")
    .aggregate(pw.agg.count("sentiment").alias("count"), partition_by="sentiment")
)
sentiments.output("sentiments_table")
```

### 4.5 Run the Pipelines

```python
pw.run()
```

## 5. Model Training & Evaluation

| Pipeline           | Model/Method          | Training Frequency               | Evaluation                      |
|--------------------|----------------------|--------------------------------|--------------------------------|
| Demand Forecasting  | Prophet / ARIMA      | Batch at startup (or daily)     | MAPE on hold‑out slice          |
| Anomaly Detection   | Rule‑based (z‑score) | N/A                            | Inject synthetic spikes; measure recall |
| Sentiment Analysis  | Pretrained Transformer | N/A                            | Sample 200 texts; accuracy ≥85% |

## 6. Streamlit Prototype

```python
import streamlit as st
from pathway_client import get_table

st.title("Flash‑Insight Copilot (Prototype)")

# 1. Demand Forecast
df_fc = get_table("forecast_table")
st.subheader("Demand Forecast")
st.line_chart(df_fc.set_index("timestamp")[["qty", "forecast"]])

# 2. Anomalies
df_anom = get_table("anomalies_table")
st.subheader("Recent Anomalies")
st.dataframe(df_anom)

# 3. Sentiment Trends
df_sent = get_table("sentiments_table")
st.subheader("Sentiment Trends")
st.bar_chart(df_sent.set_index("sentiment")["count"])
```

## 7. Development Roadmap & Milestones

| Stage               | Deliverable                          | Est. Time  |
|---------------------|------------------------------------|------------|
| A. Data Prep & Sim  | Download & trim datasets; simulator scripts | 3 hrs      |
| B. Forecast Pipeline | Pathway job + Prophet integration  | 4 hrs      |
| C. Anomaly Pipeline  | Rolling z‑score in Pathway          | 2 hrs      |
| D. Sentiment Pipeline| HuggingFace mapping + aggregation  | 3 hrs      |
| E. Streamlit Integration | Wire UI to Pathway tables        | 3 hrs      |
| F. Testing & Polish  | End‑to‑end test, demo prep          | 2 hrs      |

Total: ~17 hours + buffer

## 8. Next Steps

- Environment Setup: Install pathway, prophet, transformers, streamlit.
- Data Prep: Download, trim, and format CSV/JSON files.
- Simulator: Implement and test replay scripts.
- Pathway Pipelines: Code, run, and validate each analytics pipeline.
- UI Prototype: Build Streamlit app, connect to live outputs.
- Demo: Run a full live simulation, collect metrics, and prepare presentation.
