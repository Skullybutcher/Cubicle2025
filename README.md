# Retail Flash-Insight Copilot

> An AI-powered streaming analytics dashboard for modern retail, built with **Pathway**, **Prophet**, **Transformers**, and **Streamlit**.

---

## 🚀 Overview

The Retail Flash-Insight Copilot enables **real-time, intelligent monitoring** of sales and social buzz. It combines live data ingestion with predictive analytics and sentiment-aware insights, giving retailers the edge they need to:

* Forecast demand per SKU every hour
* Detect anomalies in item-store sales patterns
* Analyze customer sentiment across platforms
* Visualize everything in an interactive dashboard

---

## 🧠 Features

### ✅ Real-Time Forecasting

* Uses Facebook Prophet to forecast demand per SKU in hourly windows
* Powered by streaming data ingestion using Pathway

### 🚨 Anomaly Detection

* Calculates z-score–based anomalies in sales quantity
* Customizable thresholds possible per SKU

### 💬 Sentiment Analysis

* Uses Hugging Face's DistilBERT to classify tweets/reviews into sentiment
* Streaming mentions visualized as trend lines over time

### 📊 Interactive Dashboard

Built with Streamlit:

* Lookback window slider (7 to 180 days)
* Per-SKU & per-store filtering
* Sentiment breakdown by label
* Anomaly bars and forecast line charts

---

## 🏗️ Architecture

* **Pathway**: Streaming computation + windowing
* **Prophet**: Forecasting per SKU post-windowing
* **Transformers**: Sentiment classification (DistilBERT)
* **Watchdog**: Triggers forecast/anomaly updates based on CSV growth
* **Streamlit**: Real-time dashboard UI

---

## 📂 Project Structure

```
Cubicle2025/
├── app.py                  # Streamlit dashboard
├── run_pipelines.py       # Core ETL and streaming pipeline
├── data_simulation.py     # Generates and streams sample data
├── sales.csv              # Streamed sales input
├── mentions.csv           # Streamed mentions input
├── forecast.csv           # Forecasted outputs
├── anomalies.csv          # Anomalies detected
├── sentiments.csv         # Sentiment trend data
└── README.md              # You're here
```

---

## 🛠️ Setup & Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Start everything

```bash
# Terminal 1: Stream data
python data_simulation.py

# Terminal 2: Run Pathway pipelines + triggers
python run_pipelines.py

# Terminal 3: Launch dashboard
streamlit run app.py
```

---

## 📦 Data Sources (for Simulation)

* [Kaggle Store-Item Demand Dataset](https://www.kaggle.com/c/demand-forecasting-kernels)
* [Amazon Reviews Dataset](https://nijianmo.github.io/amazon/index.html)
* [Sentiment140 Tweets Dataset](http://help.sentiment140.com/for-students/)

---

## 🌟 What's Next

* Show latest timestamp seen in each dashboard tab
* SKU-specific anomaly thresholds
* Store-level breakdown and filtering
* Revenue forecasts = `quantity × price × forecast`
* Dockerize for reproducibility

---

## 👥 Authors

* team Zeroday
* Built during CodeCubicle 2025

---

## 📃 License

MIT
