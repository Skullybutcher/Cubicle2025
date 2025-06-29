# data_simulation.py

import pandas as pd
import random
import re
from datetime import datetime

# === CONFIGURATION ===
STORE_ITEM_CSV       = 'store_item_demand.csv'       # Kaggle store-item demand analysis
ROSSMANN_CSV         = 'rossmann_store_sales.csv'    # Kaggle Rossmann store sales (optional)
AMAZON_REVIEWS_CSV   = 'amazon_reviews.csv'          # Amazon reviews dataset
SENTIMENT140_CSV     = 'sentiment140.csv'            # Sentiment140 tweets dataset

SALES_OUTPUT_CSV     = 'sales.csv'
MENTIONS_OUTPUT_CSV  = 'mentions.csv'

SELECTED_SKUS        = ['1', '2', '3', '4', '5']     # string IDs matching 'item' column
SAMPLE_DAYS          = 7                             # days of sales history to simulate
MENTION_SAMPLE_SIZE  = 200                           # total mentions to simulate

# === 1. Sales Simulation ===
def prepare_sales_simulation_demand(input_csv, output_csv, skus, days):
    # Load store-item demand data; each row is one sale
    df = pd.read_csv(input_csv, parse_dates=['date'], usecols=['date', 'store', 'item'])
    df = df.rename(columns={'date': 'timestamp', 'item': 'sku', 'store': 'store_id'})
    df['sku'] = df['sku'].astype(str)

    # Filter for selected SKUs
    df = df[df['sku'].isin(skus)]
    if df.empty:
        print(f"Warning: No sales rows found for SKUs {skus} in the last {days} days.")

    # Trim to recent days
    max_date = df['timestamp'].max()
    df = df[df['timestamp'] >= (max_date - pd.Timedelta(days=days))]

    # Aggregate counts per day, SKU, store
    qty_df = (
        df.groupby(['timestamp', 'store_id', 'sku'])
          .size()
          .reset_index(name='quantity')
    )

    # Add price and sort
    qty_df['price'] = qty_df['quantity'].apply(lambda q: round(random.uniform(10, 100), 2))
    qty_df = qty_df.sort_values('timestamp')

    # Save to CSV
    qty_df.to_csv(output_csv, index=False)
    print(f"Sales simulation file created: {output_csv} ({len(qty_df)} rows)")

# === 2. Mentions Simulation ===
def prepare_mentions_simulation(amzn_csv, senti_csv, output_csv, sample_size):
    # Load Amazon reviews and convert UNIX Time to datetime
    df_amzn = pd.read_csv(amzn_csv, usecols=['Time', 'Text'])
    df_amzn = df_amzn.dropna().rename(columns={'Time': 'timestamp', 'Text': 'text'})
    df_amzn['timestamp'] = pd.to_datetime(df_amzn['timestamp'], unit='s', errors='coerce')
    df_amzn = df_amzn.dropna(subset=['timestamp'])
    df_amzn['platform'] = 'amazon'

    # Load Sentiment140 tweets and strip timezone
    df_senti = pd.read_csv(senti_csv, usecols=[2, 5], header=None, names=['timestamp', 'text'])
    # Remove trailing timezone (e.g., 'PDT 2009') using regex
    df_senti['timestamp'] = df_senti['timestamp'].str.replace(r' [A-Z]{3,4} \d{4}$', '', regex=True)
    df_senti['timestamp'] = pd.to_datetime(df_senti['timestamp'], format='%a %b %d %H:%M:%S', errors='coerce')
    df_senti = df_senti.dropna(subset=['timestamp'])
    df_senti['platform'] = 'twitter'

    # Combine, sample, sort
    df = pd.concat([df_amzn, df_senti], ignore_index=True)
    df = df.dropna(subset=['text'])
    df = df.sample(n=min(sample_size, len(df))).reset_index(drop=True)
    df = df.sort_values('timestamp')

    # Save to CSV
    df.to_csv(output_csv, index=False)
    print(f"Mentions simulation file created: {output_csv} ({len(df)} rows)")

# === MAIN ===
if __name__ == '__main__':
    prepare_sales_simulation_demand(
        STORE_ITEM_CSV,
        SALES_OUTPUT_CSV,
        SELECTED_SKUS,
        SAMPLE_DAYS
    )
    prepare_mentions_simulation(
        AMAZON_REVIEWS_CSV,
        SENTIMENT140_CSV,
        MENTIONS_OUTPUT_CSV,
        MENTION_SAMPLE_SIZE
    )
