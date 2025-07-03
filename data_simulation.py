import pandas as pd
import random
import time
from datetime import timedelta
from pathlib import Path
from threading import Thread

STORE_ITEM_CSV = '/root/codeCubicle/store_item_demand.csv'
AMAZON_REVIEWS_CSV = '/root/codeCubicle/amazon_reviews.csv'
SENTIMENT140_CSV = '/root/codeCubicle/sentiment140.csv'

SALES_OUT = 'sales.csv'
MENTIONS_OUT = 'mentions.csv'

SKUS = ['1', '2', '3', '4', '5']
SLEEP_SECONDS = 10
BATCH_SIZE = 5  # rows per push
LOOP_LIMIT = None  # Set to an int to stop after X batches

def prepare_sales_simulation_demand(input_csv, skus, days):
    # Load store-item demand data
    df = pd.read_csv(input_csv, parse_dates=['date'], usecols=['date', 'store', 'item'])
    df = df.rename(columns={'date': 'timestamp', 'item': 'sku', 'store': 'store_id'})
    df['sku'] = df['sku'].astype(str)

    # Filter for selected SKUs
    df = df[df['sku'].isin(skus)]
    if df.empty:
        print(f"⚠️ Warning: No sales rows found for SKUs {skus} in the last {days} days.")

    # Trim to recent days
    max_date = df['timestamp'].max()
    df = df[df['timestamp'] >= (max_date - pd.Timedelta(days=days))]

    # Aggregate quantity per day/store/sku
    qty_df = (
        df.groupby(['timestamp', 'store_id', 'sku'])
          .size()
          .reset_index(name='quantity')
    )

    # Add variability to quantity
    qty_df['quantity'] += [random.randint(0, 5) for _ in range(len(qty_df))]

    # Add price based on quantity
    qty_df['price'] = qty_df['quantity'].apply(lambda q: round(random.uniform(10, 100), 2))

    # Sort
    qty_df = qty_df.sort_values('timestamp')

    # Normalize years for sales data
    def safe_replace_year(ts, year):
        try:
            return ts.replace(year=year)
        except ValueError:
            if ts.month == 2 and ts.day == 29:
                return ts.replace(year=year, day=28)
            return ts

    now = pd.Timestamp.now()
    qty_df["timestamp"] = qty_df["timestamp"].apply(lambda t: safe_replace_year(t, now.year) if t.year < 2020 else t)

    print(f"✅ Sales simulation DataFrame created: {len(qty_df)} rows")
    return qty_df

def prepare_mentions_feed():
    df_amzn = pd.read_csv(AMAZON_REVIEWS_CSV, usecols=['Time', 'Text']).dropna()
    df_amzn['timestamp'] = pd.to_datetime(df_amzn['Time'], unit='s', errors='coerce')
    df_amzn['platform'] = 'amazon'
    df_amzn = df_amzn.rename(columns={'Text': 'text'})[['timestamp', 'text', 'platform']]

    df_senti = pd.read_csv(SENTIMENT140_CSV, usecols=[2, 5], header=None, names=['timestamp', 'text'])
    df_senti['timestamp'] = df_senti['timestamp'].str.replace(r' [A-Z]{3,4} \d{4}$', '', regex=True)
    df_senti['timestamp'] = pd.to_datetime(df_senti['timestamp'], format='%a %b %d %H:%M:%S', errors='coerce')
    df_senti['platform'] = 'twitter'

    df = pd.concat([df_amzn, df_senti]).dropna(subset=['timestamp', 'text'])
    df = df.sort_values('timestamp')

    # Normalize years
    def safe_replace_year(ts, year):
        try:
            return ts.replace(year=year)
        except ValueError:
        # fallback: if Feb 29 is invalid, use Feb 28
            if ts.month == 2 and ts.day == 29:
                return ts.replace(year=year, day=28)
        # fallback: if still invalid, return original
            return ts

    now = pd.Timestamp.now()
    df["timestamp"] = df["timestamp"].apply(lambda t: safe_replace_year(t, now.year) if t.year < 2020 else t)

    print(f"✅ Mentions simulation DataFrame created: {len(df)} rows")
    return df
def stream_data(df, output_path, columns, name="stream", sleep=SLEEP_SECONDS, limit=LOOP_LIMIT):
    print(f"🔁 Starting stream for {name} → {output_path}")
    if not Path(output_path).exists():
        Path(output_path).write_text(','.join(columns) + '\n')

    i = 0
    while True if limit is None else i < limit:
        batch = df.sample(n=BATCH_SIZE)
        batch.to_csv(output_path, mode='a', header=False, index=False)
        print(f"[{name}] ⬆️ Sent batch {i + 1} ({len(batch)} rows)")
        time.sleep(sleep)
        i += 1

if __name__ == "__main__":
    sales_df = prepare_sales_simulation_demand(STORE_ITEM_CSV, SKUS, 30)
    mentions_df = prepare_mentions_feed()

    Thread(target=stream_data, args=(sales_df, SALES_OUT, ['timestamp', 'store_id', 'sku', 'quantity', 'price'], 'Sales')).start()
    Thread(target=stream_data, args=(mentions_df, MENTIONS_OUT, ['timestamp', 'text', 'platform'], 'Mentions')).start()
