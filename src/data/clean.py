import pandas as pd
import numpy as np
import os

RAW_PATH = "data/raw/retailrocket"
PROCESSED_PATH = "data/processed"
os.makedirs(PROCESSED_PATH, exist_ok=True)

def clean_and_split():
    print("Loading events...")
    df = pd.read_csv(os.path.join(RAW_PATH, "events.csv"))
    
    # Standardize column names
    df.rename(columns={'visitorid': 'user_id', 'itemid': 'item_id'}, inplace=True)
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
    
    # Assign implicit weights: view=1, addtocart=2, transaction=3
    weight_map = {'view': 1, 'addtocart': 2, 'transaction': 3}
    df['event_weight'] = df['event'].map(weight_map)

    # 1. Deduplication
    df = df.drop_duplicates(subset=['user_id', 'item_id', 'timestamp'])

    # 2. Filter bots & high-frequency anomalies
    user_counts = df['user_id'].value_counts()
    valid_users = user_counts[(user_counts >= 3) & (user_counts <= 500)].index
    df = df[df['user_id'].isin(valid_users)].copy()

    # 3. Compute sparsity profile
    n_users = df['user_id'].nunique()
    n_items = df['item_id'].nunique()
    n_interactions = len(df)
    sparsity = 1.0 - (n_interactions / (n_users * n_items))

    print("--- Dataset Profile (Cleaned) ---")
    print(f"Users: {n_users:,}")
    print(f"Items: {n_items:,}")
    print(f"Interactions: {n_interactions:,}")
    print(f"Sparsity: {sparsity * 100:.4f}%")

    # 4. Chronological Split (Train: first 85%, Test: last 15%)
    df = df.sort_values('timestamp').reset_index(drop=True)
    cutoff_idx = int(len(df) * 0.85)
    cutoff_time = df.loc[cutoff_idx, 'datetime']

    train_df = df[df['datetime'] < cutoff_time].copy()
    test_df = df[df['datetime'] >= cutoff_time].copy()

    # Only evaluate on test users that appeared in training (warm evaluation)
    test_df = test_df[test_df['user_id'].isin(train_df['user_id'].unique())]

    print(f"\nCutoff timestamp: {cutoff_time}")
    print(f"Train set: {len(train_df):,} events")
    print(f"Test set: {len(test_df):,} events")

    # 5. Save processed data
    train_df.to_parquet(os.path.join(PROCESSED_PATH, "train_events.parquet"), index=False)
    test_df.to_parquet(os.path.join(PROCESSED_PATH, "test_events.parquet"), index=False)
    print(f"\nSaved train and test sets to {PROCESSED_PATH}/")

if __name__ == "__main__":
    clean_and_split()
