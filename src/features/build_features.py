import os
import pandas as pd
import numpy as np

PROCESSED_PATH = "data/processed"
RAW_PATH = "data/raw/retailrocket"

def extract_item_categories():
    print("Extracting item categories from metadata...")
    prop_path = os.path.join(RAW_PATH, "item_properties_part1.csv")
    if not os.path.exists(prop_path):
        print("Property file not found, continuing without explicit categories...")
        return None

    # Read properties and filter for categoryid
    props = pd.read_csv(prop_path)
    cat_props = props[props['property'] == 'categoryid'][['itemid', 'value']].drop_duplicates('itemid')
    cat_props.rename(columns={'itemid': 'item_id', 'value': 'category_id'}, inplace=True)
    cat_props['category_id'] = pd.to_numeric(cat_props['category_id'], errors='coerce').fillna(-1).astype(int)
    return cat_props

def build_features():
    print("Loading training events and retrieval candidates...")
    train_df = pd.read_parquet(os.path.join(PROCESSED_PATH, "train_events.parquet"))
    test_df = pd.read_parquet(os.path.join(PROCESSED_PATH, "test_events.parquet"))
    candidates = pd.read_parquet(os.path.join(PROCESSED_PATH, "retrieval_candidates.parquet"))

    max_train_time = train_df['datetime'].max()

    # 1. User Profile Features (Computed strictly on train set)
    print("Engineering user features...")
    user_grp = train_df.groupby('user_id')

    user_features = user_grp.agg(
        user_total_events=('event', 'count'),
        user_views=('event', lambda x: (x == 'view').sum()),
        user_cart_adds=('event', lambda x: (x == 'addtocart').sum()),
        user_transactions=('event', lambda x: (x == 'transaction').sum()),
        user_last_timestamp=('datetime', 'max')
    ).reset_index()

    # Recency in days relative to split cutoff
    user_features['user_days_since_last_event'] = (
        (max_train_time - user_features['user_last_timestamp']).dt.total_seconds() / 86400.0
    ).fillna(999.0)
    user_features.drop(columns=['user_last_timestamp'], inplace=True)

    # 2. Item Profile Features (Computed strictly on train set)
    print("Engineering item features...")
    item_grp = train_df.groupby('item_id')

    item_features = item_grp.agg(
        item_total_events=('event', 'count'),
        item_views=('event', lambda x: (x == 'view').sum()),
        item_cart_adds=('event', lambda x: (x == 'addtocart').sum()),
        item_transactions=('event', lambda x: (x == 'transaction').sum()),
        item_unique_users=('user_id', 'nunique'),
        item_last_timestamp=('datetime', 'max')
    ).reset_index()

    # Conversion rate: transactions / (views + 1)
    item_features['item_conversion_rate'] = (
        item_features['item_transactions'] / (item_features['item_views'] + 1.0)
    )
    item_features['item_days_since_last_event'] = (
        (max_train_time - item_features['item_last_timestamp']).dt.total_seconds() / 86400.0
    ).fillna(999.0)
    item_features.drop(columns=['item_last_timestamp'], inplace=True)

    # Optional: Attach categories
    cat_df = extract_item_categories()
    if cat_df is not None:
        item_features = item_features.merge(cat_df, on='item_id', how='left')
        item_features['category_id'] = item_features['category_id'].fillna(-1).astype(int)
    else:
        item_features['category_id'] = -1

    # 3. Ground Truth Labels from Held-Out Test Set
    print("Assigning ground-truth labels to candidate pairs...")
    test_ground_truth = (
        test_df.groupby(['user_id', 'item_id'])['event_weight']
        .max()
        .reset_index()
    )
    # Binary label: 1 if user interacted in test period, 0 otherwise
    test_ground_truth['label'] = 1

    # Merge candidates with ground truth
    dataset = candidates.merge(
        test_ground_truth[['user_id', 'item_id', 'label']],
        on=['user_id', 'item_id'],
        how='left'
    )
    dataset['label'] = dataset['label'].fillna(0).astype(int)

    # 4. Join User & Item Features
    print("Merging user and item feature tables...")
    dataset = dataset.merge(user_features, on='user_id', how='left')
    dataset = dataset.merge(item_features, on='item_id', how='left')

    # Fill missing values for long-tail items not seen in train aggregations
    dataset = dataset.fillna(0)

    # Sort strictly by user_id to ensure proper LightGBM query grouping
    dataset = dataset.sort_values('user_id').reset_index(drop=True)

    print(f"Final feature dataset shape: {dataset.shape}")
    print(f"Positive labels: {dataset['label'].sum():,} ({dataset['label'].mean() * 100:.2f}%)")

    dataset.to_parquet(os.path.join(PROCESSED_PATH, "ranking_dataset.parquet"), index=False)
    print(f"Saved ranking dataset to {PROCESSED_PATH}/ranking_dataset.parquet")

if __name__ == "__main__":
    build_features()
