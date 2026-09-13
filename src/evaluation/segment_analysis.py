import os
import pickle
import numpy as np
import pandas as pd
import lightgbm as lgb

PROCESSED_PATH = "data/processed"
MODELS_PATH = "models"
RESULTS_PATH = "results"
os.makedirs(RESULTS_PATH, exist_ok=True)

# ----------------- Ranking Metrics -----------------

def compute_ndcg_at_k(actual_items, ranked_items, k=10):
    ranked_k = ranked_items[:k]
    dcg = sum([1.0 / np.log2(idx + 2) for idx, item in enumerate(ranked_k) if item in actual_items])
    idcg = sum([1.0 / np.log2(i + 2) for i in range(min(len(actual_items), k))])
    return (dcg / idcg) if idcg > 0 else 0.0

def compute_map_at_k(actual_items, ranked_items, k=10):
    ranked_k = ranked_items[:k]
    hits, score = 0, 0.0
    for idx, item in enumerate(ranked_k):
        if item in actual_items:
            hits += 1
            score += hits / (idx + 1.0)
    return (score / min(len(actual_items), k)) if actual_items else 0.0

def compute_mrr(actual_items, ranked_items, k=10):
    ranked_k = ranked_items[:k]
    for idx, item in enumerate(ranked_k):
        if item in actual_items:
            return 1.0 / (idx + 1.0)
    return 0.0

# ----------------- Segment Analysis Pipeline -----------------

def run_segment_analysis():
    print("Loading data and model for segment evaluation...")
    df = pd.read_parquet(os.path.join(PROCESSED_PATH, "ranking_dataset.parquet"))
    test_df = pd.read_parquet(os.path.join(PROCESSED_PATH, "test_events.parquet"))
    train_df = pd.read_parquet(os.path.join(PROCESSED_PATH, "train_events.parquet"))

    ranker = lgb.Booster(model_file=os.path.join(MODELS_PATH, "ranker_model.txt"))

    feature_cols = [
        'retrieval_score',
        'user_total_events', 'user_views', 'user_cart_adds', 'user_transactions',
        'user_days_since_last_event',
        'item_total_events', 'item_views', 'item_cart_adds', 'item_transactions',
        'item_unique_users', 'item_conversion_rate', 'item_days_since_last_event',
        'category_id'
    ]

    # Predict LightGBM scores
    df['pred_score'] = ranker.predict(df[feature_cols])

    # Ground truth mapping
    user_actuals = test_df.groupby('user_id')['item_id'].apply(set).to_dict()

    # Define User Segments: Cold (<= 5 events) vs Warm (> 5 events)
    user_counts = train_df.groupby('user_id').size().to_dict()

    # Define Item Segments: Head (top 20% by interaction) vs Long-Tail (bottom 80%)
    item_counts = train_df.groupby('item_id').size().sort_values(ascending=False)
    top_20_pct_idx = int(len(item_counts) * 0.20)
    head_items = set(item_counts.iloc[:top_20_pct_idx].index)

    # Collect per-user metrics
    user_records = []

    for uid, group in df.groupby('user_id'):
        actuals = user_actuals.get(uid, set())
        if not actuals:
            continue

        lgb_ranked = group.sort_values('pred_score', ascending=False)['item_id'].tolist()

        ndcg = compute_ndcg_at_k(actuals, lgb_ranked, k=10)
        map_score = compute_map_at_k(actuals, lgb_ranked, k=10)
        mrr = compute_mrr(actuals, lgb_ranked, k=10)

        n_history = user_counts.get(uid, 0)
        user_segment = "Cold (<=5)" if n_history <= 5 else "Warm (>5)"

        # Check proportion of recommended items from the head
        recs_at_10 = lgb_ranked[:10]
        head_rec_ratio = sum([1 for item in recs_at_10 if item in head_items]) / 10.0

        user_records.append({
            'user_id': uid,
            'user_segment': user_segment,
            'ndcg@10': ndcg,
            'map@10': map_score,
            'mrr@10': mrr,
            'head_item_ratio@10': head_rec_ratio
        })

    eval_df = pd.DataFrame(user_records)

    # Aggregate by user segment
    user_segment_summary = eval_df.groupby('user_segment').agg({
        'ndcg@10': 'mean',
        'map@10': 'mean',
        'mrr@10': 'mean',
        'head_item_ratio@10': 'mean',
        'user_id': 'count'
    }).rename(columns={'user_id': 'user_count'})

    user_segment_summary.to_csv(os.path.join(RESULTS_PATH, "segment_user_performance.csv"))

    print("\n--- User Segment Performance Breakdown ---")
    print(user_segment_summary.round(4).to_string())

    overall = {
        'NDCG@10': eval_df['ndcg@10'].mean(),
        'MAP@10': eval_df['map@10'].mean(),
        'MRR@10': eval_df['mrr@10'].mean(),
        'Avg Head Item Exposure @ 10': eval_df['head_item_ratio@10'].mean()
    }
    print("\n--- Overall Ranker Metrics (All Active Users) ---")
    for k, v in overall.items():
        print(f"{k}: {v:.4f}")

if __name__ == "__main__":
    run_segment_analysis()
