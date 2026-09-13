import os
import pickle
import numpy as np
import pandas as pd
import lightgbm as lgb

PROCESSED_PATH = "data/processed"
MODELS_PATH = "models"
RESULTS_PATH = "results"
os.makedirs(RESULTS_PATH, exist_ok=True)

def compute_ndcg_at_k(actual_items, ranked_items, k=10):
    """Calculates NDCG@k for a single user ranking."""
    ranked_k = ranked_items[:k]
    dcg = 0.0
    for idx, item in enumerate(ranked_k):
        if item in actual_items:
            dcg += 1.0 / np.log2(idx + 2)
            
    # Ideal DCG
    idcg = sum([1.0 / np.log2(i + 2) for i in range(min(len(actual_items), k))])
    if idcg == 0:
        return 0.0
    return dcg / idcg

def compute_recall_at_k(actual_items, ranked_items, k=10):
    """Calculates Recall@k for a single user."""
    ranked_k = set(ranked_items[:k])
    actual = set(actual_items)
    if not actual:
        return 0.0
    return len(ranked_k & actual) / len(actual)

def train_and_evaluate():
    print("Loading ranking dataset...")
    df = pd.read_parquet(os.path.join(PROCESSED_PATH, "ranking_dataset.parquet"))
    test_df = pd.read_parquet(os.path.join(PROCESSED_PATH, "test_events.parquet"))
    
    with open(os.path.join(MODELS_PATH, "popularity_baseline.pkl"), "rb") as f:
        top_popular_items = pickle.load(f)

    # 1. Prepare Features and Group Identifiers
    feature_cols = [
        'retrieval_score',
        'user_total_events', 'user_views', 'user_cart_adds', 'user_transactions',
        'user_days_since_last_event',
        'item_total_events', 'item_views', 'item_cart_adds', 'item_transactions',
        'item_unique_users', 'item_conversion_rate', 'item_days_since_last_event',
        'category_id'
    ]

    # Split users into 80% train and 20% validation for ranker tuning
    unique_users = df['user_id'].unique()
    np.random.seed(42)
    val_users = np.random.choice(unique_users, size=int(len(unique_users) * 0.2), replace=False)
    train_mask = ~df['user_id'].isin(val_users)
    val_mask = df['user_id'].isin(val_users)

    train_data = df[train_mask].sort_values('user_id')
    val_data = df[val_mask].sort_values('user_id')

    # Group counts per user query
    train_groups = train_data.groupby('user_id', sort=False).size().to_numpy()
    val_groups = val_data.groupby('user_id', sort=False).size().to_numpy()

    X_train, y_train = train_data[feature_cols], train_data['label']
    X_val, y_val = val_data[feature_cols], val_data['label']

    print(f"Training on {len(train_groups):,} users, validating on {len(val_groups):,} users...")

    # 2. Train LightGBM Ranker (LambdaMART)
    lgb_train = lgb.Dataset(X_train, label=y_train, group=train_groups)
    lgb_val = lgb.Dataset(X_val, label=y_val, group=val_groups, reference=lgb_train)

    params = {
        'objective': 'lambdarank',
        'metric': 'ndcg',
        'eval_at': [5, 10, 20],
        'learning_rate': 0.05,
        'num_leaves': 31,
        'min_data_in_leaf': 20,
        'feature_fraction': 0.8,
        'random_state': 42,
        'verbose': -1
    }

    ranker = lgb.train(
        params,
        lgb_train,
        num_boost_round=150,
        valid_sets=[lgb_val],
        callbacks=[lgb.early_stopping(50, verbose=False)]
    )

    # Save Ranker Artifact
    ranker.save_model(os.path.join(MODELS_PATH, "ranker_model.txt"))
    print(f"Ranker saved to {MODELS_PATH}/ranker_model.txt")

    # 3. Full Benchmark on Validation Users
    print("\nEvaluating Popularity vs. ALS vs. LightGBM Ranker on Held-out Users...")
    
    val_data = val_data.copy()
    val_data['predicted_score'] = ranker.predict(X_val)

    # Ground truth actual test events per user
    user_actuals = test_df.groupby('user_id')['item_id'].apply(set).to_dict()

    pop_ndcg, pop_recall = [], []
    als_ndcg, als_recall = [], []
    lgb_ndcg, lgb_recall = [], []

    for uid in val_users:
        actuals = user_actuals.get(uid, set())
        if not actuals:
            continue

        user_rows = val_data[val_data['user_id'] == uid]
        if len(user_rows) == 0:
            continue

        # A. Popularity Baseline
        pop_ndcg.append(compute_ndcg_at_k(actuals, top_popular_items, k=10))
        pop_recall.append(compute_recall_at_k(actuals, top_popular_items, k=10))

        # B. Raw ALS Order (sorted by initial retrieval_score)
        als_ranked = user_rows.sort_values('retrieval_score', ascending=False)['item_id'].tolist()
        als_ndcg.append(compute_ndcg_at_k(actuals, als_ranked, k=10))
        als_recall.append(compute_recall_at_k(actuals, als_ranked, k=10))

        # C. LightGBM Re-Ranked Order
        lgb_ranked = user_rows.sort_values('predicted_score', ascending=False)['item_id'].tolist()
        lgb_ndcg.append(compute_ndcg_at_k(actuals, lgb_ranked, k=10))
        lgb_recall.append(compute_recall_at_k(actuals, lgb_ranked, k=10))

    benchmark_df = pd.DataFrame([
        {"Model": "1. Popularity Baseline", "NDCG@10": np.mean(pop_ndcg), "Recall@10": np.mean(pop_recall)},
        {"Model": "2. ALS Candidate Retrieval", "NDCG@10": np.mean(als_ndcg), "Recall@10": np.mean(als_recall)},
        {"Model": "3. LightGBM LambdaMART Ranker", "NDCG@10": np.mean(lgb_ndcg), "Recall@10": np.mean(lgb_recall)}
    ])

    benchmark_df.to_csv(os.path.join(RESULTS_PATH, "comparison_table.csv"), index=False)
    print("\n--- Final Model Benchmark ---")
    print(benchmark_df.to_string(index=False))

if __name__ == "__main__":
    train_and_evaluate()
