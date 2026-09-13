import os
import pickle
import pandas as pd
import numpy as np
import scipy.sparse as sp
import implicit

PROCESSED_PATH = "data/processed"
MODELS_PATH = "models"
os.makedirs(MODELS_PATH, exist_ok=True)

def train_retrieval_models(top_n: int = 100):
    print("Loading processed train and test data...")
    train_df = pd.read_parquet(os.path.join(PROCESSED_PATH, "train_events.parquet"))
    test_df = pd.read_parquet(os.path.join(PROCESSED_PATH, "test_events.parquet"))

    # 1. Popularity Baseline
    print("Computing Popularity Baseline...")
    item_pop = (
        train_df.groupby('item_id')['event_weight']
        .sum()
        .sort_values(ascending=False)
        .reset_index()
    )
    top_popular_items = item_pop['item_id'].head(top_n).tolist()
    
    with open(os.path.join(MODELS_PATH, "popularity_baseline.pkl"), "wb") as f:
        pickle.dump(top_popular_items, f)

    # 2. Map Categorical IDs to Continuous Indices for Sparse Matrix
    unique_users = train_df['user_id'].unique()
    unique_items = train_df['item_id'].unique()

    user_to_idx = {uid: i for i, uid in enumerate(unique_users)}
    idx_to_user = {i: uid for uid, i in user_to_idx.items()}
    item_to_idx = {iid: i for i, iid in enumerate(unique_items)}
    idx_to_item = {i: iid for iid, i in item_to_idx.items()}

    # Save ID index mappings
    mappings = {
        'user_to_idx': user_to_idx,
        'idx_to_user': idx_to_user,
        'item_to_idx': item_to_idx,
        'idx_to_item': idx_to_item
    }
    with open(os.path.join(MODELS_PATH, "id_mappings.pkl"), "wb") as f:
        pickle.dump(mappings, f)

    # Aggregate interaction weights for user-item pairs
    grouped_train = train_df.groupby(['user_id', 'item_id'])['event_weight'].sum().reset_index()
    
    rows = grouped_train['user_id'].map(user_to_idx).values
    cols = grouped_train['item_id'].map(item_to_idx).values
    data = grouped_train['event_weight'].values

    user_item_matrix = sp.csr_matrix(
        (data, (rows, cols)), 
        shape=(len(unique_users), len(unique_items)),
        dtype=np.float32
    )

    # 3. Train Alternating Least Squares (ALS)
    print("Fitting Alternating Least Squares (ALS) model...")
    als_model = implicit.als.AlternatingLeastSquares(
        factors=64,
        regularization=0.05,
        iterations=20,
        random_state=42
    )
    # implicit expects user_item sparse matrix
    als_model.fit(user_item_matrix)

    # Save ALS model
    with open(os.path.join(MODELS_PATH, "als_model.pkl"), "wb") as f:
        pickle.dump(als_model, f)

    # 4. Generate Top-N Candidates for Test Users
    test_users = test_df['user_id'].unique()
    print(f"Generating top-{top_n} candidates for {len(test_users):,} test users...")

    test_user_indices = np.array([user_to_idx[uid] for uid in test_users if uid in user_to_idx])
    
    # Batch recommend
    ids, scores = als_model.recommend(
        test_user_indices, 
        user_item_matrix[test_user_indices], 
        N=top_n, 
        filter_already_liked_items=False
    )

    # Build candidate pairs table
    candidates = []
    for u_idx, item_indices, score_list in zip(test_user_indices, ids, scores):
        uid = idx_to_user[u_idx]
        for i_idx, score in zip(item_indices, score_list):
            candidates.append({
                'user_id': uid,
                'item_id': idx_to_item[i_idx],
                'retrieval_score': float(score)
            })

    candidate_df = pd.DataFrame(candidates)
    candidate_df.to_parquet(os.path.join(PROCESSED_PATH, "retrieval_candidates.parquet"), index=False)
    print(f"Generated {len(candidate_df):,} candidates saved to {PROCESSED_PATH}/retrieval_candidates.parquet")

if __name__ == "__main__":
    train_retrieval_models(top_n=100)
