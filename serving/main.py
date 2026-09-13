import time
import os
import pickle
import pandas as pd
import numpy as np
import lightgbm as lgb
from fastapi import FastAPI, HTTPException
from serving.schemas import RecommendationResponse, RecommendationItem
from serving.cold_start import ColdStartHandler

app = FastAPI(title="Production Recommendation & Ranking API", version="1.0.0")

# In-memory artifact holders
MODELS_PATH = "models"
PROCESSED_PATH = "data/processed"

ranker = None
cold_start_handler = None
feature_store = None
candidate_store = None

feature_cols = [
    'retrieval_score',
    'user_total_events', 'user_views', 'user_cart_adds', 'user_transactions',
    'user_days_since_last_event',
    'item_total_events', 'item_views', 'item_cart_adds', 'item_transactions',
    'item_unique_users', 'item_conversion_rate', 'item_days_since_last_event',
    'category_id'
]

@app.on_event("startup")
def load_artifacts():
    global ranker, cold_start_handler, feature_store, candidate_store
    print("Loading models and candidate stores into memory...")

    # 1. Load trained LightGBM ranker
    ranker = lgb.Booster(model_file=os.path.join(MODELS_PATH, "ranker_model.txt"))

    # 2. Load cold-start fallback handler
    cold_start_handler = ColdStartHandler()

    # 3. Load precomputed candidates and pre-engineered feature table
    full_df = pd.read_parquet(os.path.join(PROCESSED_PATH, "ranking_dataset.parquet"))

    # Group candidates and features by user_id for fast in-memory indexing
    candidate_store = {
        uid: group[['item_id'] + feature_cols].copy()
        for uid, group in full_df.groupby('user_id')
    }
    print(f"Server ready. Loaded candidate profiles for {len(candidate_store):,} users.")

@app.get("/recommend/{user_id}", response_model=RecommendationResponse)
def get_recommendations(user_id: int, k: int = 10):
    start_time = time.perf_counter()

    # Cold start check
    if user_id not in candidate_store:
        fallback_items = cold_start_handler.get_fallback_recommendations(k=k)
        recs = [
            RecommendationItem(item_id=iid, score=0.0, rank=idx + 1)
            for idx, iid in enumerate(fallback_items)
        ]
        latency = (time.perf_counter() - start_time) * 1000.0
        return RecommendationResponse(
            user_id=user_id,
            is_cold_start=True,
            recommendations=recs,
            latency_ms=round(latency, 2)
        )

    # Fetch user candidate records
    user_candidates = candidate_store[user_id].copy()

    # Predict ranking scores on candidate pool
    scores = ranker.predict(user_candidates[feature_cols])
    user_candidates['score'] = scores

    # Sort and take top-k
    top_items = user_candidates.sort_values('score', ascending=False).head(k)

    recommendations = [
        RecommendationItem(
            item_id=int(row['item_id']),
            score=float(round(row['score'], 4)),
            rank=idx + 1
        )
        for idx, (_, row) in enumerate(top_items.iterrows())
    ]

    latency = (time.perf_counter() - start_time) * 1000.0

    return RecommendationResponse(
        user_id=user_id,
        is_cold_start=False,
        recommendations=recommendations,
        latency_ms=round(latency, 2)
    )
