import os
import numpy as np
import pandas as pd
import lightgbm as lgb

PROCESSED_PATH = "data/processed"
MODELS_PATH = "models"
RESULTS_PATH = "results"

def calculate_psi(baseline, target, num_buckets=10):
    """Computes Population Stability Index (PSI) between two score distributions."""
    # Create quantiles based on baseline scores
    quantiles = np.linspace(0, 100, num_buckets + 1)
    bucket_bounds = np.percentile(baseline, quantiles)
    bucket_bounds[0] -= 1e-5
    bucket_bounds[-1] += 1e-5

    # Count occurrences in each quantile
    base_counts, _ = np.histogram(baseline, bins=bucket_bounds)
    target_counts, _ = np.histogram(target, bins=bucket_bounds)

    # Convert to fractions with Laplace smoothing
    base_fractions = (base_counts + 1e-4) / (len(baseline) + 1e-4 * num_buckets)
    target_fractions = (target_counts + 1e-4) / (len(target) + 1e-4 * num_buckets)

    # PSI calculation formula: sum((target - base) * ln(target / base))
    psi_value = np.sum((target_fractions - base_fractions) * np.log(target_fractions / base_fractions))
    return psi_value

def run_drift_check():
    print("Loading test data and ranker for score drift monitoring...")
    df = pd.read_parquet(os.path.join(PROCESSED_PATH, "ranking_dataset.parquet"))
    test_df = pd.read_parquet(os.path.join(PROCESSED_PATH, "test_events.parquet"))
    ranker = lgb.Booster(model_file=os.path.join(MODELS_PATH, "ranker_model.txt"))

    feature_cols = [
        'retrieval_score',
        'user_total_events', 'user_views', 'user_cart_adds', 'user_transactions',
        'user_days_since_last_event',
        'item_total_events', 'item_views', 'item_cart_adds', 'item_transactions',
        'item_unique_users', 'item_conversion_rate', 'item_days_since_last_event',
        'category_id'
    ]

    # Predict scores for the entire candidate pool
    df['pred_score'] = ranker.predict(df[feature_cols])

    # Join timestamps to split predictions chronologically into two windows (W1 vs W2)
    user_times = test_df.groupby('user_id')['datetime'].min().reset_index()
    merged = df.merge(user_times, on='user_id', how='left')

    median_time = merged['datetime'].median()
    baseline_scores = merged[merged['datetime'] <= median_time]['pred_score'].dropna().values
    target_scores = merged[merged['datetime'] > median_time]['pred_score'].dropna().values

    psi = calculate_psi(baseline_scores, target_scores, num_buckets=10)

    if psi < 0.10:
        status = "HEALTHY (No significant drift detected)"
    elif psi < 0.25:
        status = "WARNING (Moderate score shift observed)"
    else:
        status = "ALERT (Critical drift: Trigger model retraining)"

    print("\n--- Model Score Drift Check ---")
    print(f"Baseline window score count: {len(baseline_scores):,}")
    print(f"Target window score count:   {len(target_scores):,}")
    print(f"Calculated PSI:              {psi:.5f}")
    print(f"Monitoring Status:           {status}")

if __name__ == "__main__":
    run_drift_check()
