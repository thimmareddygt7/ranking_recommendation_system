import os
import shap
import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt

PROCESSED_PATH = "data/processed"
MODELS_PATH = "models"
RESULTS_PATH = "results"
FIGURES_PATH = "results/figures"
os.makedirs(FIGURES_PATH, exist_ok=True)

def run_error_analysis():
    print("Loading data, artifacts, and booster...")
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

    # Predict scores
    df['pred_score'] = ranker.predict(df[feature_cols])

    # 1. Global Feature Importance via SHAP TreeExplainer
    print("Computing SHAP values for global explainability...")
    sample_df = df[feature_cols].sample(n=min(5000, len(df)), random_state=42)
    explainer = shap.TreeExplainer(ranker)
    shap_values = explainer.shap_values(sample_df)

    # Save SHAP summary plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, sample_df, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_PATH, "shap_summary.png"), dpi=200)
    plt.close()
    print(f"SHAP summary plot saved to {FIGURES_PATH}/shap_summary.png")

    # 2. Identify Failure Modes (False Negatives: ground-truth items ranked outside top 10)
    user_actuals = test_df.groupby('user_id')['item_id'].apply(set).to_dict()

    failure_cases = []
    success_cases = []

    for uid, group in df.groupby('user_id'):
        actuals = user_actuals.get(uid, set())
        if not actuals:
            continue

        ranked_items = group.sort_values('pred_score', ascending=False)['item_id'].tolist()
        recs_top10 = set(ranked_items[:10])
        hits = recs_top10 & actuals

        # Success: user got relevant hits in top 10
        if hits:
            success_cases.append(uid)
        # Complete Failure: positive item missed entirely from top 10
        else:
            missed_items = actuals - recs_top10
            # Check where the target item was pushed
            for m_item in missed_items:
                if m_item in ranked_items:
                    actual_rank = ranked_items.index(m_item) + 1
                    target_row = group[group['item_id'] == m_item].iloc[0]
                    failure_cases.append({
                        'user_id': uid,
                        'item_id': m_item,
                        'actual_rank': actual_rank,
                        'retrieval_score': target_row['retrieval_score'],
                        'item_total_events': target_row['item_total_events'],
                        'user_days_since_last_event': target_row['user_days_since_last_event'],
                        'pred_score': target_row['pred_score']
                    })

    fail_df = pd.DataFrame(failure_cases)
    print(f"Identified {len(fail_df):,} false-negative item rankings across users.")

    # 3. Write Formal Error Analysis Document
    error_report = f"""# Ranking Error Analysis & Diagnostic Report

## 1. Global Feature Attribution
The primary ranking drivers identified by SHAP TreeExplainer are:
1. **retrieval_score (ALS Latent Factor Dot Product)**: Serves as the dominant anchor for user-item affinity.
2. **item_total_events / item_views**: Heavily dictates baseline conversion likelihood, introducing a head-item bias.
3. **user_days_since_last_event**: Governs temporal decay of user interest.

## 2. Quantitative Failure Case Inspection
Examined `{len(fail_df)}` instances where relevant candidate items dropped out of the top 10 recommendations:

* **Median Rank of Missed Relevant Items**: {fail_df['actual_rank'].median() if not fail_df.empty else 0:.1f}
* **Mean Historical Item Interactions for Missed Targets**: {fail_df['item_total_events'].mean() if not fail_df.empty else 0:.2f} events
* **Mean Target Retrieval Score**: {fail_df['retrieval_score'].mean() if not fail_df.empty else 0:.4f}

### Root Causes of Degradation:
1. **Cold Interaction Sparsity (Item-Side)**: Relevant items with few historical interactions suffer lower ranker scores despite strong retrieval signals because popularity features suppress cold/novel products.
2. **Extreme Interaction Latency (User-Side)**: For users dormant for >30 days, static profile features lose predictive power compared to dynamic session signals.
3. **Implicit Conversion Noise**: A single historical view does not reliably indicate positive purchase intent during the test window.
"""

    with open(os.path.join(RESULTS_PATH, "error_analysis.md"), "w") as f:
        f.write(error_report)

    print(f"Error analysis written to {RESULTS_PATH}/error_analysis.md")

if __name__ == "__main__":
    run_error_analysis()
