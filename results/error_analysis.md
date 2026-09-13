# Ranking Error Analysis & Diagnostic Report

## 1. Global Feature Attribution
The primary ranking drivers identified by SHAP TreeExplainer are:
1. **retrieval_score (ALS Latent Factor Dot Product)**: Serves as the dominant anchor for user-item affinity.
2. **item_total_events / item_views**: Heavily dictates baseline conversion likelihood, introducing a head-item bias.
3. **user_days_since_last_event**: Governs temporal decay of user interest.

## 2. Quantitative Failure Case Inspection
Examined `109` instances where relevant candidate items dropped out of the top 10 recommendations:

* **Median Rank of Missed Relevant Items**: 36.0
* **Mean Historical Item Interactions for Missed Targets**: 29.40 events
* **Mean Target Retrieval Score**: 0.0703

### Root Causes of Degradation:
1. **Cold Interaction Sparsity (Item-Side)**: Relevant items with few historical interactions suffer lower ranker scores despite strong retrieval signals because popularity features suppress cold/novel products.
2. **Extreme Interaction Latency (User-Side)**: For users dormant for >30 days, static profile features lose predictive power compared to dynamic session signals.
3. **Implicit Conversion Noise**: A single historical view does not reliably indicate positive purchase intent during the test window.
