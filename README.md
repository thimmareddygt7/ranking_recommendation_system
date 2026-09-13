# Production Recommendation & Ranking System

An end-to-end, two-stage industrial recommendation pipeline trained on 2.75M+ implicit e-commerce interactions from the Retailrocket dataset. Implements sub-10ms candidate scoring via an ALS retrieval model coupled with a LightGBM LambdaMART ranker.

---

## Benchmark Results

Evaluated on held-out, future temporal interactions without lookahead leakage:

| Model Architecture | NDCG@10 | Recall@10 | Latency (p95) |
| :--- | :--- | :--- | :--- |
| **1. Global Popularity Baseline** | 0.0030 | 0.77% | < 0.1 ms |
| **2. ALS Latent Vector Retrieval** | 0.0877 | 11.86% | ~12.0 ms |
| **3. LightGBM LambdaMART Ranker** | **0.1012** | **12.54%** | **7.38 ms** |

### Segment Performance
* **Warm Users (>5 past events)**: NDCG@10 = **0.2593** | MRR@10 = **0.3163**
* **Cold Users (≤5 past events)**: NDCG@10 = **0.1037** | MRR@10 = **0.1201**

---

## System Architecture

```text
[Raw Implicit Events: view (1), cart (2), buy (3)]
                       │
                       ▼
         [Chronological Train/Test Split]
                       │
                       ▼
           [Stage 1: ALS Retrieval]
          Retrieves top-100 candidates
                       │
                       ▼
      [Stage 2: Point-in-Time Feature Store]
     (User Recency, Item Conversion, Categories)
                       │
                       ▼
        [Stage 3: LightGBM LambdaMART]
           Re-ranks top-100 candidates
                       │
                       ▼
        [FastAPI Real-Time Serving Layer]
      Warm: Ranked Top-K (7.4ms)
      Cold: Fallback Popularity Baseline (0.05ms)
