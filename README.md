# Two-Stage Recommendation & Ranking System

[![RecSys CI Pipeline](https://github.com/thimmareddygt7/ranking_recommendation_system/actions/workflows/ci.yml/badge.svg)](https://github.com/thimmareddygt7/ranking_recommendation_system/actions)
![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## Problem

E-commerce platforms need to rank a huge item catalog down to a handful of personalized recommendations per user, in real time, from only implicit signals (views, cart-adds, purchases) rather than explicit ratings. This project builds a production-style **two-stage retrieval + ranking system** on the [Retailrocket e-commerce dataset](https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset) (~2.75M raw interaction events) and evaluates it the way a real recommender system is evaluated: against a chronological holdout, with ranking-specific metrics, segmented by user activity level.

## Architecture

```text
               +----------------------------------+
               | User Request (visitorid, top_k)  |
               +-----------------+----------------+
                                 |
                                 v
        [ Stage 1: Candidate Generation (~4.2 ms) ]
       Implicit ALS Matrix Factorization (Factors=64)
            Retrieves Top-100 High-Recall Items
                                 |
                                 v
        [ Stage 2: LambdaMART Ranking (~8.6 ms) ]
      LightGBM Ranker trained with lambdarank objective
      Ranks candidates using ALS scores, item conversion rates,
      user activity levels, and session-level dwell momentum
                                 |
                                 v
        [ Business Fallback & Real-time Serving ]
      FastAPI Endpoints (Cold-start fallback via popularity baseline)
                                 |
                                 v
                 Top-K Personalized Output (< 20 ms)
```

## Results

Evaluated on a chronologically held-out test period (never seen during training), against 2,255 users used for training the ranker and 563 held out for validation.

| Model | NDCG@10 | Recall@10 |
|---|---|---|
| 1. Popularity Baseline | 0.0061 | 0.0094 |
| 2. ALS Candidate Retrieval | 0.0814 | 0.1100 |
| 3. **LightGBM LambdaMART Ranker** | **0.1016** | **0.1349** |

The full two-stage pipeline delivers a **~16.5x lift in NDCG@10** over the popularity baseline, and a **~25% relative lift over ALS retrieval alone** — showing that re-ranking candidates with learned features meaningfully outperforms both a naive baseline and raw retrieval order.

### Segment breakdown (all active users)

| Segment | NDCG@10 | MAP@10 | MRR@10 | Users |
|---|---|---|---|---|
| Cold (≤5 interactions) | 0.0887 | 0.0761 | 0.0963 | 2,229 |
| Warm (>5 interactions) | 0.2581 | 0.2239 | 0.3124 | 589 |

**Overall ranker metrics (all active users):** NDCG@10 = 0.1241, MAP@10 = 0.1070, MRR@10 = 0.1415.

Warm users see roughly 3x the ranking quality of cold users — expected given sparser signal for new users, and the reason this project includes a dedicated cold-start path rather than treating all users identically.

### Model monitoring

A population stability index (PSI) check between two scoring windows (140,900 scores each) returned **PSI = 0.0184**, below the standard 0.1 drift threshold — flagged `HEALTHY` with no significant drift detected. See `src/monitoring/` for the drift-check implementation.

## Key design decisions

- **Chronological train/test split**, not random — the model is evaluated on genuinely future interactions relative to training, avoiding leakage that inflates offline metrics.
- **Weighted implicit feedback** (view / add-to-cart / transaction weighted differently) rather than collapsing all events to a binary label.
- **Two-stage cascade**: ALS handles high-recall candidate generation cheaply over the full catalog; LightGBM re-ranks only the ~100 retrieved candidates, keeping inference fast (~13ms combined).
- **Segmented evaluation** (cold vs. warm users) rather than a single aggregate metric, since aggregate NDCG hides how the model performs for the users who need good recommendations most.
- **`filter_already_liked_items=False`** in ALS retrieval — repeat purchases/views are common and meaningful in e-commerce, so previously-seen items remain eligible candidates rather than being excluded by default.
- **Error analysis over vibes**: `results/error_analysis.md` documents 150 concrete failure cases (relevant items that dropped out of the top 10) with SHAP-based root-cause attribution, not just a summary metric.

## Project structure

```
├── notebooks/              # EDA → features → candidate gen → ranking → evaluation, in pipeline order
├── src/
│   ├── data/                # Loading, cleaning, chronological split
│   ├── features/             # User, item, and session feature engineering
│   ├── retrieval/            # ALS candidate generation
│   ├── ranking/               # LightGBM LambdaMART training
│   ├── evaluation/            # NDCG/MAP/MRR/Recall implementations
│   └── monitoring/           # PSI drift checks
├── serving/                 # FastAPI app, request/response schemas, cold-start handler
├── models/                  # Saved ranker (ranker_model.txt) and popularity baseline
├── results/                  # Error analysis, SHAP figures, pipeline status
├── docs/                     # Architecture diagram, proposed A/B test design
├── tests/                    # pytest suite: metrics, pipeline, serving, session features
├── .github/workflows/         # CI: runs the test suite on every push/PR
└── Dockerfile / docker-compose.yml
```

## How to run locally

```bash
git clone https://github.com/thimmareddygt7/ranking_recommendation_system.git
cd ranking_recommendation_system
pip install -r requirements.txt

# Run the test suite
pytest tests/ -v

# Serve the API locally
uvicorn serving.main:app --reload
# then: POST /recommend  {"user_id": 12345, "top_k": 10}
```

Or with Docker:
```bash
docker-compose up --build
```

To reproduce the pipeline end-to-end, run the notebooks in `notebooks/` in numeric order (`01_eda.ipynb` through `05_evaluation_error_analysis.ipynb`), or run `src/pipeline.py` directly against a locally downloaded copy of the [Retailrocket dataset](https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset).

## What I'd do next in production

- Replace the offline PSI check with a live monitoring dashboard and alerting
- Run an actual online A/B test (design proposed in `docs/ab_test_proposal.md`) rather than relying solely on offline metrics
- Extend cold-start beyond a static popularity fallback to a contextual bandit for faster personalization on new users
- Explore a learned two-tower retrieval model as a candidate-generation upgrade over ALS

## License

MIT
