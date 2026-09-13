# Production-Grade Two-Stage Recommendation & Ranking System

[![RecSys CI Pipeline](https://github.com/thimmareddygt7/ranking_recommendation_system/actions/workflows/ci.yml/badge.svg)](https://github.com/thimmareddygt7/ranking_recommendation_system/actions)
![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

An end-to-end, production-grade recommendation engine built on the Retailrocket e-commerce dataset (~2.75M interactions). The architecture implements an industry-standard **Two-Stage Cascade (Retrieval + Ranking)** to deliver sub-20ms personalized recommendations while maximizing conversion rates.

---

## 🏗️ System Architecture

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
      FastAPI Endpoints (Cold-start fallback for new users)
                                 |
                                 v
                 Top-K Personalized Output (< 20 ms)
