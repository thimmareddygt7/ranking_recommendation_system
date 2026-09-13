# Production A/B Testing Proposal: Two-Stage Ranker vs. Popularity Baseline

## 1. Objective & Hypothesis
* **Business Context**: The current candidate generation system serves global popularity baselines to cold users and simple top-ranked items.
* **Hypothesis**: Replacing the raw popularity baseline with an Alternating Least Squares (ALS) candidate retrieval layer + LightGBM LambdaMART re-ranker will increase Click-Through Rate (CTR) and item discovery without violating latency Service Level Objectives (SLOs < 50ms).

---

## 2. Experiment Setup
* **Unit of Randomization**: `user_id` (hashed via SHA-256 modulo 100 to ensure consistent user bucket assignment across sessions).
* **Target Audience**: All active authenticated users with at least 1 historical interaction.
* **Duration**: 14 days (to capture full weekly seasonality and day-of-week purchase cycles).

### Variant Definitions:
| Variant | Logic | Description |
| :--- | :--- | :--- |
| **Control (Group A - 50%)** | Popularity Floor | Top-k globally popular items over a rolling 7-day window. |
| **Treatment (Group B - 50%)** | ALS + LambdaMART | Top-100 candidates retrieved via ALS latent dot products, re-ranked via LightGBM. |

---

## 3. Metrics Framework

### Primary Metric (Decision Driver):
* **Recommendation Click-Through Rate (CTR@10)**:
  $$\text{CTR@10} = \frac{\sum \text{Clicks on Top-10 Recommended Items}}{\sum \text{Recommendation Carousel Impressions}}$$
  * *Minimum Detectable Effect (MDE)*: +5.0% relative increase at $\alpha = 0.05, 1 - \beta = 0.80$.

### Secondary & Business Metrics:
* **Add-to-Cart Conversion Rate (CR)**: Total `addtocart` events originating from recommendations.
* **Catalog Coverage / Long-Tail Exploration**: Percentage of unique items in catalog shown at least once across all users.

### Guardrail Metrics (Must Not Degrade):
* **P99 API Latency**: Must remain under 100ms.
* **Error Rate (5xx HTTP Status Codes)**: Must remain under 0.05%.
* **Unsubscribe / Bounce Rate**: Must not increase by $>0.5\%$.

---

## 4. Rollout & Risk Mitigation Strategy
1. **Day 1**: 5% Treatment Canary deployment to verify real-time inference latency and infrastructure health.
2. **Day 2–7**: Ramp up to 50/50 split if P99 latency remains $<50\text{ ms}$.
3. **Day 14**: Run two-tailed Student's t-test / Chi-square test on CTR and Conversion Rate.
4. **Rollback Plan**: Automatic circuit-breaker triggers routing all requests to `serving/cold_start.py` (popularity fallback) if 5xx errors exceed 0.5% for two consecutive minutes.
