# Online A/B Testing Proposal: Two-Stage Recommendation Engine

## 1. Executive Summary & Objective
Validate whether transitioning from single-stage ALS candidate retrieval (Control: Variant A) to a Two-Stage Pipeline with LightGBM LambdaMART ranking (Treatment: Variant B) increases downstream business engagement and conversion rates on Retailrocket user sessions without degrading latency budgets.

---

## 2. Hypothesis
* **Primary Hypothesis ($H_1$):** Re-ranking candidate items using real-time interaction features, historical conversion rates, and item popularity via LambdaMART will increase Session Add-to-Cart Rate (Cart-Through Rate / CTR) by at least **+8.0%** over pure ALS retrieval.
* **Secondary Hypothesis:** Overall purchase conversion rate will improve by **+5.0%** while maintaining server-side p95 response times below **20 ms**.

---

## 3. Experiment Design & Routing

* **Traffic Allocation:** 50/50 randomized split across active non-bot visitors.
* **Randomization Unit:** Hashed `visitorid` with salt (`hash(visitorid + salt) % 100`) to guarantee deterministic routing across user sessions.
* **Variants:**
  * **Variant A (Control):** Implicit ALS retrieval top-10 candidates ranked directly by ALS dot-product score.
  * **Variant B (Treatment):** Stage 1 Implicit ALS (100 candidates) $\rightarrow$ Stage 2 LightGBM LambdaMART reranking (serving top-10).
  * **Cold-Start Fallback (Both):** Top-N trending/popular items for visitors with no interaction history.

---

## 4. Evaluation Metrics

### Primary Metric (Decision Driver)
* **Add-to-Cart Conversion Rate (Cart-Through Rate):**
  $$\text{Cart Rate} = \frac{\text{Unique Sessions with } \ge 1 \text{ 'addtocart' Event}}{\text{Total Recommendation Impressions}}$$

### Secondary Business Metrics
* **Transaction Conversion Rate (CVR):** Sessions resulting in completed checkout transactions.
* **Average Order Value (AOV):** Mean transaction monetary volume per converted user.

### Guardrail Metrics (Safety Bounds)
* **p95 / p99 Latency:** FastAPI inference and ranking response must remain $\le 20 \text{ ms}$ (p95) and $\le 50 \text{ ms}$ (p99).
* **Catalog Coverage / Novelty:** Long-tail item impression ratio must not drop by more than $10\%$ (preventing popularity bias collapse).
* **Error / Fallback Rate:** $5\text{xx}$ API responses and fallback to global popularity must remain $< 0.1\%$.

---

## 5. Sample Size & Duration Calculation

* **Baseline Metric:** Historical Cart-Through Rate $\approx 4.5\%$.
* **Minimum Detectable Effect (MDE):** Relative change of $8\%$ (absolute change $\approx 0.36\%$).
* **Statistical Parameters:**
  * Significance Level ($\alpha$): $0.05$ (two-tailed)
  * Statistical Power ($1 - \beta$): $0.80$
* **Required Sample Size:** $\approx 92,000$ unique user sessions per variant ($\sim 184,000$ total sessions).
* **Test Duration:** Given Retailrocket traffic volume, run for **14 consecutive days** to account for day-of-week seasonality and novelty effects.

---

## 6. Ramp-up & Rollback Plan

1. **Phase 1 (Day 1 - Canary):** 5% Treatment / 95% Control. Validate system latency, logging telemetry, and error rates.
2. **Phase 2 (Days 2–14):** Ramp to 50% Treatment / 50% Control if p95 latency remains $< 20\text{ ms}$.
3. **Immediate Abort Conditions:**
   * Treatment p95 latency exceeds $35\text{ ms}$ continuously for $> 15$ minutes.
   * Cart conversion rate drops by $> 5\%$ with statistical significance ($p < 0.01$).
   * $5\text{xx}$ error rate exceeds $0.5\%$.
