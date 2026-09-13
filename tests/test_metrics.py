import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from src.evaluation.segment_analysis import compute_ndcg_at_k, compute_map_at_k, compute_mrr

def test_perfect_ndcg():
    actual = {101, 102}
    ranked = [101, 102, 103, 104]
    score = compute_ndcg_at_k(actual, ranked, k=2)
    assert np.isclose(score, 1.0)

def test_zero_ndcg():
    actual = {999}
    ranked = [101, 102, 103, 104]
    score = compute_ndcg_at_k(actual, ranked, k=4)
    assert score == 0.0

def test_mrr_first_position():
    actual = {101}
    ranked = [101, 102, 103]
    assert compute_mrr(actual, ranked, k=3) == 1.0

def test_mrr_second_position():
    actual = {102}
    ranked = [101, 102, 103]
    assert compute_mrr(actual, ranked, k=3) == 0.5

if __name__ == "__main__":
    test_perfect_ndcg()
    test_zero_ndcg()
    test_mrr_first_position()
    test_mrr_second_position()
    print("All ranking metric tests passed successfully!")
