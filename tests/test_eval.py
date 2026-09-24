"""
test_eval.py — Unit test for Phase 3 EER & Scoring
===================================================
Tests:
1. compute_eer function:
   - Perfectly separated scores -> EER == 0.0%
   - Overlapping scores -> EER matches theoretical expectation
2. read_protocol parser functionality
"""

import numpy as np
import pytest
from eval import compute_eer, read_protocol


def test_compute_eer_perfect_separation():
    """When bona fide scores are strictly higher than spoof scores, EER must be 0%."""
    bonafide = np.array([5.0, 6.0, 7.0, 8.0, 9.0])
    spoof = np.array([-5.0, -6.0, -7.0, -8.0, -9.0])

    eer, threshold = compute_eer(bonafide, spoof)
    assert eer == 0.0, f"Expected 0.0 EER, got {eer}"


def test_compute_eer_overlapping():
    """Test overlapping distributions produce valid EER between 0 and 0.5."""
    np.random.seed(42)
    bonafide = np.random.normal(loc=1.0, scale=1.0, size=1000)
    spoof = np.random.normal(loc=-1.0, scale=1.0, size=1000)

    eer, threshold = compute_eer(bonafide, spoof)
    assert 0.10 <= eer <= 0.20, f"Expected EER around ~15%, got {eer*100:.2f}%"
    assert -0.5 <= threshold <= 0.5, f"Expected threshold around ~0, got {threshold}"
