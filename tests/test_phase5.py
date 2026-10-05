"""
tests/test_phase5.py — Unit tests for Phase 5 calibration module
=================================================================
Run with:
    .venv\\Scripts\\python.exe -m pytest tests/test_phase5.py -v
"""

import csv
import pickle
import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.calibration.calibrate import (
    Calibrator,
    BandReport,
    Verdict,
    _load_scores_csv,
    _compute_report,
    LOWER_DEFAULT,
    UPPER_DEFAULT,
)


# ─── Fixtures ────────────────────────────────────────────────────────────────

def make_scores_csv(tmp_path: Path, n_bonafide=200, n_spoof=200,
                    bon_mean=8.0, spo_mean=-8.0, std=3.0) -> Path:
    """
    Create a synthetic scores CSV that mimics AASIST-L output on ASVspoof.
    Bonafide LLR: high positive scores, Spoof: high negative scores.
    label=0 → bonafide, label=1 → spoof  (eval.py convention)
    """
    rng = np.random.default_rng(42)
    bonafide_scores = rng.normal(bon_mean, std, n_bonafide)
    spoof_scores    = rng.normal(spo_mean, std, n_spoof)

    csv_path = tmp_path / "test_scores.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "label", "score", "prob_spoof"])
        for i, s in enumerate(bonafide_scores):
            writer.writerow([f"bon_{i}", 0, round(float(s), 6), 0.0])
        for i, s in enumerate(spoof_scores):
            writer.writerow([f"spo_{i}", 1, round(float(s), 6), 1.0])
    return csv_path


def make_ambiguous_scores_csv(tmp_path: Path, n=200) -> Path:
    """
    Scores near 0 — the 'uncertain codec-degraded' scenario (2021 DF proxy).
    """
    rng = np.random.default_rng(99)
    scores = rng.normal(0.0, 1.5, n)       # small magnitude → ambiguous
    labels = rng.integers(0, 2, n).tolist()

    csv_path = tmp_path / "ambiguous_scores.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "label", "score", "prob_spoof"])
        for i, (s, l) in enumerate(zip(scores, labels)):
            writer.writerow([f"amb_{i}", l, round(float(s), 6), 0.5])
    return csv_path


# ─── Test 1: CSV loader ───────────────────────────────────────────────────────

def test_load_scores_csv(tmp_path):
    csv_path = make_scores_csv(tmp_path, n_bonafide=50, n_spoof=50)
    scores, labels = _load_scores_csv(str(csv_path))

    assert len(scores) == 100
    assert len(labels) == 100
    assert all(isinstance(s, float) for s in scores)
    assert set(labels) == {0, 1}


# ─── Test 2: Calibrator fits without error ────────────────────────────────────

def test_fit(tmp_path):
    csv_path = make_scores_csv(tmp_path)
    cal = Calibrator()
    cal.fit(str(csv_path))

    assert cal._clf is not None
    assert cal._fit_n == 400


# ─── Test 3: Predict returns correct band for obvious scores ─────────────────

def test_predict_obvious_scores(tmp_path):
    csv_path = make_scores_csv(tmp_path)
    cal = Calibrator().fit(str(csv_path))

    # Very high positive score → authentic (bonafide)
    verdict_bon = cal.predict(20.0)
    assert verdict_bon.band == "authentic", f"Expected authentic, got {verdict_bon.band}"
    assert verdict_bon.prob_spoof < LOWER_DEFAULT
    assert verdict_bon.confidence == "high"

    # Very high negative score → spoof
    verdict_spo = cal.predict(-20.0)
    assert verdict_spo.band == "spoof", f"Expected spoof, got {verdict_spo.band}"
    assert verdict_spo.prob_spoof > UPPER_DEFAULT
    assert verdict_spo.confidence == "high"


# ─── Test 4: Ambiguous scores fall into uncertain band ───────────────────────

def test_uncertain_band_on_ambiguous_data(tmp_path):
    """
    Scores near zero should produce P(spoof) ≈ 0.5 → UNCERTAIN band.
    This simulates what happens to AASIST on 2021 DF codec-degraded audio.
    """
    clean_csv    = make_scores_csv(tmp_path)
    ambig_csv    = make_ambiguous_scores_csv(tmp_path)

    cal = Calibrator().fit(str(clean_csv))

    report_clean = cal.report(str(clean_csv), name="in-domain (clean)")
    report_ambig = cal.report(str(ambig_csv), name="cross-domain (ambiguous proxy)")

    # KEY ASSERTION: ambiguous data must have MORE uncertain samples than clean data
    assert report_ambig.uncertain_pct > report_clean.uncertain_pct, (
        f"Expected more uncertainty on ambiguous data: "
        f"ambig={report_ambig.uncertain_pct:.1f}%  clean={report_clean.uncertain_pct:.1f}%"
    )
    print(f"\n  In-domain  abstain: {report_clean.uncertain_pct:.1f}%")
    print(f"  Ambiguous  abstain: {report_ambig.uncertain_pct:.1f}%")


# ─── Test 5: Save and reload calibrator ──────────────────────────────────────

def test_save_load_roundtrip(tmp_path):
    csv_path = make_scores_csv(tmp_path)
    cal = Calibrator().fit(str(csv_path))

    pkl_path = tmp_path / "calibrator.pkl"
    cal.save(str(pkl_path))

    cal2 = Calibrator.load(str(pkl_path))
    assert cal2._fit_n == cal._fit_n
    assert cal2.lower  == cal.lower
    assert cal2.upper  == cal.upper

    # Predictions must be identical
    for score in [-15.0, -5.0, 0.0, 5.0, 15.0]:
        v1 = cal.predict(score)
        v2 = cal2.predict(score)
        assert v1.band       == v2.band
        assert v1.prob_spoof == v2.prob_spoof


# ─── Test 6: BandReport totals are consistent ────────────────────────────────

def test_band_report_totals(tmp_path):
    csv_path = make_scores_csv(tmp_path, n_bonafide=300, n_spoof=300)
    cal = Calibrator().fit(str(csv_path))
    report = cal.report(str(csv_path))

    assert report.authentic_n + report.uncertain_n + report.spoof_n == report.total
    assert 0.0 <= report.authentic_pct <= 100.0
    assert 0.0 <= report.uncertain_pct <= 100.0
    assert 0.0 <= report.spoof_pct     <= 100.0
    assert abs(report.authentic_pct + report.uncertain_pct + report.spoof_pct - 100.0) < 0.01


# ─── Test 7: Threshold tuning stays in valid range ───────────────────────────

def test_tune_thresholds(tmp_path):
    csv_path = make_scores_csv(tmp_path)
    cal = Calibrator().fit(str(csv_path))
    lo, hi = cal.tune_thresholds(str(csv_path))

    assert 0.0 < lo < hi < 1.0, f"Invalid thresholds: lower={lo}  upper={hi}"
    assert cal.lower == lo
    assert cal.upper == hi


# ─── Test 8: Unfitted calibrator raises RuntimeError ─────────────────────────

def test_predict_unfitted_raises():
    cal = Calibrator()
    with pytest.raises(RuntimeError, match="not fitted"):
        cal.predict(0.0)
