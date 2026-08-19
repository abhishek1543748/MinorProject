"""
calibrate.py — Platt calibration + 3-band verdict (Phase 5)
=============================================================
Your research contribution. No commercial tool publishes this.

Usage:
    # Fit calibration on dev set scores
    cal = Calibrator()
    cal.fit("outputs/scores/dev_scores.csv")
    cal.save("outputs/weights/calibrator.pkl")

    # Apply to a raw score at inference time
    cal = Calibrator.load("outputs/weights/calibrator.pkl")
    verdict = cal.predict(raw_score=2.31)
    # -> {"band": "spoof", "prob": 0.87, "confidence": "high"}
"""

import pickle
import numpy as np
from pathlib import Path
from dataclasses import dataclass


@dataclass
class Verdict:
    band: str           # "authentic" | "uncertain" | "spoof"
    prob_spoof: float   # calibrated probability (0-1)
    confidence: str     # "high" | "low" (uncertain = always low)
    raw_score: float


LOWER_THRESHOLD = 0.35   # below this -> authentic
UPPER_THRESHOLD = 0.65   # above this -> spoof
                          # between   -> uncertain (abstain)


class Calibrator:
    """
    Platt scaling: fits a logistic regression on (raw_score, label) pairs
    from the development set to produce calibrated probabilities.
    """

    def __init__(self, lower: float = LOWER_THRESHOLD,
                 upper: float = UPPER_THRESHOLD):
        self.lower = lower
        self.upper = upper
        self._clf = None

    def fit(self, scores_csv: str) -> "Calibrator":
        """
        Fit on a scores CSV file with columns: filename, label, score, prob_spoof
        label: 1=spoof, 0=bonafide
        """
        from sklearn.linear_model import LogisticRegression
        import csv

        scores, labels = [], []
        with open(scores_csv) as f:
            reader = csv.DictReader(f)
            for row in reader:
                scores.append(float(row["score"]))
                labels.append(int(row["label"]))

        X = np.array(scores).reshape(-1, 1)
        y = np.array(labels)

        self._clf = LogisticRegression()
        self._clf.fit(X, y)
        print(f"Calibrator fitted on {len(scores)} samples.")
        return self

    def predict(self, raw_score: float) -> Verdict:
        """Map a raw score to a calibrated 3-band verdict."""
        if self._clf is None:
            raise RuntimeError("Calibrator not fitted. Call .fit() or .load() first.")

        prob = float(self._clf.predict_proba([[raw_score]])[0, 1])

        if prob < self.lower:
            band = "authentic"; confidence = "high"
        elif prob > self.upper:
            band = "spoof"; confidence = "high"
        else:
            band = "uncertain"; confidence = "low"

        return Verdict(
            band=band,
            prob_spoof=round(prob, 4),
            confidence=confidence,
            raw_score=round(raw_score, 4),
        )

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"Calibrator saved to {path}")

    @classmethod
    def load(cls, path: str) -> "Calibrator":
        with open(path, "rb") as f:
            obj = pickle.load(f)
        return obj

    def report_abstain_rate(self, scores_csv: str) -> dict:
        """
        Compute per-band distribution on a scored dataset.
        This is the research finding: compare in-domain vs cross-domain.
        """
        import csv
        bands = {"authentic": 0, "uncertain": 0, "spoof": 0}
        correct, total = 0, 0

        with open(scores_csv) as f:
            reader = csv.DictReader(f)
            for row in reader:
                score = float(row["score"])
                true_label = int(row["label"])
                verdict = self.predict(score)
                bands[verdict.band] += 1
                total += 1
                if verdict.band != "uncertain":
                    pred = 1 if verdict.band == "spoof" else 0
                    if pred == true_label:
                        correct += 1

        non_abstain = total - bands["uncertain"]
        return {
            "total": total,
            "authentic_pct": round(bands["authentic"] / total * 100, 2),
            "uncertain_pct": round(bands["uncertain"] / total * 100, 2),
            "spoof_pct":     round(bands["spoof"]     / total * 100, 2),
            "accuracy_excl_abstain": round(correct / non_abstain * 100, 2) if non_abstain else None,
        }
