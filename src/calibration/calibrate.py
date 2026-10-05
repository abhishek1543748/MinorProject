"""
calibrate.py — Platt Calibration + 3-Band Abstain Verdict (Phase 5)
=====================================================================
Research contribution: no commercial deepfake detector publishes this.

The 3-band system:
  P(spoof) < lower_threshold  → AUTHENTIC (confident)
  P(spoof) > upper_threshold  → SPOOF     (confident)
  otherwise                   → UNCERTAIN (abstain)

Key finding: when the model encounters out-of-distribution audio
(e.g. ASVspoof 2021 DF with lossy codecs), raw logit scores become
small/ambiguous and Platt calibration maps them to P~0.5 → UNCERTAIN.
This is the safe behaviour: flag ambiguity instead of confident error.

Usage (standalone CLI):
    # Step 1 — generate dev scores (2019 LA dev set):
    python eval.py \\
        --protocol data/raw/ASVspoof2019_LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt \\
        --audio-dir data/raw/ASVspoof2019_LA/ASVspoof2019_LA_dev/flac/ \\
        --run-name dev_for_calibration \\
        --weights outputs/weights/AASIST-L.pth

    # Step 2 — fit calibrator and report:
    python src/calibration/calibrate.py fit \\
        --scores outputs/scores/dev_for_calibration_scores.csv \\
        --save   outputs/weights/calibrator.pkl

    # Step 3 — report abstain rate on any scored dataset:
    python src/calibration/calibrate.py report \\
        --scores   outputs/scores/asvspoof19_LA_eval_scores.csv \\
        --calibrator outputs/weights/calibrator.pkl \\
        --name     "2019 LA eval (in-domain)"

    python src/calibration/calibrate.py report \\
        --scores   outputs/scores/asvspoof21_DF_eval_scores.csv \\
        --calibrator outputs/weights/calibrator.pkl \\
        --name     "2021 DF eval (cross-domain)"
"""

import csv
import pickle
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# ── Default thresholds (tune with `tune` subcommand) ──────────────────────────
LOWER_DEFAULT = 0.35   # below → AUTHENTIC
UPPER_DEFAULT = 0.65   # above → SPOOF
                       # between → UNCERTAIN / ABSTAIN


# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class Verdict:
    """Single-utterance calibrated verdict."""
    band: str          # "authentic" | "uncertain" | "spoof"
    prob_spoof: float  # calibrated posterior P(spoof) in [0, 1]
    confidence: str    # "high" | "low"
    raw_score: float   # bonafide LLR from AASIST (higher = more likely bonafide)


@dataclass
class BandReport:
    """Per-dataset abstain-rate report."""
    name: str
    total: int
    authentic_n: int
    uncertain_n: int
    spoof_n: int
    correct_outside_abstain: int
    thresholds: tuple = field(default=(LOWER_DEFAULT, UPPER_DEFAULT))

    @property
    def authentic_pct(self): return self.authentic_n / self.total * 100
    @property
    def uncertain_pct(self): return self.uncertain_n / self.total * 100
    @property
    def spoof_pct(self):     return self.spoof_n     / self.total * 100
    @property
    def non_abstain(self):   return self.total - self.uncertain_n
    @property
    def accuracy_excl_abstain(self):
        if self.non_abstain == 0:
            return float("nan")
        return self.correct_outside_abstain / self.non_abstain * 100

    def __str__(self):
        lines = [
            f"  Dataset   : {self.name}",
            f"  Total     : {self.total}",
            f"  Thresholds: lower={self.thresholds[0]:.2f}  upper={self.thresholds[1]:.2f}",
            f"  +-------------------------------------+",
            f"  |  AUTHENTIC : {self.authentic_n:6d}  ({self.authentic_pct:5.1f}%)  |",
            f"  |  UNCERTAIN : {self.uncertain_n:6d}  ({self.uncertain_pct:5.1f}%)  |",
            f"  |  SPOOF     : {self.spoof_n:6d}  ({self.spoof_pct:5.1f}%)  |",
            f"  +-------------------------------------+",
            f"  Accuracy (excl. abstain): {self.accuracy_excl_abstain:.2f}%",
        ]
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
class Calibrator:
    """
    Platt scaling calibrator for AASIST-L bonafide log-likelihood scores.

    Fits a logistic regression on (raw_score, label) pairs from the
    development set.  The resulting P(spoof) is used to assign a 3-band
    verdict: AUTHENTIC / UNCERTAIN / SPOOF.

    Notes
    -----
    - raw_score = output[1] - output[0]  (bonafide LLR, higher = more bonafide)
    - sklearn LogisticRegression with C=1e4 approximates Platt scaling
      (minimal regularisation so the sigmoid is data-driven, not shrunk)
    - label convention matches eval.py:  0 = bonafide, 1 = spoof
    """

    def __init__(self,
                 lower: float = LOWER_DEFAULT,
                 upper: float = UPPER_DEFAULT):
        self.lower = lower
        self.upper = upper
        self._clf = None
        self._fit_n = 0

    # ── Fitting ───────────────────────────────────────────────────────────────
    def fit(self, scores_csv: str) -> "Calibrator":
        """
        Fit Platt scaler from a scores CSV produced by eval.py.
        CSV columns: filename, label, score, prob_spoof
        """
        from sklearn.linear_model import LogisticRegression

        scores, labels = _load_scores_csv(scores_csv)
        if len(scores) == 0:
            raise ValueError(f"No scores loaded from {scores_csv}")

        X = np.array(scores).reshape(-1, 1)
        y = np.array(labels)

        # C=1e4 → minimal regularisation (Platt scaling convention)
        self._clf = LogisticRegression(C=1e4, max_iter=1000, solver="lbfgs")
        self._clf.fit(X, y)
        self._fit_n = len(scores)

        n_bon = (y == 0).sum()
        n_spo = (y == 1).sum()
        print(f"[calibrate] Fitted on {len(scores)} samples "
              f"({n_bon} bonafide, {n_spo} spoof).")
        print(f"[calibrate] LR coef={self._clf.coef_[0][0]:.4f}  "
              f"intercept={self._clf.intercept_[0]:.4f}")
        return self

    def tune_thresholds(self, scores_csv: str,
                        lower_range=(0.25, 0.45),
                        upper_range=(0.55, 0.75),
                        n_steps: int = 9) -> tuple[float, float]:
        """
        Grid-search for thresholds that maximise accuracy-outside-abstain
        while keeping abstain rate ≤ 30%.
        Returns (best_lower, best_upper) and updates self.lower / self.upper.
        """
        scores, labels = _load_scores_csv(scores_csv)
        probs = self._probs(scores)

        lowers = np.linspace(*lower_range, n_steps)
        uppers = np.linspace(*upper_range, n_steps)

        best = {"lower": self.lower, "upper": self.upper,
                "acc": 0.0, "abstain_pct": 100.0}

        for lo in lowers:
            for hi in uppers:
                if lo >= hi:
                    continue
                report = _compute_report("tune", probs, labels, lo, hi)
                if report.uncertain_pct > 30.0:
                    continue
                if report.accuracy_excl_abstain > best["acc"]:
                    best = {"lower": lo, "upper": hi,
                            "acc": report.accuracy_excl_abstain,
                            "abstain_pct": report.uncertain_pct}

        self.lower = round(float(best["lower"]), 4)
        self.upper = round(float(best["upper"]), 4)
        print(f"[calibrate] Tuned thresholds -> lower={self.lower}  upper={self.upper}")
        print(f"            Accuracy outside abstain: {best['acc']:.2f}%  "
              f"Abstain rate: {best['abstain_pct']:.1f}%")
        return self.lower, self.upper

    # ── Inference ─────────────────────────────────────────────────────────────
    def predict(self, raw_score: float) -> Verdict:
        """Map a single raw AASIST score to a 3-band Verdict."""
        self._require_fitted()
        prob = float(self._clf.predict_proba([[raw_score]])[0, 1])

        if prob < self.lower:
            band, confidence = "authentic", "high"
        elif prob > self.upper:
            band, confidence = "spoof", "high"
        else:
            band, confidence = "uncertain", "low"

        return Verdict(
            band=band,
            prob_spoof=round(prob, 4),
            confidence=confidence,
            raw_score=round(raw_score, 4),
        )

    def predict_batch(self, scores: list[float]) -> list[Verdict]:
        """Vectorized batch prediction."""
        self._require_fitted()
        X = np.array(scores).reshape(-1, 1)
        probs = self._clf.predict_proba(X)[:, 1]
        return [
            self.predict.__wrapped__(self, s, p)
            if hasattr(self.predict, "__wrapped__") else
            Verdict(
                band=("authentic" if p < self.lower else
                      "spoof"     if p > self.upper else "uncertain"),
                prob_spoof=round(float(p), 4),
                confidence="high" if p < self.lower or p > self.upper else "low",
                raw_score=round(float(s), 4),
            )
            for s, p in zip(scores, probs)
        ]

    # ── Reporting ─────────────────────────────────────────────────────────────
    def report(self, scores_csv: str, name: str = "") -> BandReport:
        """
        Compute per-band distribution on a scored CSV.
        This IS the research finding.
        """
        self._require_fitted()
        scores, labels = _load_scores_csv(scores_csv)
        probs = self._probs(scores)
        report = _compute_report(name or Path(scores_csv).stem,
                                 probs, labels, self.lower, self.upper)
        print(str(report))
        return report

    # ── Persistence ───────────────────────────────────────────────────────────
    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"[calibrate] Calibrator saved → {path}")

    @classmethod
    def load(cls, path: str) -> "Calibrator":
        with open(path, "rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, cls):
            raise TypeError(f"Loaded object is {type(obj)}, expected Calibrator")
        print(f"[calibrate] Calibrator loaded from {path}  "
              f"(fitted on {obj._fit_n} samples, "
              f"lower={obj.lower}, upper={obj.upper})")
        return obj

    # ── Internal helpers ──────────────────────────────────────────────────────
    def _require_fitted(self):
        if self._clf is None:
            raise RuntimeError("Calibrator not fitted. Call .fit() or .load() first.")

    def _probs(self, scores: list[float]) -> np.ndarray:
        X = np.array(scores).reshape(-1, 1)
        return self._clf.predict_proba(X)[:, 1]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_scores_csv(path: str) -> tuple[list[float], list[int]]:
    """Load scores + labels from an eval.py output CSV."""
    scores, labels = [], []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            scores.append(float(row["score"]))
            labels.append(int(row["label"]))
    return scores, labels


def _compute_report(name: str,
                    probs: np.ndarray,
                    labels: list[int],
                    lower: float,
                    upper: float) -> BandReport:
    authentic_n = uncertain_n = spoof_n = correct = 0
    for p, label in zip(probs, labels):
        if p < lower:
            band = "authentic"; pred = 0
        elif p > upper:
            band = "spoof";     pred = 1
        else:
            band = "uncertain"; pred = -1

        if band == "authentic":   authentic_n += 1
        elif band == "uncertain": uncertain_n += 1
        else:                     spoof_n += 1

        if pred != -1 and pred == label:
            correct += 1

    return BandReport(
        name=name,
        total=len(labels),
        authentic_n=authentic_n,
        uncertain_n=uncertain_n,
        spoof_n=spoof_n,
        correct_outside_abstain=correct,
        thresholds=(lower, upper),
    )


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def _cmd_fit(args):
    cal = Calibrator()
    cal.fit(args.scores)
    if args.tune:
        cal.tune_thresholds(args.scores)
    cal.save(args.save)


def _cmd_report(args):
    cal = Calibrator.load(args.calibrator)
    report = cal.report(args.scores, name=args.name or Path(args.scores).stem)
    if args.save_report:
        out = Path(args.save_report)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            f.write(str(report) + "\n")
        print(f"[calibrate] Report saved → {out}")


def _cmd_tune(args):
    cal = Calibrator.load(args.calibrator)
    lo, hi = cal.tune_thresholds(args.scores)
    cal.lower = lo
    cal.upper = hi
    cal.save(args.calibrator)


def main():
    import argparse
    p = argparse.ArgumentParser(description="Platt calibration + 3-band abstain verdict")
    sub = p.add_subparsers(dest="cmd", required=True)

    # ── fit ──────────────────────────────────────────────────────────────────
    fit_p = sub.add_parser("fit", help="Fit calibrator on dev-set scores CSV")
    fit_p.add_argument("--scores", required=True,
                       help="Path to scores CSV from eval.py (dev set)")
    fit_p.add_argument("--save", required=True,
                       help="Where to save calibrator .pkl")
    fit_p.add_argument("--tune", action="store_true",
                       help="Grid-search thresholds after fitting")

    # ── report ───────────────────────────────────────────────────────────────
    rep_p = sub.add_parser("report", help="Report abstain-band distribution on a scored CSV")
    rep_p.add_argument("--scores",      required=True)
    rep_p.add_argument("--calibrator",  required=True,
                       help="Path to saved .pkl calibrator")
    rep_p.add_argument("--name",        default="",
                       help="Dataset name label for the report")
    rep_p.add_argument("--save-report", default=None,
                       help="Optional file to save report text")

    # ── tune ─────────────────────────────────────────────────────────────────
    tun_p = sub.add_parser("tune", help="Re-tune thresholds and update calibrator in place")
    tun_p.add_argument("--scores",     required=True)
    tun_p.add_argument("--calibrator", required=True)

    args = p.parse_args()

    if args.cmd == "fit":    _cmd_fit(args)
    elif args.cmd == "report": _cmd_report(args)
    elif args.cmd == "tune":   _cmd_tune(args)


if __name__ == "__main__":
    main()
