"""
analyze_abstain.py — Where does cross-domain error land? (Phase 5 analysis)
============================================================================
Answers the research question from CLAUDE.md using existing score CSVs
(no model runs):

  "how much degradation lands in the abstain zone vs a confident wrong answer"

Reports, for in-domain (2019 LA eval) and cross-domain (2021 DF sample):
  1. Band split per true class (bonafide / spoof)
  2. Error split: confident-wrong vs abstained
  3. Score shift (median score per class, EER threshold)
  4. Risk-coverage: abstain rate needed to hit a target accuracy
  5. DF breakdown by codec and by source corpus

Usage:
    python scripts/analyze_abstain.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.calibration.calibrate import Calibrator  # noqa: E402

LA_CSV  = ROOT / "outputs/scores/asvspoof19_LA_eval_scores.csv"
DF_CSV  = ROOT / "outputs/scores/asvspoof21_DF_sample20k_scores.csv"
DF_META = ROOT / "data/raw/ASVspoof2021_DF/df_sample_20k.txt"
CAL     = ROOT / "outputs/weights/calibrator.pkl"
OUT     = ROOT / "outputs/scores/phase5_abstain_analysis.txt"

TARGET_ACC = (0.99, 0.95)


def eer(scores, labels):
    """EER and threshold. scores: bonafide LLR. labels: 0=bonafide, 1=spoof."""
    bon, spo = scores[labels == 0], scores[labels == 1]
    ts = np.sort(scores)
    far = 1 - np.searchsorted(np.sort(spo), ts, side="left") / len(spo)   # spoof >= t
    frr = np.searchsorted(np.sort(bon), ts, side="left") / len(bon)       # bonafide < t
    i = np.argmin(np.abs(far - frr))
    return (far[i] + frr[i]) / 2, ts[i]


def bands(probs, lo, hi):
    return np.where(probs < lo, "authentic", np.where(probs > hi, "spoof", "uncertain"))


def risk_coverage(probs, labels):
    """Abstain on least-confident first. Returns (coverage, accuracy) arrays."""
    conf = np.abs(probs - 0.5)
    order = np.argsort(-conf)
    correct = ((probs[order] > 0.5).astype(int) == labels[order]).astype(float)
    n = np.arange(1, len(order) + 1)
    return n / len(order), np.cumsum(correct) / n


def section(df, name, lo, hi, out):
    p, y, s = df["prob"].values, df["label"].values, df["score"].values
    b = bands(p, lo, hi)
    e, t = eer(s, y)

    out.append(f"\n=== {name}  (n={len(df)}) ===")
    out.append(f"EER {e*100:.2f}%  at score threshold {t:.3f}")
    out.append(f"Median score  bonafide {np.median(s[y==0]):+.2f}   spoof {np.median(s[y==1]):+.2f}")

    out.append("Band split per true class (% of that class):")
    for cls, nm in ((0, "bonafide"), (1, "spoof")):
        m = y == cls
        pct = {k: (b[m] == k).mean() * 100 for k in ("authentic", "uncertain", "spoof")}
        out.append(f"  {nm:9s} n={m.sum():6d}  authentic {pct['authentic']:5.1f}  "
                   f"uncertain {pct['uncertain']:5.1f}  spoof {pct['spoof']:5.1f}")

    # Error = binary decision at P=0.5 is wrong. Where does it land?
    wrong = (p > 0.5).astype(int) != y
    in_abstain = wrong & (b == "uncertain")
    out.append(f"Binary errors (P=0.5): {wrong.sum()}  "
               f"-> abstained {in_abstain.sum()} ({in_abstain.sum()/max(wrong.sum(),1)*100:.1f}%), "
               f"confident-wrong {wrong.sum()-in_abstain.sum()} "
               f"({(wrong.sum()-in_abstain.sum())/max(wrong.sum(),1)*100:.1f}%)")

    cov, acc = risk_coverage(p, y)
    for ta in TARGET_ACC:
        ok = np.where(acc >= ta)[0]
        c = cov[ok[-1]] if len(ok) else 0.0
        out.append(f"To reach {ta*100:.0f}% accuracy: abstain on {(1-c)*100:5.1f}% of inputs")
    out.append(f"AURC (lower=better): {np.mean(1 - acc):.4f}")
    return b


def breakdown(df, col, b, out):
    out.append(f"\nDF by {col}:")
    out.append(f"  {'group':24s} {'n':>6s} {'EER%':>6s} {'uncert%':>8s} {'conf-wrong%':>11s}")
    p, y = df["prob"].values, df["label"].values
    wrong = ((p > 0.5).astype(int) != y) & (b != "uncertain")
    for g, idx in df.groupby(col).groups.items():
        i = df.index.get_indexer(idx)
        yy = y[i]
        e = eer(df["score"].values[i], yy)[0] * 100 if 0 < yy.sum() < len(yy) else float("nan")
        out.append(f"  {str(g):24s} {len(i):6d} {e:6.2f} "
                   f"{(b[i]=='uncertain').mean()*100:8.1f} {wrong[i].mean()*100:11.1f}")


def main():
    cal = Calibrator.load(str(CAL))
    lo, hi = cal.lower, cal.upper

    la = pd.read_csv(LA_CSV)
    df = pd.read_csv(DF_CSV)
    for d in (la, df):
        d["prob"] = cal._probs(d["score"].tolist())

    meta = pd.read_csv(DF_META, sep=r"\s+", header=None, usecols=[1, 2, 3],
                       names=["filename", "codec", "source"])
    df = df.merge(meta, on="filename", how="left").reset_index(drop=True)

    out = [f"Calibrator thresholds: lower={lo}  upper={hi}  (tuned on 2019 LA dev)"]
    section(la, "2019 LA eval (in-domain)", lo, hi, out)
    b_df = section(df, "2021 DF sample20k (cross-domain)", lo, hi, out)
    breakdown(df, "codec", b_df, out)
    breakdown(df, "source", b_df, out)

    text = "\n".join(out)
    print(text)
    OUT.write_text(text + "\n", encoding="utf-8")
    print(f"\nSaved -> {OUT}")


if __name__ == "__main__":
    main()
