"""
domain_calibration.py — Can calibration make the abstain band absorb domain shift?
===================================================================================
Uses existing score CSVs only (no model runs).

Splits the 2021 DF sample by SPEAKER into DF-cal / DF-test (no speaker in both,
so DF-test measures generalisation to unseen speakers, the failure mode found in
analyze_abstain.py). Compares three calibrators, all reported on the same
held-out sets (2019 LA eval, DF-test):

  A  baseline   : Platt on LA dev, legacy grid thresholds (0.25 / 0.75)
  B  selective  : Platt on LA dev, least-abstain band with dev acc >= TARGET
  C  domain-mix : Platt on LA dev + DF-cal, least-abstain band with acc >= TARGET
                  on that pooled set

Usage:
    python scripts/domain_calibration.py [--target 0.99] [--save-mixed]
      --save-mixed writes calibrator C to outputs/weights/calibrator_mixed.pkl
                   (the server keeps using calibrator.pkl unless you swap it)
"""

import argparse
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.calibration.calibrate import Calibrator  # noqa: E402

DEV_CSV = ROOT / "outputs/scores/dev_for_calibration_scores.csv"
LA_CSV  = ROOT / "outputs/scores/asvspoof19_LA_eval_scores.csv"
DF_CSV  = ROOT / "outputs/scores/asvspoof21_DF_sample20k_scores.csv"
DF_META = ROOT / "data/raw/ASVspoof2021_DF/df_sample_20k.txt"
OUT     = ROOT / "outputs/scores/phase5_domain_calibration.txt"


def split_df():
    df = pd.read_csv(DF_CSV)
    meta = pd.read_csv(DF_META, sep=r"\s+", header=None, usecols=[0, 1, 3],
                       names=["speaker", "filename", "source"])
    df = df.merge(meta, on="filename")
    # deterministic speaker-level split (crc32, not hash(): stable across runs)
    in_cal = df["speaker"].map(lambda s: zlib.crc32(s.encode()) % 2 == 0)
    return df[in_cal].reset_index(drop=True), df[~in_cal].reset_index(drop=True)


def evaluate(cal, df):
    p = cal._probs(df["score"].tolist())
    y = df["label"].values
    auth, spoof = p < cal.lower, p > cal.upper
    unc = ~(auth | spoof)
    wrong_bin = (p > 0.5).astype(int) != y
    covered = ~unc
    correct = (auth & (y == 0)) | (spoof & (y == 1))
    return {
        "uncert%":   unc.mean() * 100,
        "acc_excl%": correct[covered].mean() * 100 if covered.any() else float("nan"),
        "caught%":   (wrong_bin & unc).sum() / max(wrong_bin.sum(), 1) * 100,
        "conf_wrong%": (covered & ~correct).mean() * 100,
        "bon->spoof%": (spoof & (y == 0)).sum() / max((y == 0).sum(), 1) * 100,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=float, default=0.99)
    ap.add_argument("--save-mixed", action="store_true")
    args = ap.parse_args()

    dev = pd.read_csv(DEV_CSV)
    la = pd.read_csv(LA_CSV)
    df_cal, df_test = split_df()

    out = [f"DF speaker split: cal {df_cal.speaker.nunique()} spk / {len(df_cal)} files, "
           f"test {df_test.speaker.nunique()} spk / {len(df_test)} files "
           f"(overlap {len(set(df_cal.speaker) & set(df_test.speaker))})",
           f"Target accuracy for selective tuning: {args.target:.3f}"]

    A = Calibrator().fit_arrays(dev.score.tolist(), dev.label.tolist())
    A.lower, A.upper = 0.25, 0.75

    B = Calibrator().fit_arrays(dev.score.tolist(), dev.label.tolist())
    B.tune_selective(dev.score, dev.label, target_acc=args.target)

    pool_s = dev.score.tolist() + df_cal.score.tolist()
    pool_y = dev.label.tolist() + df_cal.label.tolist()
    C = Calibrator().fit_arrays(pool_s, pool_y)
    C.tune_selective(pool_s, pool_y, target_acc=args.target)

    cols = ["uncert%", "acc_excl%", "caught%", "conf_wrong%", "bon->spoof%"]
    for name, cal in (("A baseline", A), ("B selective", B), ("C domain-mix", C)):
        coef, icpt = cal._clf.coef_[0][0], cal._clf.intercept_[0]
        lg = lambda q: np.log(q / (1 - q))
        s_hi, s_lo = (lg(cal.lower) - icpt) / coef, (lg(cal.upper) - icpt) / coef
        out.append(f"\n{name}: P band [{cal.lower:.3f}, {cal.upper:.3f}]  "
                   f"= raw score [{s_lo:+.2f}, {s_hi:+.2f}]  (coef {coef:.3f}, icpt {icpt:.3f})")
        out.append(f"  {'set':22s}" + "".join(f"{c:>13s}" for c in cols))
        for set_name, d in (("2019 LA eval", la), ("2021 DF-test (unseen spk)", df_test)):
            r = evaluate(cal, d)
            out.append(f"  {set_name:22s}" + "".join(f"{r[c]:13.2f}" for c in cols))
        if name.startswith("C"):
            for src, g in df_test.groupby("source"):
                r = evaluate(cal, g)
                out.append(f"    DF-test {src:14s}" + "".join(f"{r[c]:13.2f}" for c in cols))

    out.append("\ncaught% = share of binary (P=0.5) errors that land in UNCERTAIN; "
               "conf_wrong% = share of ALL inputs given a confident wrong verdict.")
    text = "\n".join(out)
    print(text)
    OUT.write_text(text + "\n", encoding="utf-8")
    print(f"\nSaved -> {OUT}")

    if args.save_mixed:
        C.save(str(ROOT / "outputs/weights/calibrator_mixed.pkl"))


if __name__ == "__main__":
    main()
