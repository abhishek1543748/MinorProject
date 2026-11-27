"""
compare_models.py — AASIST-L vs SSL-AASIST on the SAME files
=============================================================
SSL-AASIST scores come from eval.py --model ssl on the 2k subsets
(data/processed/*_subset2k.txt). AASIST-L scores for the same filenames
are taken from the existing full-set CSVs, so both models are judged on
identical audio.

Usage:
    python scripts/compare_models.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
S = ROOT / "outputs/scores"
DF_META = ROOT / "data/raw/ASVspoof2021_DF/df_sample_20k.txt"
OUT = S / "model_comparison.txt"

SETS = [
    ("2019 LA eval (2k subset)", S / "ssl_la_eval_subset2k_scores.csv", S / "asvspoof19_LA_eval_scores.csv"),
    ("2021 DF sample (2k subset)", S / "ssl_df_subset2k_scores.csv", S / "asvspoof21_DF_sample20k_scores.csv"),
]


def eer(scores, labels):
    """EER and threshold. scores: bonafide LLR. labels: 0=bonafide, 1=spoof."""
    scores, labels = np.asarray(scores), np.asarray(labels)
    bon, spo = np.sort(scores[labels == 0]), np.sort(scores[labels == 1])
    ts = np.sort(scores)
    far = 1 - np.searchsorted(spo, ts, side="left") / len(spo)
    frr = np.searchsorted(bon, ts, side="left") / len(bon)
    i = np.argmin(np.abs(far - frr))
    return (far[i] + frr[i]) / 2, ts[i]


def row(name, d, col):
    e, t = eer(d[col], d["label"])
    y = d["label"].values
    bon_as_spoof = ((d[col].values < 0) & (y == 0)).sum() / max((y == 0).sum(), 1)
    spoof_as_bon = ((d[col].values >= 0) & (y == 1)).sum() / max((y == 1).sum(), 1)
    return (f"  {name:14s} EER {e*100:6.2f}%  (thr {t:+6.2f})   at score 0: "
            f"real->fake {bon_as_spoof*100:5.1f}%  fake->real {spoof_as_bon*100:5.1f}%")


def main():
    out = []
    for title, ssl_csv, base_csv in SETS:
        if not ssl_csv.exists():
            out.append(f"\n{title}: {ssl_csv.name} missing — run eval.py --model ssl first")
            continue
        ssl = pd.read_csv(ssl_csv)[["filename", "label", "score"]].rename(columns={"score": "ssl"})
        base = pd.read_csv(base_csv)[["filename", "score"]].rename(columns={"score": "aasist_l"})
        d = ssl.merge(base, on="filename", how="inner")
        out.append(f"\n=== {title}: {len(d)} files "
                   f"({(d.label == 0).sum()} bonafide, {(d.label == 1).sum()} spoof) ===")
        out.append(row("AASIST-L", d, "aasist_l"))
        out.append(row("SSL-AASIST", d, "ssl"))

        if "DF" in title and DF_META.exists():
            meta = pd.read_csv(DF_META, sep=r"\s+", header=None, usecols=[1, 2, 3],
                               names=["filename", "codec", "source"])
            d = d.merge(meta, on="filename")
            for col in ("source", "codec"):
                out.append(f"  by {col}:            {'n':>5s}  {'AASIST-L':>9s}  {'SSL':>7s}   (EER %)")
                for g, gd in d.groupby(col):
                    if gd.label.nunique() < 2:
                        continue
                    out.append(f"    {str(g):18s} {len(gd):5d}  {eer(gd.aasist_l, gd.label)[0]*100:9.2f}"
                               f"  {eer(gd.ssl, gd.label)[0]*100:7.2f}")

    text = "\n".join(out)
    print(text)
    OUT.write_text(text + "\n", encoding="utf-8")
    print(f"\nSaved -> {OUT}")


if __name__ == "__main__":
    sys.exit(main())
