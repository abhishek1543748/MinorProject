"""
make_df_protocols.py — DF-cal / DF-test protocols in the 5-column ASVspoof 2019 format
=====================================================================================
Splits data/raw/ASVspoof2021_DF/df_sample_20k.txt by SPEAKER with the same rule
as scripts/domain_calibration.py (is_cal_speaker), so DF-test speakers never
touch calibration or checkpoint selection.

Writes (next to the sample):
  df_cal_protocol.txt   -> train.py --select-protocol  (checkpoint selection)
  df_test_protocol.txt  -> eval.py --protocol          (held-out report)

Line format: speaker filename codec attack label   (ASVspoofDataset / eval.py)

Usage (from deepfake_detector/):
  python scripts/make_df_protocols.py
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from domain_calibration import is_cal_speaker  # noqa: E402

DF = ROOT / "data/raw/ASVspoof2021_DF"


def convert(sample: Path, out_dir: Path) -> dict:
    out = {"cal": [], "test": []}
    for line in sample.read_text().splitlines():
        p = line.split()
        if len(p) < 6 or p[5] not in ("bonafide", "spoof"):
            continue
        speaker, filename, codec, attack, label = p[0], p[1], p[2], p[4], p[5]
        out["cal" if is_cal_speaker(speaker) else "test"].append(
            f"{speaker} {filename} {codec} {attack} {label}")
    for name, lines in out.items():
        (out_dir / f"df_{name}_protocol.txt").write_text("\n".join(lines) + "\n")
    return {k: len(v) for k, v in out.items()}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", default=str(DF / "df_sample_20k.txt"))
    ap.add_argument("--out-dir", default=str(DF))
    a = ap.parse_args(argv)
    n = convert(Path(a.sample), Path(a.out_dir))
    print(f"[df-protocols] cal={n['cal']}  test={n['test']}  -> {a.out_dir}")


if __name__ == "__main__":
    main()
