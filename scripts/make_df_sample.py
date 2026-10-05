"""
make_df_sample.py — Build a seeded, balanced sample protocol from the DF 2021
files actually present on disk (part00 only until the other parts are downloaded).

Usage:
    python scripts/make_df_sample.py --n 20000 --out data/raw/ASVspoof2021_DF/df_sample_20k.txt
Output keeps trial_metadata.txt line format so eval.py reads it unchanged.
"""
import argparse
import random
from pathlib import Path

ROOT = Path(__file__).parent.parent
DF = ROOT / "data/raw/ASVspoof2021_DF"

p = argparse.ArgumentParser()
p.add_argument("--n", type=int, default=20000)
p.add_argument("--seed", type=int, default=42)
p.add_argument("--meta", default=str(DF / "keys/DF/CM/trial_metadata.txt"))
p.add_argument("--audio-dir", default=str(DF / "ASVspoof2021_DF_eval/flac"))
p.add_argument("--out", default=str(DF / "df_sample_20k.txt"))
a = p.parse_args()

on_disk = {f.stem for f in Path(a.audio_dir).glob("*.flac")}
bona, spoof = [], []
for line in open(a.meta, encoding="utf-8"):
    parts = line.split()
    if len(parts) < 8 or parts[1] not in on_disk:
        continue
    (bona if parts[5].lower() == "bonafide" else spoof).append(line)

print(f"on disk: {len(on_disk)}  bonafide={len(bona)}  spoof={len(spoof)}")
rng = random.Random(a.seed)
k_b = min(len(bona), a.n // 2)
k_s = min(len(spoof), a.n - k_b)
sample = rng.sample(bona, k_b) + rng.sample(spoof, k_s)
rng.shuffle(sample)
Path(a.out).write_text("".join(sample), encoding="utf-8")
print(f"wrote {len(sample)} (bonafide={k_b}, spoof={k_s}) -> {a.out}")
