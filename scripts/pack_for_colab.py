"""
pack_for_colab.py — zip ONLY the data Colab/Kaggle needs (run on your laptop)
=============================================================================
data/raw/ is gitignored, so Colab cannot get it from the repo. This writes
one zip (paths relative to data/raw/) to upload to Google Drive:

  ASVspoof2019_LA/ASVspoof2019_LA_cm_protocols/      (all protocols)
  ASVspoof2019_LA/ASVspoof2019_LA_train/flac/         ~1.6 GB  (fine-tune)
  ASVspoof2019_LA/ASVspoof2019_LA_dev/flac/           ~1.6 GB  (dev EER + calibration)
  ASVspoof2021_DF/df_sample_20k.txt + its 20k flacs   (DF-cal calibration, DF-test eval)

FLAC is already compressed, so the zip is STORED (fast, no CPU burn).

Usage (from deepfake_detector/):
  .venv\\Scripts\\python.exe scripts/pack_for_colab.py                 # everything
  .venv\\Scripts\\python.exe scripts/pack_for_colab.py --no-train      # skip LA train
  .venv\\Scripts\\python.exe scripts/pack_for_colab.py --only-df       # just the DF sample
Then upload colab_data.zip to Google Drive -> MyDrive/deepfake_detector/
"""

import argparse
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data/raw"
LA = RAW / "ASVspoof2019_LA"
DF = RAW / "ASVspoof2021_DF"
DF_SAMPLE = DF / "df_sample_20k.txt"
DF_FLAC = DF / "ASVspoof2021_DF_eval/flac"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "colab_data.zip"))
    ap.add_argument("--no-train", action="store_true", help="skip LA train audio")
    ap.add_argument("--no-df", action="store_true", help="skip the 2021 DF sample")
    ap.add_argument("--only-df", action="store_true")
    a = ap.parse_args()

    files: list[Path] = []
    if not a.only_df:
        files += sorted((LA / "ASVspoof2019_LA_cm_protocols").glob("*.txt"))
        files += sorted((LA / "ASVspoof2019_LA_dev/flac").glob("*.flac"))
        if not a.no_train:
            files += sorted((LA / "ASVspoof2019_LA_train/flac").glob("*.flac"))
    if not a.no_df and DF_SAMPLE.exists():
        files.append(DF_SAMPLE)
        names = [l.split()[1] for l in DF_SAMPLE.read_text().splitlines() if len(l.split()) > 1]
        missing = 0
        for n in names:
            p = DF_FLAC / f"{n}.flac"
            if p.exists():
                files.append(p)
            else:
                missing += 1
        if missing:
            print(f"[pack] WARN: {missing} DF sample files not found locally (skipped)")
    if not files:
        raise SystemExit("[pack] nothing to pack — is data/raw/ populated?")

    total = sum(f.stat().st_size for f in files)
    print(f"[pack] {len(files)} files, {total / 1e9:.2f} GB -> {a.out}")
    with zipfile.ZipFile(a.out, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as z:
        for i, f in enumerate(files, 1):
            z.write(f, f.relative_to(RAW).as_posix())
            if i % 5000 == 0:
                print(f"  {i}/{len(files)}", flush=True)
    print("[pack] done. Upload it to Google Drive: MyDrive/deepfake_detector/colab_data.zip")


if __name__ == "__main__":
    main()
