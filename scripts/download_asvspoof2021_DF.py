"""
download_asvspoof2021_DF.py — Download ASVspoof 2021 DF eval partition from Zenodo
=====================================================================================
Zenodo record 4835108: https://zenodo.org/records/4835108

⚠️  IMPORTANT — Labels / Protocol file:
    The audio files (below) are on Zenodo.
    The evaluation KEYS (bonafide/spoof labels) are separate:
        https://www.asvspoof.org/index2021.html
        → Download: "DF keys" (DF2021_eval_CM_metadata.txt)
    Place it at:  data/raw/ASVspoof2021_DF/DF2021_eval_CM_metadata.txt

Usage:
    python scripts/download_asvspoof2021_DF.py

After download + key placement, run Phase 4 eval with:
    python eval.py \\
        --protocol data/raw/ASVspoof2021_DF/DF2021_eval_CM_metadata.txt \\
        --audio-dir data/raw/ASVspoof2021_DF/flac/ \\
        --run-name asvspoof21_DF_eval \\
        --weights outputs/weights/AASIST-L.pth

Expected EER: ~15–20%  (vs 2019 LA AASIST-L baseline ~0.99%)
The degradation IS the Phase 4 finding.
"""

import hashlib
import sys
import tarfile
from pathlib import Path
from urllib.request import urlretrieve

ROOT = Path(__file__).parent.parent
DEST_DIR = ROOT / "data" / "raw" / "ASVspoof2021_DF"
ZENODO_BASE = "https://zenodo.org/api/records/4835108/files"

# Exact file list from Zenodo API (record 4835108, verified 2026-09-25)
FILES = [
    # (key, size_bytes, md5)
    ("ASVspoof2021_DF_eval_part00.tar.gz", 8_637_050_238, "4f2cbae07cf3ede2a1dde3b8d2ee55ea"),
    ("ASVspoof2021_DF_eval_part01.tar.gz", 8_624_944_890, "1578c89ab433c2b60b1dce93bdf8fbec"),
    ("ASVspoof2021_DF_eval_part02.tar.gz", 8_616_337_195, "5497a35f0126e94a1d7a7d26db57b4f7"),
    ("ASVspoof2021_DF_eval_part03.tar.gz", 8_656_151_261, "42b7512ba2943e98a32a53c9608cf83c"),
]
TOTAL_GB = sum(s for _, s, _ in FILES) / 1e9


def _progress_hook(downloaded_blocks, block_size, total_size):
    downloaded = downloaded_blocks * block_size
    if total_size > 0:
        pct = min(100, downloaded * 100 // total_size)
        mb_done = downloaded / (1024 * 1024)
        mb_total = total_size / (1024 * 1024)
        bar = "█" * (pct // 5) + "░" * (20 - pct // 5)
        print(f"\r  [{bar}] {pct:3d}%  {mb_done:,.0f}/{mb_total:,.0f} MB", end="", flush=True)
    else:
        mb_done = downloaded / (1024 * 1024)
        print(f"\r  {mb_done:,.0f} MB downloaded...", end="", flush=True)


def verify_md5(path: Path, expected_md5: str) -> bool:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest() == expected_md5


def download_file(key: str, size_bytes: int, expected_md5: str) -> bool:
    dest = DEST_DIR / key
    url = f"{ZENODO_BASE}/{key}/content"

    if dest.exists():
        size_ok = dest.stat().st_size == size_bytes
        if size_ok:
            print(f"  ✓ Already exists (correct size): {key} — skipping.")
            return True
        else:
            print(f"  ↺ Incomplete download found ({dest.stat().st_size / 1e6:.0f} MB / {size_bytes / 1e6:.0f} MB) — re-downloading.")
            dest.unlink()

    print(f"\n  ⬇  Downloading {key} ({size_bytes / 1e9:.2f} GB)...")
    try:
        urlretrieve(url, dest, reporthook=_progress_hook)
        print()
    except Exception as e:
        print(f"\n  ✗ FAILED to download {key}: {e}")
        if dest.exists():
            dest.unlink()
        return False

    # Verify MD5
    print(f"  Verifying MD5 for {key}...")
    if not verify_md5(dest, expected_md5):
        print(f"  ✗ MD5 mismatch! File may be corrupted — deleting.")
        dest.unlink()
        return False
    print(f"  ✓ MD5 OK.")
    return True


def extract_tar(key: str):
    path = DEST_DIR / key
    print(f"  Extracting {key}...")
    with tarfile.open(path, "r:gz") as tar:
        tar.extractall(DEST_DIR)
    print(f"  ✓ Extracted.")


def check_labels():
    """Warn if the labels file is not present."""
    label_candidates = [
        DEST_DIR / "DF2021_eval_CM_metadata.txt",
        DEST_DIR / "keys" / "DF" / "CM" / "trial_metadata.txt",
    ]
    for p in label_candidates:
        if p.exists():
            return p
    print("""
⚠️  WARNING: Evaluation labels (keys) file not found!
   ASVspoof 2021 DF labels must be downloaded separately from:
   https://www.asvspoof.org/index2021.html
   → Look for 'DF eval keys' or 'DF2021_eval_CM_metadata.txt'
   → Place at: data/raw/ASVspoof2021_DF/DF2021_eval_CM_metadata.txt

   Without this file, eval.py cannot compute EER.
""")
    return None


def main():
    DEST_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("ASVspoof 2021 DF Eval — Downloader")
    print(f"  Zenodo record : https://zenodo.org/records/4835108")
    print(f"  Destination   : {DEST_DIR}")
    print(f"  Files         : {len(FILES)} parts")
    print(f"  Total size    : {TOTAL_GB:.1f} GB")
    print(f"  Free disk req : ~{TOTAL_GB * 2.5:.0f} GB (compressed + extracted)")
    print("=" * 65)

    # Check disk space
    import shutil
    free_bytes = shutil.disk_usage(DEST_DIR).free
    free_gb = free_bytes / 1e9
    needed_gb = TOTAL_GB * 2.5
    if free_gb < needed_gb:
        print(f"✗ Insufficient disk space! {free_gb:.1f} GB free, ~{needed_gb:.0f} GB needed.")
        sys.exit(1)
    print(f"  Disk space    : {free_gb:.1f} GB free ✓\n")

    failed = []
    for key, size_bytes, md5 in FILES:
        ok = download_file(key, size_bytes, md5)
        if not ok:
            failed.append(key)
            continue
        extract_tar(key)
        # Optionally delete the tar after extraction to save space
        # (DEST_DIR / key).unlink()

    print("\n" + "=" * 65)
    if failed:
        print(f"⚠  {len(failed)} file(s) failed:")
        for f in failed:
            print(f"   - {f}")
        print("Re-run this script to retry.")
        sys.exit(1)

    print("✅ All 4 parts downloaded and extracted.")
    labels_path = check_labels()
    print()
    print("━" * 65)
    print("NEXT STEPS")
    print("━" * 65)
    if not labels_path:
        print("1. Download DF eval keys from https://www.asvspoof.org/index2021.html")
        print("   Place at: data/raw/ASVspoof2021_DF/DF2021_eval_CM_metadata.txt")
        print()
    print("2. Run Phase 4 cross-domain evaluation:")
    print()
    print("   python eval.py \\")
    print("     --protocol data/raw/ASVspoof2021_DF/DF2021_eval_CM_metadata.txt \\")
    print("     --audio-dir data/raw/ASVspoof2021_DF/flac/ \\")
    print("     --run-name asvspoof21_DF_eval \\")
    print("     --weights outputs/weights/AASIST-L.pth")
    print()
    print("Expected EER: ~15–20%  (vs 2019 LA AASIST-L baseline ~0.99%)")
    print("The degradation is due to unseen Opus/MP3 codec artifacts.")
    print("This IS the finding — motivates Phase 5 calibration.")
    print("=" * 65)


if __name__ == "__main__":
    main()
