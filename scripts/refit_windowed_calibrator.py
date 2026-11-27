"""
refit_windowed_calibrator.py — Platt calibration on WHOLE-CLIP (window-mean) scores
=====================================================================================
Why: calibrator_mixed.pkl was fitted on single-window scores (preprocess(),
first 4.04 s), but POST /analyze feeds it the mean bonafide LLR over speech
windows. 29% of LA dev files are longer than one window, so the score the
calibrator sees at serving time has a different distribution than what it
was fitted on. This script scores files with src/api/clip_scoring.score_file
— the SAME code path the server uses — and refits on those clip scores.

Sets (same composition as calibrator_mixed.pkl, scripts/domain_calibration.py):
  dev   : ASVspoof 2019 LA dev (all files)
  dfcal : 2021 DF sample (df_sample_20k.txt), DF-cal speakers only
          (is_cal_speaker), so DF-test speakers stay held out.

Scores are appended to a CSV as they are computed, so an interrupted run
(Kaggle/Colab session limit) resumes where it stopped.

Usage:
  # local smoke test (~2-3 min CPU): 100 files per set, writes *_smoke.pkl
  python scripts/refit_windowed_calibrator.py --max_files 100

  # full run (Kaggle/Colab; ~4-5 h on CPU for AASIST-L, use --device cuda)
  python scripts/refit_windowed_calibrator.py --device cuda
  #   -> outputs/weights/calibrator_mixed.pkl (overwritten; back it up first)

  # SSL-AASIST, quantized like the demo server (DFD_QUANTIZE=1)
  python scripts/refit_windowed_calibrator.py --model ssl --quantize
  #   -> outputs/weights/calibrator_ssl_int8.pkl

  # Kaggle paths: pass --dev-protocol/--dev-audio/--df-protocol/--df-audio
"""

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from src.api.clip_scoring import score_file  # noqa: E402
from src.calibration.calibrate import Calibrator  # noqa: E402

_cfg = json.loads((ROOT / "configs/train_config.json").read_text())["data"]
DEV_PROTOCOL = ROOT / _cfg["dev_protocol"]
DEV_AUDIO = ROOT / _cfg["dev_audio_dir"]
DF_PROTOCOL = ROOT / "data/raw/ASVspoof2021_DF/df_sample_20k.txt"
DF_AUDIO = ROOT / "data/raw/ASVspoof2021_DF/ASVspoof2021_DF_eval/flac"


def get_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--model", choices=["aasist_l", "ssl"], default="aasist_l")
    p.add_argument("--quantize", action="store_true", help="ssl only: dynamic INT8 (CPU)")
    p.add_argument("--weights", default=None, help="default: same file the server loads")
    p.add_argument("--sets", default="dev,dfcal", help="comma list of: dev, dfcal")
    p.add_argument("--max_files", type=int, default=None,
                   help="per set; balanced bonafide/spoof sample (seeded) for quick tests")
    p.add_argument("--target", type=float, default=0.98,
                   help="selective-band accuracy target (calibrator_mixed.pkl used 0.98)")
    p.add_argument("--dev-protocol", default=str(DEV_PROTOCOL))
    p.add_argument("--dev-audio", default=str(DEV_AUDIO))
    p.add_argument("--df-protocol", default=str(DF_PROTOCOL))
    p.add_argument("--df-audio", default=str(DF_AUDIO))
    p.add_argument("--scores-csv", default=None, help="per-file clip scores (resumable)")
    p.add_argument("--out", default=None, help="calibrator .pkl path")
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)
    if a.quantize and a.model != "ssl":
        p.error("--quantize only applies to --model ssl")

    tag = a.model + ("_int8" if a.quantize else "")
    smoke = "_smoke" if a.max_files else ""
    if a.scores_csv is None:
        a.scores_csv = str(ROOT / f"outputs/scores/windowed_calib_{tag}{smoke}_scores.csv")
    if a.out is None:
        name = {"aasist_l": "calibrator_mixed", "ssl": "calibrator_ssl",
                "ssl_int8": "calibrator_ssl_int8"}[tag]
        a.out = str(ROOT / f"outputs/weights/{name}{smoke}.pkl")
    return a


# ── Protocols ────────────────────────────────────────────────────────────────

def read_items(protocol: str, only_cal_speakers: bool = False) -> list[dict]:
    """
    Whitespace protocol: col 0 speaker, col 1 file id, label = first
    'bonafide'/'spoof' token (col 4 in 2019 LA, col 5 in 2021 DF metadata).
    Label convention 0 = bonafide, 1 = spoof (eval.py / Calibrator).
    """
    from domain_calibration import is_cal_speaker

    items = []
    for line in Path(protocol).read_text().splitlines():
        parts = line.split()
        lab = next((t for t in parts[2:] if t in ("bonafide", "spoof")), None)
        if len(parts) < 3 or lab is None:
            continue
        if only_cal_speakers and not is_cal_speaker(parts[0]):
            continue
        items.append({"speaker": parts[0], "filename": parts[1],
                      "label": 0 if lab == "bonafide" else 1})
    return items


def subsample(items: list[dict], n: int | None, seed: int) -> list[dict]:
    """Balanced, seeded subset (n//2 bonafide + rest spoof, as available)."""
    if n is None or n >= len(items):
        return items
    rng = random.Random(seed)
    bon = [x for x in items if x["label"] == 0]
    spo = [x for x in items if x["label"] == 1]
    rng.shuffle(bon), rng.shuffle(spo)
    nb = min(len(bon), n // 2)
    out = bon[:nb] + spo[:n - nb]
    rng.shuffle(out)
    return out


def resolve(audio_dir: Path, name: str) -> Path | None:
    for ext in (".flac", ".wav"):
        if (audio_dir / (name + ext)).exists():
            return audio_dir / (name + ext)
    return None


# ── Model ────────────────────────────────────────────────────────────────────

def load(args):
    device = torch.device(args.device)
    if args.model == "ssl":
        from src.model.ssl_aasist import load_ssl_aasist
        w = args.weights or ROOT / "outputs/weights/SSL_AASIST_DF.pth"
        return load_ssl_aasist(w, device, quantize=args.quantize), Path(w)
    from infer import load_model
    best = ROOT / "outputs/weights/best_model.pth"
    w = args.weights or (best if best.exists() else ROOT / "outputs/weights/AASIST-L.pth")
    return load_model(Path(w), ROOT / "configs/aasist_L.json", device), Path(w)


# ── Main ─────────────────────────────────────────────────────────────────────

FIELDS = ["set", "filename", "label", "score", "n_windows", "n_speech"]


def main(argv=None):
    args = get_args(argv)
    sets = [s.strip() for s in args.sets.split(",") if s.strip()]
    plan = []
    for s in sets:
        if s == "dev":
            items, audio = read_items(args.dev_protocol), Path(args.dev_audio)
        elif s == "dfcal":
            items, audio = read_items(args.df_protocol, only_cal_speakers=True), Path(args.df_audio)
        else:
            raise SystemExit(f"unknown set {s!r}")
        items = subsample(items, args.max_files, args.seed)
        plan.append((s, items, audio))
        print(f"[refit] {s}: {len(items)} files "
              f"({sum(x['label'] == 0 for x in items)} bonafide)")

    csv_path = Path(args.scores_csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if csv_path.exists():
        with open(csv_path, newline="") as f:
            done = {(r["set"], r["filename"]) for r in csv.DictReader(f)}
        print(f"[refit] resuming: {len(done)} files already in {csv_path.name}")

    todo = sum(1 for s, items, _ in plan for x in items if (s, x["filename"]) not in done)
    if todo:
        model, wpath = load(args)
        print(f"[refit] model {args.model}{' int8' if args.quantize else ''} from {wpath.name}, "
              f"{todo} files to score")
        new = not csv_path.exists()
        with open(csv_path, "a", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=FIELDS)
            if new:
                wr.writeheader()
            t0, k, missing = time.time(), 0, 0
            for s, items, audio in plan:
                for x in items:
                    if (s, x["filename"]) in done:
                        continue
                    path = resolve(audio, x["filename"])
                    if path is None:
                        missing += 1
                        continue
                    c = score_file(model, str(path))
                    wr.writerow({"set": s, "filename": x["filename"], "label": x["label"],
                                 "score": round(c.score, 6), "n_windows": len(c.window_scores),
                                 "n_speech": sum(not q for q in c.silent)})
                    k += 1
                    if k % 200 == 0:
                        f.flush()
                        rate = k / (time.time() - t0)
                        print(f"  {k}/{todo}  {rate:.2f} files/s  "
                              f"ETA {(todo - k) / rate / 60:.0f} min", flush=True)
        if missing:
            print(f"[refit] WARN: {missing} audio files not found (skipped)")

    # ── Fit on the requested sets only ───────────────────────────────────────
    wanted = {(s, x["filename"]) for s, items, _ in plan for x in items}
    with open(csv_path, newline="") as f:
        rows = [r for r in csv.DictReader(f) if (r["set"], r["filename"]) in wanted]
    scores = np.array([float(r["score"]) for r in rows])
    labels = np.array([int(r["label"]) for r in rows])
    if len(set(labels.tolist())) < 2:
        raise SystemExit("[refit] need both bonafide and spoof scores to fit")

    cal = Calibrator().fit_arrays(scores.tolist(), labels.tolist())
    cal.tune_selective(scores, labels, target_acc=args.target)
    multi = np.array([int(r["n_windows"]) > 1 for r in rows])
    print(f"[refit] {multi.mean() * 100:.1f}% of files have >1 window "
          f"(these are where window-mean != single-window score)")
    for s in sets:
        m = np.array([r["set"] == s for r in rows])
        if m.any() and len(set(labels[m].tolist())) == 2:
            p = cal._probs(scores[m].tolist())
            unc = (p >= cal.lower) & (p <= cal.upper)
            cov = ~unc
            ok = ((p < cal.lower) & (labels[m] == 0)) | ((p > cal.upper) & (labels[m] == 1))
            print(f"  {s:6s} n={m.sum():6d}  abstain {unc.mean() * 100:5.1f}%  "
                  f"acc-excl-abstain {ok[cov].mean() * 100 if cov.any() else float('nan'):6.2f}%")
    cal.save(args.out)
    return cal


if __name__ == "__main__":
    main()
