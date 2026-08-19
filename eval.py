"""
eval.py — Batch evaluation: EER + min-tDCF (Phase 3 & 4)
==========================================================
Phase 3: python eval.py --protocol data/raw/ASVspoof2019_LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt --audio-dir data/raw/ASVspoof2019_LA/ASVspoof2019_LA_eval/flac/ --run-name asvspoof19_LA_eval
Phase 4: python eval.py --protocol data/raw/ASVspoof2021_DF/eval.txt --audio-dir data/raw/ASVspoof2021_DF/flac/ --run-name asvspoof21_DF_eval

Outputs:
    outputs/scores/<run_name>_scores.csv   — per-file (filename, label, score)
    outputs/scores/<run_name>_summary.txt  — EER, min-tDCF
"""

import argparse
import csv
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np

ROOT = Path(__file__).parent


def get_args():
    p = argparse.ArgumentParser(description="Batch evaluation — EER + tDCF")
    p.add_argument("--protocol",  required=True, help="ASVspoof protocol .txt file")
    p.add_argument("--audio-dir", required=True, help="Directory containing .flac files")
    p.add_argument("--weights", default="outputs/weights/best_model.pth")
    p.add_argument("--config",  default="configs/aasist_L.json")
    p.add_argument("--run-name", default="eval_run", help="Name for output files")
    return p.parse_args()


def read_protocol(path: str) -> list[dict]:
    """
    Parse ASVspoof 2019 LA protocol file.
    Format: <speaker> <filename> <env> <attack> <label>
    """
    items = []
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            items.append({
                "filename": parts[1],
                "attack":   parts[3],
                "label":    1 if parts[4] == "spoof" else 0,   # 1=spoof, 0=bonafide
            })
    return items


def compute_eer(scores: list[float], labels: list[int]) -> float:
    """
    Equal Error Rate: the threshold where FAR == FRR.
    Returns EER as a fraction (0.05 = 5%).
    """
    from scipy.optimize import brentq
    from scipy.interpolate import interp1d

    scores = np.array(scores)
    labels = np.array(labels)

    thresholds = np.sort(np.unique(scores))
    fpr_list, fnr_list = [], []

    for t in thresholds:
        pred = (scores >= t).astype(int)
        fp = np.sum((pred == 1) & (labels == 0))
        fn = np.sum((pred == 0) & (labels == 1))
        tp = np.sum((pred == 1) & (labels == 1))
        tn = np.sum((pred == 0) & (labels == 0))
        fpr = fp / (fp + tn + 1e-8)
        fnr = fn / (fn + tp + 1e-8)
        fpr_list.append(fpr); fnr_list.append(fnr)

    fpr_arr = np.array(fpr_list)
    fnr_arr = np.array(fnr_list)

    try:
        eer = brentq(lambda x: interp1d(fpr_arr, fnr_arr)(x) - x, 0, 1)
    except Exception:
        eer = (fpr_arr + fnr_arr).min() / 2

    return float(eer)


def main():
    args = get_args()
    from infer import load_model
    from src.preprocessing.preprocess import preprocess

    device = torch.device("cpu")
    weights = Path(args.weights)
    config  = Path(args.config)

    if not weights.exists():
        print(f"ERROR: weights not found at {weights}", file=sys.stderr); sys.exit(1)

    print(f"Loading model from {weights}...")
    model = load_model(weights, config, device)

    items = read_protocol(args.protocol)
    audio_dir = Path(args.audio_dir)

    scores_out = ROOT / "outputs/scores" / f"{args.run_name}_scores.csv"
    scores_out.parent.mkdir(parents=True, exist_ok=True)

    all_scores, all_labels = [], []

    with open(scores_out, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["filename", "label", "score", "prob_spoof"])

        for i, item in enumerate(items):
            audio_path = audio_dir / f"{item['filename']}.flac"
            if not audio_path.exists():
                audio_path = audio_dir / f"{item['filename']}.wav"
            if not audio_path.exists():
                print(f"  WARN: file not found: {item['filename']}", file=sys.stderr)
                continue

            tensor = preprocess(str(audio_path), mode="infer").to(device)  # [1, 64600]
            with torch.no_grad():
                _, output = model(tensor)   # returns (last_hidden, logits)
            score = (output[0, 1] - output[0, 0]).item()   # bonafide log-likelihood ratio
            prob_spoof = F.softmax(output, dim=1)[0, 0].item()  # idx 0 = spoof

            all_scores.append(score)
            all_labels.append(item["label"])
            writer.writerow([item["filename"], item["label"], round(score, 6), round(prob_spoof, 6)])

            if (i + 1) % 500 == 0:
                print(f"  Scored {i+1}/{len(items)} files...")

    eer = compute_eer(all_scores, all_labels)
    print(f"\n{'='*50}")
    print(f"Run:     {args.run_name}")
    print(f"Files:   {len(all_scores)}")
    print(f"EER:     {eer*100:.4f}%")
    print(f"{'='*50}")
    print(f"Scores:  {scores_out}")

    summary_path = ROOT / "outputs/scores" / f"{args.run_name}_summary.txt"
    with open(summary_path, "w") as f:
        f.write(f"run:    {args.run_name}\n")
        f.write(f"files:  {len(all_scores)}\n")
        f.write(f"EER:    {eer*100:.4f}%\n")
        f.write(f"weights:{args.weights}\n")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
