"""
eval.py — Batch evaluation: EER & scoring (Phase 3 & 4)
========================================================
Phase 3 (ASVspoof 2019 LA):
    python eval.py --protocol data/raw/ASVspoof2019_LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt --audio-dir data/raw/ASVspoof2019_LA/ASVspoof2019_LA_eval/flac/ --run-name asvspoof19_LA_eval

Quick test on 500 files:
    python eval.py --protocol data/raw/ASVspoof2019_LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt --audio-dir data/raw/ASVspoof2019_LA/ASVspoof2019_LA_dev/flac/ --run-name quick_dev_test --limit 500

Outputs:
    outputs/scores/<run_name>_scores.csv   — per-file (filename, label, score, prob_spoof)
    outputs/scores/<run_name>_summary.txt  — EER summary and metrics
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_curve

ROOT = Path(__file__).parent
DEFAULT_WEIGHTS = ROOT / "outputs/weights/best_model.pth"
FALLBACK_WEIGHTS = ROOT / "outputs/weights/AASIST-L.pth"


def get_args():
    p = argparse.ArgumentParser(description="Batch evaluation — EER computation")
    p.add_argument("--protocol",  required=True, help="ASVspoof protocol .txt file")
    p.add_argument("--audio-dir", required=True, help="Directory containing audio files (.flac or .wav)")
    p.add_argument(
        "--weights",
        default=str(DEFAULT_WEIGHTS if DEFAULT_WEIGHTS.exists() else FALLBACK_WEIGHTS),
        help="Path to model checkpoint (.pth)",
    )
    p.add_argument("--config", default="configs/aasist_L.json", help="Path to model config JSON")
    p.add_argument("--run-name", default="eval_run", help="Prefix for output score CSV and summary")
    p.add_argument("--batch-size", type=int, default=32, help="Inference batch size")
    p.add_argument("--limit", type=int, default=None, help="Optional maximum number of files to score (for fast test)")
    p.add_argument("--device", default=None, help="Device to use ('cpu' or 'cuda')")
    return p.parse_args()


def read_protocol(path: str, limit: int = None) -> list[dict]:
    """
    Parse ASVspoof protocol file — auto-detects format:

    Format A — ASVspoof 2019 LA (space-separated, no header, 5 columns):
        <speaker> <filename> <env> <attack> <label>
        e.g.: LA_0069 LA_E_5849185 - - bonafide

    Format B — ASVspoof 2021 DF (tab-separated, has header row):
        speaker_id  filename  codec  source  label  trim
        e.g.: LA_0069\\tLA_E_5849185\\t-\\t-\\tbonafide\\ttrue

    Label convention: 0 = bonafide, 1 = spoof
    If limit is specified, samples equally from bonafide and spoof.
    """
    bonafide_items = []
    spoof_items = []
    target_each = (limit // 2) if limit else None

    with open(path, "r", encoding="utf-8") as f:
        raw_lines = f.readlines()

    # ── Detect format ────────────────────────────────────────────────────────
    first_non_empty = next((l.strip() for l in raw_lines if l.strip()), "")
    has_tab_header = "\t" in first_non_empty and first_non_empty.lower().startswith("speaker")
    is_2021_format = has_tab_header

    # Format C — official 2021 DF keys (trial_metadata.txt), space-separated, no header:
    #   <spk> <DF_E_xxxxxxx> <codec> <source> <attack> <label> <trim> <subset> ...
    _fp = first_non_empty.split()
    is_2021_meta = (not has_tab_header and len(_fp) >= 8
                    and _fp[1].startswith("DF_") and _fp[5].lower() in ("bonafide", "spoof"))

    if is_2021_format:
        # Format B: tab-separated with header
        header = first_non_empty.split("\t")
        try:
            col_filename = header.index("filename")
            col_label    = header.index("label")
            col_attack   = header.index("source") if "source" in header else -1
        except ValueError:
            raise ValueError(f"Cannot parse 2021 DF header: {header}")
        lines_to_parse = raw_lines[1:]  # skip header
    elif is_2021_meta:
        col_filename, col_label, col_attack = 1, 5, 4
        lines_to_parse = raw_lines
    else:
        # Format A: 2019 LA space-separated, no header
        col_filename, col_label, col_attack = 1, 4, 3
        lines_to_parse = raw_lines

    for line in lines_to_parse:
        sep = "\t" if is_2021_format else None
        parts = line.strip().split(sep)
        if len(parts) <= max(col_filename, col_label):
            continue

        label_str = parts[col_label].strip().lower()
        if label_str not in ("bonafide", "spoof"):
            continue

        attack = parts[col_attack].strip() if col_attack >= 0 and col_attack < len(parts) else "-"
        entry = {
            "filename": parts[col_filename].strip(),
            "attack":   attack,
            "label":    0 if label_str == "bonafide" else 1,
        }

        if entry["label"] == 0:
            if target_each is None or len(bonafide_items) < target_each:
                bonafide_items.append(entry)
        else:
            if target_each is None or len(spoof_items) < target_each:
                spoof_items.append(entry)

        if target_each and len(bonafide_items) >= target_each and len(spoof_items) >= target_each:
            break

    items = bonafide_items + spoof_items
    fmt_name = ("ASVspoof 2021 DF (tab/header)" if is_2021_format
                else "ASVspoof 2021 DF trial_metadata" if is_2021_meta
                else "ASVspoof 2019 LA (space/no-header)")
    print(f"[eval] Protocol format detected: {fmt_name}")
    return items


def compute_eer(bonafide_scores: np.ndarray, spoof_scores: np.ndarray) -> tuple[float, float]:
    """
    Calculate Equal Error Rate (EER) where FAR == FRR.
    Score is bonafide LLR: higher score = more likely bonafide.
    Returns:
        eer (fraction, e.g. 0.0083 = 0.83%)
        threshold (decision threshold at EER)
    """
    if len(bonafide_scores) == 0 or len(spoof_scores) == 0:
        return 0.0, 0.0

    y_true = np.concatenate([np.zeros(len(bonafide_scores)), np.ones(len(spoof_scores))])
    y_scores = np.concatenate([bonafide_scores, spoof_scores])

    # pos_label=0 (bonafide):
    # fpr = false accept of spoof as bonafide (FAR)
    # tpr = true accept of bonafide (1 - FRR)
    # fnr = FRR
    fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=0)
    fnr = 1.0 - tpr

    idx = np.nanargmin(np.abs(fpr - fnr))
    eer = float((fpr[idx] + fnr[idx]) / 2.0)
    threshold = float(thresholds[idx])
    return eer, threshold


def main():
    args = get_args()
    from infer import load_model
    from src.preprocessing.preprocess import preprocess

    device_str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)

    weights_path = Path(args.weights)
    config_path = Path(args.config)

    if not weights_path.exists():
        print(f"ERROR: Model weights not found at {weights_path}", file=sys.stderr)
        sys.exit(1)

    print(f"[eval] Loading model from {weights_path} onto {device}...")
    model = load_model(weights_path, config_path, device)

    items = read_protocol(args.protocol, limit=args.limit)
    audio_dir = Path(args.audio_dir)

    print(f"[eval] Found {len(items)} items in protocol {args.protocol}")
    if args.limit:
        print(f"[eval] Limited to first {args.limit} utterances.")

    scores_out = ROOT / "outputs/scores" / f"{args.run_name}_scores.csv"
    summary_out = ROOT / "outputs/scores" / f"{args.run_name}_summary.txt"
    scores_out.parent.mkdir(parents=True, exist_ok=True)

    all_scores, all_labels = [], []
    bonafide_scores, spoof_scores = [], []
    start_time = time.time()

    with open(scores_out, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["filename", "label", "score", "prob_spoof"])

        # Process in mini-batches for performance
        batch_size = args.batch_size
        for batch_start in range(0, len(items), batch_size):
            batch_items = items[batch_start:batch_start + batch_size]
            waveforms = []
            valid_items = []

            for item in batch_items:
                audio_path = audio_dir / f"{item['filename']}.flac"
                if not audio_path.exists():
                    audio_path = audio_dir / f"{item['filename']}.wav"
                if not audio_path.exists():
                    continue

                w = preprocess(str(audio_path), mode="infer")  # [1, 64600]
                waveforms.append(w.squeeze(0))                 # [64600]
                valid_items.append(item)

            if not waveforms:
                continue

            batch_tensor = torch.stack(waveforms).to(device)  # [B, 64600]

            with torch.no_grad():
                _, logits = model(batch_tensor)  # (last_hidden, [B, 2])

            # AASIST convention: idx 0 = spoof, idx 1 = bonafide
            # Bonafide LLR: output[1] - output[0]
            scores = (logits[:, 1] - logits[:, 0]).cpu().tolist()
            probs_spoof = F.softmax(logits, dim=1)[:, 0].cpu().tolist()

            for item, score, prob_spoof in zip(valid_items, scores, probs_spoof):
                all_scores.append(score)
                all_labels.append(item["label"])

                if item["label"] == 0:
                    bonafide_scores.append(score)
                else:
                    spoof_scores.append(score)

                writer.writerow([item["filename"], item["label"], round(score, 6), round(prob_spoof, 6)])

            csvfile.flush()
            processed = min(batch_start + batch_size, len(items))
            if processed % 500 == 0 or processed == len(items):
                elapsed = time.time() - start_time
                fps = processed / elapsed if elapsed > 0 else 0
                print(f"  Processed {processed}/{len(items)} utterances ({fps:.1f} audio/s)...", flush=True)

    elapsed_total = time.time() - start_time
    eer, eer_threshold = compute_eer(np.array(bonafide_scores), np.array(spoof_scores))

    print("\n" + "=" * 60)
    print(f"Evaluation Run:       {args.run_name}")
    print(f"Total Utterances:     {len(all_scores)} (Bona fide: {len(bonafide_scores)}, Spoof: {len(spoof_scores)})")
    print(f"Baseline EER:         {eer * 100:.4f}%")
    print(f"EER Threshold:        {eer_threshold:.4f}")
    print(f"Time Taken:           {elapsed_total:.1f}s ({len(all_scores)/max(elapsed_total, 1):.1f} files/s)")
    print(f"Scores CSV:           {scores_out}")
    print(f"Summary File:         {summary_out}")
    print("=" * 60)

    with open(summary_out, "w", encoding="utf-8") as f:
        f.write(f"run_name: {args.run_name}\n")
        f.write(f"weights: {weights_path}\n")
        f.write(f"protocol: {args.protocol}\n")
        f.write(f"total_evaluated: {len(all_scores)}\n")
        f.write(f"bonafide_count: {len(bonafide_scores)}\n")
        f.write(f"spoof_count: {len(spoof_scores)}\n")
        f.write(f"EER_percent: {eer * 100:.4f}%\n")
        f.write(f"EER_threshold: {eer_threshold:.6f}\n")
        f.write(f"elapsed_seconds: {elapsed_total:.2f}\n")


if __name__ == "__main__":
    main()
