"""
plot_metrics.py — Visualisation for Phase 3 EER & Score Distributions
======================================================================
Usage:
    python scripts/plot_metrics.py --scores outputs/scores/phase3_dev_baseline_sample_scores.csv --run-name phase3_dev_baseline_sample

Generates:
    outputs/plots/<run_name>_score_hist.png  — Bona fide vs Spoof score distribution
    outputs/plots/<run_name>_roc_curve.png   — ROC curve with EER marked
"""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve

ROOT = Path(__file__).parent.parent


def get_args():
    p = argparse.ArgumentParser(description="Generate EER & score distribution plots")
    p.add_argument("--scores", required=True, help="Path to scores CSV file")
    p.add_argument("--run-name", default="plot", help="Prefix for saved plots")
    return p.parse_args()


def main():
    args = get_args()
    scores_path = Path(args.scores)
    plots_dir = ROOT / "outputs/plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    if not scores_path.exists():
        print(f"Error: scores file not found at {scores_path}")
        return

    bonafide_scores = []
    spoof_scores = []

    with open(scores_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            score = float(row["score"])
            label = int(row["label"])
            if label == 0:
                bonafide_scores.append(score)
            else:
                spoof_scores.append(score)

    bonafide_scores = np.array(bonafide_scores)
    spoof_scores = np.array(spoof_scores)

    print(f"Loaded {len(bonafide_scores)} bonafide and {len(spoof_scores)} spoof scores.")

    # ── 1. Score Distribution Histogram ──────────────────────────────────────
    plt.figure(figsize=(9, 5), dpi=150)
    plt.hist(spoof_scores, bins=40, alpha=0.6, color="#ef4444", label="Spoof (Fake)", density=True)
    plt.hist(bonafide_scores, bins=40, alpha=0.6, color="#10b981", label="Bona fide (Real)", density=True)
    plt.axvline(0.0, color="#6b7280", linestyle="--", linewidth=1.2, label="Zero Decision Line")

    plt.title(f"AASIST-L Score Distribution ({args.run_name})", fontsize=13, fontweight="bold")
    plt.xlabel("Bonafide Log-Likelihood Ratio (Score > 0 = Real, Score < 0 = Fake)", fontsize=11)
    plt.ylabel("Density", fontsize=11)
    plt.legend(frameon=True, facecolor="#ffffff", framealpha=0.9)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()

    hist_out = plots_dir / f"{args.run_name}_score_hist.png"
    plt.savefig(hist_out)
    plt.close()
    print(f"Saved score distribution plot to: {hist_out}")

    # ── 2. ROC / EER Curve ───────────────────────────────────────────────────
    if len(bonafide_scores) > 0 and len(spoof_scores) > 0:
        y_true = np.concatenate([np.zeros(len(bonafide_scores)), np.ones(len(spoof_scores))])
        y_scores = np.concatenate([bonafide_scores, spoof_scores])

        # pos_label=0 (bonafide)
        fpr, tpr, _ = roc_curve(y_true, y_scores, pos_label=0)
        fnr = 1.0 - tpr
        idx = np.nanargmin(np.abs(fpr - fnr))
        eer = float((fpr[idx] + fnr[idx]) / 2.0)

        plt.figure(figsize=(7, 6), dpi=150)
        plt.plot(fpr * 100, fnr * 100, color="#3b82f6", lw=2, label="DET Curve (FAR vs FRR)")
        plt.plot([0, 100], [0, 100], color="#9ca3af", linestyle="--", label="Equal Error Diagonal")
        plt.scatter([eer * 100], [eer * 100], color="#dc2626", s=60, zorder=5, label=f"EER = {eer*100:.2f}%")

        plt.title(f"ROC / Detection Error Tradeoff — EER: {eer*100:.2f}%", fontsize=13, fontweight="bold")
        plt.xlabel("False Acceptance Rate (%) [Spoof accepted as Real]", fontsize=11)
        plt.ylabel("False Rejection Rate (%) [Real rejected as Fake]", fontsize=11)
        plt.xlim([-0.5, min(max(eer * 300, 10), 100)])
        plt.ylim([-0.5, min(max(eer * 300, 10), 100)])
        plt.legend(frameon=True)
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.tight_layout()

        roc_out = plots_dir / f"{args.run_name}_roc_curve.png"
        plt.savefig(roc_out)
        plt.close()
        print(f"Saved ROC/EER plot to: {roc_out}")


if __name__ == "__main__":
    main()
