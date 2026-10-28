"""
train.py — Training entry point (Phase 2)
==========================================
Usage:
    python train.py                                  # uses default config
    python train.py --config configs/train_config.json
    python train.py --config configs/train_config.json --loss CE   # ablation
    python train.py --config configs/train_config.json --epochs 5  # quick smoke-test

GOLDEN RULE: preprocessing is performed by src.preprocessing.preprocess —
             the exact same module used in infer.py and eval.py.

Checkpoint saved to: outputs/weights/best_model.pth  (best dev EER)
Progress log saved to: outputs/scores/train_log.csv
"""

import csv
import json
import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).parent


# ── CLI ───────────────────────────────────────────────────────────────────────

def get_args():
    parser = argparse.ArgumentParser(description="Train AASIST-L on ASVspoof 2019 LA")
    parser.add_argument("--config",     default="configs/train_config.json",
                        help="Training config JSON")
    parser.add_argument("--loss",       default=None,
                        choices=["LMCL", "CE"],
                        help="Override loss function (default: from config)")
    parser.add_argument("--epochs",     type=int, default=None,
                        help="Override number of epochs")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override batch size")
    parser.add_argument("--workers",    type=int, default=0,
                        help="DataLoader num_workers (0 = main thread, safe on Windows)")
    parser.add_argument("--no-cache",   action="store_true",
                        help="Disable tensor caching for dev set (saves RAM)")
    parser.add_argument("--resume",     default=None,
                        help="Path to checkpoint .pth to resume training from")
    return parser.parse_args()


# ── EER (fast, threshold sweep on dev set) ───────────────────────────────────

def compute_eer_fast(scores, labels) -> float:
    """
    Vectorised EER computation — no scipy needed.
    scores : list/array of floats  (bonafide LLR: high = bonafide)
    labels : list/array of ints    (0=bonafide, 1=spoof)
    Returns EER as fraction [0, 1].
    """
    import numpy as np
    scores = torch.tensor(scores, dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.int32)

    thresholds = torch.linspace(scores.min().item(), scores.max().item(), steps=1000)

    spoof_scores   = scores[labels == 1]
    bonafide_scores = scores[labels == 0]

    # far/frr at every threshold: [T]
    far = (spoof_scores[None, :]    >= thresholds[:, None]).float().mean(dim=1)  # spoof accepted as bonafide
    frr = (bonafide_scores[None, :] <  thresholds[:, None]).float().mean(dim=1)  # bonafide rejected

    # EER = operating point where far and frr cross
    i = torch.argmin((far - frr).abs())
    return ((far[i] + frr[i]) / 2).item()


# ── Evaluation pass (dev set) ─────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    """
    Runs the model over a DataLoader and returns EER.
    """
    import torch.nn.functional as F
    model.eval()

    all_scores, all_labels = [], []
    for waveforms, labels, _ in loader:
        waveforms = waveforms.to(device)   # [B, 1, 64600]
        _, logits = model(waveforms)       # (last_hidden, [B, 2])
        # bonafide LLR: high score → bonafide  (idx1=bonafide, idx0=spoof)
        scores = (logits[:, 1] - logits[:, 0]).cpu().tolist()
        all_scores.extend(scores)
        all_labels.extend(labels.tolist())

    return compute_eer_fast(all_scores, all_labels)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = get_args()

    # ── Config ───────────────────────────────────────────────────────────────
    with open(args.config) as f:
        cfg = json.load(f)

    train_cfg = cfg["training"]
    data_cfg  = cfg["data"]
    rb_cfg    = cfg.get("rawboost", {"enabled": True, "type": 5})

    epochs      = args.epochs     or train_cfg.get("epochs",     100)
    batch_size  = args.batch_size or train_cfg.get("batch_size", 24)
    lr          = float(train_cfg.get("lr",            1e-4))
    wd          = float(train_cfg.get("weight_decay",  1e-4))
    loss_name   = args.loss       or train_cfg.get("loss",      "LMCL")
    rawboost_type = int(rb_cfg.get("type", 5)) if rb_cfg.get("enabled", True) else None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] Device:     {device}")
    print(f"[train] Epochs:     {epochs}")
    print(f"[train] Batch size: {batch_size}")
    print(f"[train] Loss:       {loss_name}")
    print(f"[train] RawBoost:   type {rawboost_type}")

    # ── Directories ──────────────────────────────────────────────────────────
    weights_dir = ROOT / cfg["paths"]["weights_dir"]
    scores_dir  = ROOT / cfg["paths"]["scores_dir"]
    weights_dir.mkdir(parents=True, exist_ok=True)
    scores_dir.mkdir(parents=True, exist_ok=True)

    # ── Datasets ─────────────────────────────────────────────────────────────
    from src.data.dataset import ASVspoofDataset, collate_fn

    train_set = ASVspoofDataset(
        protocol_path  = data_cfg["train_protocol"],
        audio_dir      = data_cfg["train_audio_dir"],
        mode           = "train",
        rawboost_type  = rawboost_type or 5,
        cache_tensors  = False,   # never cache train (RawBoost is stochastic)
    )
    dev_set = ASVspoofDataset(
        protocol_path  = data_cfg["dev_protocol"],
        audio_dir      = data_cfg["dev_audio_dir"],
        mode           = "infer",
        cache_tensors  = not args.no_cache,   # cache dev tensors — saves time each epoch
    )

    train_loader = DataLoader(
        train_set,
        batch_size  = batch_size,
        shuffle     = True,
        num_workers = args.workers,
        collate_fn  = collate_fn,
        pin_memory  = (device.type == "cuda"),
        drop_last   = True,   # keeps batch size stable for LMCL
    )
    dev_loader = DataLoader(
        dev_set,
        batch_size  = batch_size * 2,   # no grads → can use larger batch
        shuffle     = False,
        num_workers = args.workers,
        collate_fn  = collate_fn,
        pin_memory  = (device.type == "cuda"),
    )

    # ── Model ────────────────────────────────────────────────────────────────
    from src.model.AASIST import Model
    with open(cfg["model_config"]) as f:
        model_cfg = json.load(f)
    model = Model(model_cfg["model_config"]).to(device)

    start_epoch = 0
    best_dev_eer = 1.0

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        start_epoch  = ckpt.get("epoch", 0) + 1
        best_dev_eer = ckpt.get("best_dev_eer", 1.0)
        print(f"[train] Resumed from {args.resume}  (epoch {start_epoch}, best EER {best_dev_eer:.4f})")

    # ── Loss ─────────────────────────────────────────────────────────────────
    from src.model.loss import build_loss
    criterion = build_loss(loss_name, num_classes=2, s=30.0, m=0.4)
    print(f"[train] Loss module: {criterion.__class__.__name__}")

    # ── Optimizer + scheduler ─────────────────────────────────────────────────
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    # Cosine annealing: smoothly reduces lr to 1e-6 over full training
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6
    )

    # ── Log CSV ──────────────────────────────────────────────────────────────
    log_path = scores_dir / "train_log.csv"
    log_file = open(log_path, "w", newline="")
    log_writer = csv.writer(log_file)
    log_writer.writerow(["epoch", "train_loss", "dev_eer", "lr", "elapsed_s"])
    log_file.flush()

    print(f"\n[train] Log:        {log_path}")
    print(f"[train] Best model: {weights_dir / 'best_model.pth'}")
    print("=" * 60)

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(start_epoch, epochs):
        t0 = time.time()
        model.train()

        epoch_loss = 0.0
        n_batches  = 0

        for batch_idx, (waveforms, labels, _) in enumerate(train_loader):
            waveforms = waveforms.to(device)   # [B, 1, 64600]
            labels    = labels.to(device)      # [B]

            optimizer.zero_grad(set_to_none=True)

            _, logits = model(waveforms)       # (last_hidden, [B, 2])
            # Dataset labels are 0=bonafide/1=spoof (eval.py convention), but
            # AASIST's output index 1 = bonafide. Flip to class indices so a
            # trained checkpoint matches the pretrained weights and the
            # score = logits[:,1] - logits[:,0] used everywhere downstream.
            targets = 1 - labels
            loss = criterion(logits, targets)

            loss.backward()
            # Gradient clipping prevents occasional large updates that destabilise LMCL
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches  += 1

            # Progress update every 100 batches
            if (batch_idx + 1) % 100 == 0:
                avg = epoch_loss / n_batches
                print(
                    f"  Epoch {epoch+1}/{epochs}  "
                    f"batch {batch_idx+1}/{len(train_loader)}  "
                    f"loss={avg:.4f}",
                    flush=True,
                )

        # ── Dev evaluation ───────────────────────────────────────────────────
        dev_eer    = evaluate(model, dev_loader, device)
        avg_loss   = epoch_loss / max(n_batches, 1)
        elapsed    = time.time() - t0
        current_lr = scheduler.get_last_lr()[0]
        scheduler.step()

        print(
            f"Epoch {epoch+1:3d}/{epochs}  "
            f"loss={avg_loss:.4f}  "
            f"dev_EER={dev_eer*100:.4f}%  "
            f"lr={current_lr:.2e}  "
            f"time={elapsed:.0f}s",
            flush=True,
        )

        # ── Checkpoint ───────────────────────────────────────────────────────
        if dev_eer < best_dev_eer:
            best_dev_eer = dev_eer
            ckpt_path = weights_dir / "best_model.pth"
            torch.save(
                {
                    "epoch":        epoch,
                    "model":        model.state_dict(),
                    "optimizer":    optimizer.state_dict(),
                    "best_dev_eer": best_dev_eer,
                    "loss":         loss_name,
                    "config":       cfg,
                },
                ckpt_path,
            )
            print(f"  ✓ New best — saved to {ckpt_path}  (EER={best_dev_eer*100:.4f}%)")

        # ── CSV row ──────────────────────────────────────────────────────────
        log_writer.writerow([epoch + 1, round(avg_loss, 6), round(dev_eer, 6),
                             round(current_lr, 8), round(elapsed, 1)])
        log_file.flush()

    # ── Done ──────────────────────────────────────────────────────────────────
    log_file.close()
    print("=" * 60)
    print(f"Training complete.  Best dev EER: {best_dev_eer*100:.4f}%")
    print(f"Weights: {weights_dir / 'best_model.pth'}")
    print(f"Log:     {log_path}")


if __name__ == "__main__":
    main()
