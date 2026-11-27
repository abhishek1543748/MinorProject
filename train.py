"""
train.py — Training entry point (Phase 2)
==========================================
Usage:
    python train.py                                  # uses default config
    python train.py --config configs/train_config.json
    python train.py --config configs/train_config.json --loss CE   # ablation
    python train.py --config configs/train_config.json --epochs 5  # quick smoke-test

Fine-tune pretrained AASIST-L with extra real-world bonafide (Kaggle/Colab T4):
    python scripts/fetch_bonafide_diversity.py --n 2500
    python train.py --config configs/finetune_config.json
        --resume_weights outputs/weights/AASIST-L.pth
        --extra-train-protocol data/raw/bonafide_diversity/protocol_train.txt
        --extra-train-audio-dir data/raw/bonafide_diversity/flac
        --extra-dev-protocol data/raw/bonafide_diversity/protocol_dev.txt
        --extra-dev-audio-dir data/raw/bonafide_diversity/flac
        --workers 2 --save-name finetuned_aasist_l.pth
  (one command; shown on several lines)
  --resume_weights loads WEIGHTS ONLY (fresh optimizer, epoch 0, lr 1e-5 unless
  --lr is given). --resume (full checkpoint, continues the same run) is unchanged.

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

def get_args(argv=None):
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
    parser.add_argument("--resume_weights", "--resume-weights", default=None,
                        help="Fine-tune: init model from these weights only (raw state "
                             "dict like AASIST-L.pth, or a train.py checkpoint); fresh "
                             "optimizer/scheduler, epoch 0, default lr 1e-5")
    parser.add_argument("--lr",         type=float, default=None,
                        help="Override learning rate")
    parser.add_argument("--extra-train-protocol", action="append", default=[],
                        help="Extra protocol mixed into the train set (repeatable; "
                             "pair each with --extra-train-audio-dir)")
    parser.add_argument("--extra-train-audio-dir", action="append", default=[])
    parser.add_argument("--extra-dev-protocol", action="append", default=[],
                        help="Extra protocol added to the dev set used for model selection")
    parser.add_argument("--extra-dev-audio-dir", action="append", default=[])
    parser.add_argument("--save-name",  default="best_model.pth",
                        help="Checkpoint file name inside weights_dir")
    parser.add_argument("--select-protocol", default=None,
                        help="Pick the best checkpoint by EER on THIS set instead of dev "
                             "(e.g. DF-cal from scripts/make_df_protocols.py). Dev EER is "
                             "still computed and logged.")
    parser.add_argument("--select-audio-dir", default=None)
    args = parser.parse_args(argv)
    if bool(args.select_protocol) != bool(args.select_audio_dir):
        parser.error("--select-protocol and --select-audio-dir go together")
    if args.resume and args.resume_weights:
        parser.error("use --resume (continue a run) OR --resume_weights (fine-tune), not both")
    for kind in ("train", "dev"):
        protos = getattr(args, f"extra_{kind}_protocol")
        dirs = getattr(args, f"extra_{kind}_audio_dir")
        if len(protos) != len(dirs):
            parser.error(f"each --extra-{kind}-protocol needs one --extra-{kind}-audio-dir")
    return args


def load_weights_only(model: nn.Module, path: str, device) -> None:
    """Init from a raw state dict (AASIST-L.pth) or a train.py checkpoint (strict)."""
    state = torch.load(path, map_location=device)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict({k.removeprefix("module."): v for k, v in state.items()})


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

def main(argv=None):
    args = get_args(argv)

    # ── Config ───────────────────────────────────────────────────────────────
    with open(args.config) as f:
        cfg = json.load(f)

    train_cfg = cfg["training"]
    data_cfg  = cfg["data"]
    rb_cfg    = cfg.get("rawboost", {"enabled": True, "type": 5})

    epochs      = args.epochs     or train_cfg.get("epochs",     100)
    batch_size  = args.batch_size or train_cfg.get("batch_size", 24)
    # lr priority: --lr > 1e-5 when fine-tuning (--resume_weights) > config
    lr          = (args.lr if args.lr is not None
                   else 1e-5 if args.resume_weights
                   else float(train_cfg.get("lr", 1e-4)))
    wd          = float(train_cfg.get("weight_decay",  1e-4))
    loss_name   = args.loss       or train_cfg.get("loss",      "LMCL")
    rawboost_type = int(rb_cfg.get("type", 5)) if rb_cfg.get("enabled", True) else None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] Device:     {device}")
    print(f"[train] Epochs:     {epochs}")
    print(f"[train] Batch size: {batch_size}")
    print(f"[train] Loss:       {loss_name}")
    print(f"[train] RawBoost:   type {rawboost_type}")
    print(f"[train] LR:         {lr:.1e}")

    # ── Directories ──────────────────────────────────────────────────────────
    weights_dir = ROOT / cfg["paths"]["weights_dir"]
    scores_dir  = ROOT / cfg["paths"]["scores_dir"]
    weights_dir.mkdir(parents=True, exist_ok=True)
    scores_dir.mkdir(parents=True, exist_ok=True)

    # ── Datasets ─────────────────────────────────────────────────────────────
    from torch.utils.data import ConcatDataset
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
    # Extra protocols (e.g. scripts/fetch_bonafide_diversity.py). Same
    # ASVspoofDataset -> same preprocess() + RawBoost path (GOLDEN RULE).
    extra_train = [ASVspoofDataset(p, d, mode="train", rawboost_type=rawboost_type or 5)
                   for p, d in zip(args.extra_train_protocol, args.extra_train_audio_dir)]
    extra_dev = [ASVspoofDataset(p, d, mode="infer", cache_tensors=not args.no_cache)
                 for p, d in zip(args.extra_dev_protocol, args.extra_dev_audio_dir)]
    if extra_train:
        train_set = ConcatDataset([train_set, *extra_train])
    if extra_dev:
        dev_set = ConcatDataset([dev_set, *extra_dev])
    print(f"[train] Train items: {len(train_set)}   Dev items: {len(dev_set)}")

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
    select_loader = None
    if args.select_protocol:
        select_set = ASVspoofDataset(args.select_protocol, args.select_audio_dir, mode="infer",
                                     cache_tensors=not args.no_cache)
        select_loader = DataLoader(select_set, batch_size=batch_size * 2, shuffle=False,
                                   num_workers=args.workers, collate_fn=collate_fn,
                                   pin_memory=(device.type == "cuda"))
        print(f"[train] Checkpoint selection set: {len(select_set)} items ({args.select_protocol})")
    select_name = "select" if select_loader else "dev"

    # ── Model ────────────────────────────────────────────────────────────────
    from src.model.AASIST import Model
    with open(cfg["model_config"]) as f:
        model_cfg = json.load(f)
    model = Model(model_cfg["model_config"]).to(device)

    start_epoch = 0
    best_dev_eer = 1.0

    if args.resume_weights:
        load_weights_only(model, args.resume_weights, device)
        print(f"[train] Fine-tuning from weights {args.resume_weights}")
        # A fine-tune epoch is only saved if it BEATS the starting model on the
        # same (possibly mixed) dev set — otherwise fine-tuning made it worse.
        best_dev_eer = evaluate(model, dev_loader, device)
        print(f"[train] Starting-point dev EER: {best_dev_eer*100:.4f}%")
        if select_loader:
            best_dev_eer = evaluate(model, select_loader, device)
            print(f"[train] Starting-point select EER: {best_dev_eer*100:.4f}%")

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
    log_path = scores_dir / ("train_log.csv" if args.save_name == "best_model.pth"
                             else f"{Path(args.save_name).stem}_log.csv")
    log_file = open(log_path, "w", newline="")
    log_writer = csv.writer(log_file)
    log_writer.writerow(["epoch", "train_loss", "dev_eer", "lr", "elapsed_s", "select_eer"])
    log_file.flush()
    last_path = weights_dir / f"{Path(args.save_name).stem}_last.pth"

    print(f"\n[train] Log:        {log_path}")
    print(f"[train] Best model: {weights_dir / args.save_name}  (by {select_name} EER)")
    print(f"[train] Last epoch: {last_path}  (always written)")
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
        select_eer = evaluate(model, select_loader, device) if select_loader else None
        avg_loss   = epoch_loss / max(n_batches, 1)
        elapsed    = time.time() - t0
        current_lr = scheduler.get_last_lr()[0]
        scheduler.step()

        print(
            f"Epoch {epoch+1:3d}/{epochs}  "
            f"loss={avg_loss:.4f}  "
            f"dev_EER={dev_eer*100:.4f}%  "
            + (f"select_EER={select_eer*100:.4f}%  " if select_loader else "")
            + f"lr={current_lr:.2e}  "
            f"time={elapsed:.0f}s",
            flush=True,
        )

        # ── Checkpoint ───────────────────────────────────────────────────────
        crit = select_eer if select_loader else dev_eer
        improved = crit < best_dev_eer
        if improved:
            best_dev_eer = crit
        ckpt = {
            "epoch":        epoch,
            "model":        model.state_dict(),
            "optimizer":    optimizer.state_dict(),
            "best_dev_eer": best_dev_eer,
            "dev_eer":      dev_eer,
            "select_eer":   select_eer,
            "loss":         loss_name,
            "config":       cfg,
        }
        # Last epoch is always kept: a fine-tune can lose on LA dev yet win
        # cross-domain, so never throw the weights away (eval them on DF).
        torch.save(ckpt, last_path)
        if improved:
            ckpt_path = weights_dir / args.save_name
            torch.save(ckpt, ckpt_path)
            print(f"  ✓ New best — saved to {ckpt_path}  ({select_name} EER={best_dev_eer*100:.4f}%)")

        # ── CSV row ──────────────────────────────────────────────────────────
        log_writer.writerow([epoch + 1, round(avg_loss, 6), round(dev_eer, 6),
                             round(current_lr, 8), round(elapsed, 1),
                             "" if select_eer is None else round(select_eer, 6)])
        log_file.flush()

    # ── Done ──────────────────────────────────────────────────────────────────
    log_file.close()
    print("=" * 60)
    print(f"Training complete.  Best {select_name} EER: {best_dev_eer*100:.4f}%")
    if not (weights_dir / args.save_name).exists():
        print(f"WARN: no epoch beat the starting {select_name} EER — no best checkpoint "
              f"written. Last-epoch weights are in {last_path}")
    print(f"Weights: {weights_dir / args.save_name}")
    print(f"Last:    {last_path}")
    print(f"Log:     {log_path}")


if __name__ == "__main__":
    main()
