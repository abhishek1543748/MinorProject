"""
train.py — Training entry point (Phase 2)
==========================================
Usage:
    python train.py --config configs/train_config.json

GOLDEN RULE: uses src.preprocessing.preprocess — same module as infer.py.
"""

import json
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).parent


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_config.json")
    return parser.parse_args()


def main():
    args = get_args()
    with open(args.config) as f:
        cfg = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    # ── Dataset ──────────────────────────────────────────────────────────────
    # TODO Phase 2: implement ASVspoofDataset
    # from src.data.dataset import ASVspoofDataset
    # train_set = ASVspoofDataset(cfg["data"]["train_protocol"], mode="train")
    # dev_set   = ASVspoofDataset(cfg["data"]["dev_protocol"],   mode="infer")

    # ── Model ────────────────────────────────────────────────────────────────
    from src.model.AASIST import Model
    with open(cfg["model_config"]) as f:
        model_cfg = json.load(f)
    model = Model(model_cfg["model_config"]).to(device)

    # ── Loss ─────────────────────────────────────────────────────────────────
    # TODO Phase 2: from src.model.loss import LMCLLoss
    criterion = nn.CrossEntropyLoss()   # placeholder; replace with LMCL

    # ── Optimizer ────────────────────────────────────────────────────────────
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.get("lr", 1e-4),
        weight_decay=cfg.get("weight_decay", 1e-4),
    )

    best_dev_eer = 1.0
    num_epochs = cfg.get("epochs", 100)

    for epoch in range(num_epochs):
        model.train()
        # TODO Phase 2: training loop over DataLoader
        # for batch_x, batch_y in train_loader:
        #     optimizer.zero_grad()
        #     out = model(batch_x.to(device))
        #     loss = criterion(out, batch_y.to(device))
        #     loss.backward()
        #     optimizer.step()

        # ── Evaluate on dev set ──────────────────────────────────────────────
        # dev_eer = evaluate(model, dev_loader, device)
        # print(f"Epoch {epoch+1}/{num_epochs}  dev_EER={dev_eer:.4f}")
        # if dev_eer < best_dev_eer:
        #     best_dev_eer = dev_eer
        #     torch.save(model.state_dict(), "outputs/weights/best_model.pth")
        pass

    print(f"Training complete. Best dev EER: {best_dev_eer:.4f}")


if __name__ == "__main__":
    main()
