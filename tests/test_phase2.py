"""
test_phase2.py — Verification of Phase 2 Training Components
=============================================================
Tests:
1. LMCLLoss (CosFace) forward + backward gradient flow + build_loss factory.
2. ASVspoofDataset & collate_fn shape assertions ([B, 64600]).
3. Complete forward + loss + backward + optimizer step through AASIST-L.
"""

import json
import pytest
import torch
from pathlib import Path

from src.model.loss import LMCLLoss, CELoss, build_loss
from src.data.dataset import collate_fn
from src.model.AASIST import Model


def test_lmcl_loss_forward_backward():
    """Verify LMCLLoss produces finite loss and valid gradient flow."""
    criterion = LMCLLoss(num_classes=2, s=30.0, m=0.4)
    logits = torch.randn(4, 2, requires_grad=True)
    labels = torch.tensor([0, 1, 0, 1], dtype=torch.long)

    loss = criterion(logits, labels)
    assert torch.isfinite(loss), "LMCL loss returned non-finite value"
    assert loss.item() > 0, "Loss should be positive"

    loss.backward()
    assert logits.grad is not None, "Gradient did not flow back to logits"
    assert torch.all(torch.isfinite(logits.grad)), "Gradients contain NaN/Inf"


def test_build_loss_factory():
    """Test loss factory builder."""
    lmcl = build_loss("LMCL", num_classes=2)
    assert isinstance(lmcl, LMCLLoss)

    ce = build_loss("CE")
    assert isinstance(ce, CELoss)

    with pytest.raises(ValueError):
        build_loss("UNKNOWN_LOSS")


def test_collate_fn_shapes():
    """Verify collate_fn outputs 2D tensor [B, 64600] matching AASIST input."""
    item1 = (torch.randn(1, 64600), torch.tensor(0), "file1")
    item2 = (torch.randn(1, 64600), torch.tensor(1), "file2")
    batch = [item1, item2]

    waveforms, labels, filenames = collate_fn(batch)
    assert waveforms.shape == (2, 64600), f"Expected [2, 64600], got {waveforms.shape}"
    assert labels.shape == (2,)
    assert filenames == ["file1", "file2"]


def test_model_training_step():
    """Verify a complete forward + loss + backward + step pass through AASIST-L."""
    config_path = Path("configs/aasist_L.json")
    assert config_path.exists(), "configs/aasist_L.json not found"

    with open(config_path) as f:
        cfg = json.load(f)

    model = Model(cfg["model_config"])
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = LMCLLoss(num_classes=2, s=30.0, m=0.4)

    # Batch of 2 clips with shape [B, 64600]
    dummy_input = torch.randn(2, 64600)
    dummy_labels = torch.tensor([0, 1], dtype=torch.long)

    optimizer.zero_grad()
    _, logits = model(dummy_input)
    assert logits.shape == (2, 2), f"Expected logits [2, 2], got {logits.shape}"

    loss = criterion(logits, dummy_labels)
    assert torch.isfinite(loss)

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
    optimizer.step()
