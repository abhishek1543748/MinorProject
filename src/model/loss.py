"""
loss.py — LMCL (Large Margin Cosine Loss) + Cross-Entropy fallback
===================================================================
LMCL reference:  Wang et al., "CosFace" (CVPR 2018)
                 https://arxiv.org/pdf/1801.09414

In anti-spoofing the convention is:
  - class 0 = spoof
  - class 1 = bonafide

The loss encourages the model to produce well-separated cosine scores
with a margin s to discourage over-confident predictions near the
decision boundary — empirically this gives better EER than plain CE
on ASVspoof 2019 LA.

Usage:
    criterion = LMCLLoss(num_classes=2, s=30.0, m=0.4)
    loss = criterion(logits, labels)   # logits: [B,2], labels: [B] (LongTensor)

    If you prefer plain CE (e.g. for fast ablations), just pass use_lmcl=False
    to get a standard CrossEntropyLoss wrapper with the same interface.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LMCLLoss(nn.Module):
    """
    Large Margin Cosine Loss (CosFace).

    Parameters
    ----------
    num_classes : int
        Number of output classes (2 for anti-spoofing: spoof / bonafide).
    s : float
        Feature scale (temperature). Typically 30.0.
    m : float
        Cosine margin. Typically 0.2–0.4.
        Larger m → larger inter-class angular gap → better EER but harder training.

    Forward
    -------
    Input:
        logits : Tensor [B, num_classes] — raw logits from AASIST's classifier head.
        labels : LongTensor [B]          — ground-truth class indices.
    Output:
        scalar loss.

    NOTE: AASIST's out_layer already outputs un-normalised dot products
    (it has no explicit L2 normalisation on features or weights).  We
    apply L2-normalisation here so the "cosine" interpretation holds.
    If you find training unstable, try reducing s to 15 or m to 0.2.
    """

    def __init__(self, num_classes: int = 2, s: float = 30.0, m: float = 0.4):
        super().__init__()
        self.s = s
        self.m = m
        self.num_classes = num_classes

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        logits : [B, C]  raw logits (not softmaxed)
        labels : [B]     LongTensor in {0, …, C-1}
        """
        # L2-normalise along class axis so values live on the unit hypersphere
        cosine = F.normalize(logits, p=2, dim=1)          # [B, C]

        # Build one-hot mask to subtract margin only from the target class
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)

        # Subtract margin from ground-truth cosine score
        margin_cosine = cosine - one_hot * self.m          # [B, C]

        # Scale and apply cross-entropy
        scaled = self.s * margin_cosine
        loss = F.cross_entropy(scaled, labels)
        return loss


class CELoss(nn.Module):
    """
    Plain Cross-Entropy wrapper — same interface as LMCLLoss for easy swap.
    """

    def __init__(self, num_classes: int = 2, **kwargs):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return self.ce(logits, labels)


def build_loss(loss_name: str = "LMCL", **kwargs) -> nn.Module:
    """
    Factory: returns the correct loss module given a string name.

    Parameters
    ----------
    loss_name : str
        "LMCL" (default) or "CE".
    **kwargs
        Forwarded to the loss constructor (e.g. s=30.0, m=0.4).
    """
    loss_name = loss_name.upper()
    if loss_name == "LMCL":
        return LMCLLoss(**kwargs)
    elif loss_name in ("CE", "CROSSENTROPY"):
        return CELoss(**kwargs)
    else:
        raise ValueError(f"Unknown loss: {loss_name!r}. Choose 'LMCL' or 'CE'.")
