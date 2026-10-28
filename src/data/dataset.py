"""
dataset.py — ASVspoof 2019 LA Dataset (Phase 2)
================================================
Parses ASVspoof protocol files and serves preprocessed audio tensors.

Protocol format (each line):
    <speaker> <filename> <env> <attack> <label>
    LA_0079 LA_T_1138215 - - bonafide

Label convention (same as eval.py):
    0 = bonafide
    1 = spoof

NOTE: this is NOT AASIST's output-class convention (index 1 = bonafide).
train.py flips labels (1 - label) before the loss. Do not feed these
labels straight into a loss on AASIST logits.

GOLDEN RULE: all preprocessing goes through src.preprocessing.preprocess.
             Nothing is done here that infer.py doesn't also do.
"""

import os
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset

from src.preprocessing.preprocess import preprocess


class ASVspoofDataset(Dataset):
    """
    PyTorch Dataset for ASVspoof 2019 LA (train / dev / eval splits).

    Parameters
    ----------
    protocol_path : str | Path
        Path to the ASVspoof CM protocol .txt file.
    audio_dir : str | Path
        Directory that contains the .flac (or .wav) audio files.
    mode : str
        'train' — enables RawBoost augmentation (via preprocess.py).
        'infer' — no augmentation (dev / eval sets).
    rawboost_type : int
        Which RawBoost algo to apply in train mode (default 5 = conv+impulsive).
        Ignored when mode='infer'.
    cache_tensors : bool
        If True, cache preprocessed tensors in RAM after first load.
        Speeds up repeated epochs at the cost of memory (~3 GB for full train set).
        Set False if you run out of RAM.

    Returns (from __getitem__)
    --------------------------
    waveform : torch.Tensor [1, 64600] float32
    label    : torch.Tensor scalar int64  (0=bonafide, 1=spoof)
    filename : str   (for debugging / scoring CSV)
    """

    LABEL_MAP = {"bonafide": 0, "spoof": 1}

    def __init__(
        self,
        protocol_path: str | Path,
        audio_dir: str | Path,
        mode: str = "train",
        rawboost_type: int = 5,
        cache_tensors: bool = False,
    ):
        if mode not in ("train", "infer"):
            raise ValueError(f"mode must be 'train' or 'infer', got {mode!r}")

        self.audio_dir = Path(audio_dir)
        self.mode = mode
        self.rawboost_type = rawboost_type
        self.cache_tensors = cache_tensors

        self._items: list[dict] = self._parse_protocol(protocol_path)
        self._cache: dict[int, torch.Tensor] = {}

        print(
            f"[ASVspoofDataset] {mode:>5s} | {len(self._items):6d} utterances "
            f"| {self.audio_dir.name}"
        )

    # ── Protocol parsing ─────────────────────────────────────────────────────

    def _parse_protocol(self, path: str | Path) -> list[dict]:
        """
        Parse the CM protocol file.  Skips blank / malformed lines.
        """
        items = []
        with open(path) as f:
            for lineno, line in enumerate(f, 1):
                parts = line.strip().split()
                if len(parts) < 5:
                    continue   # skip header / blank lines
                speaker, filename, env, attack, label_str = parts[:5]
                if label_str not in self.LABEL_MAP:
                    continue   # skip unknown labels
                items.append(
                    {
                        "speaker":  speaker,
                        "filename": filename,
                        "attack":   attack,
                        "label":    self.LABEL_MAP[label_str],
                    }
                )
        if not items:
            raise RuntimeError(f"Protocol file yielded no items: {path}")
        return items

    # ── Data loading ─────────────────────────────────────────────────────────

    def _resolve_audio_path(self, filename: str) -> Path:
        """Try .flac first, fall back to .wav."""
        for ext in (".flac", ".wav"):
            p = self.audio_dir / (filename + ext)
            if p.exists():
                return p
        raise FileNotFoundError(
            f"Audio file not found for {filename!r} in {self.audio_dir}"
        )

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        item = self._items[idx]

        # ── Cache hit ────────────────────────────────────────────────────────
        # Only cache in infer mode (RawBoost produces different tensors each call).
        if self.cache_tensors and self.mode == "infer" and idx in self._cache:
            waveform = self._cache[idx]
            return waveform, torch.tensor(item["label"], dtype=torch.long), item["filename"]

        # ── Load + preprocess (GOLDEN RULE) ──────────────────────────────────
        audio_path = self._resolve_audio_path(item["filename"])
        waveform = preprocess(
            str(audio_path),
            mode=self.mode,
            rawboost_type=self.rawboost_type,
        )  # [1, 64600] float32

        # ── Store in cache ───────────────────────────────────────────────────
        if self.cache_tensors and self.mode == "infer":
            self._cache[idx] = waveform

        label = torch.tensor(item["label"], dtype=torch.long)
        return waveform, label, item["filename"]


# ── Collate function for DataLoader ──────────────────────────────────────────

def collate_fn(
    batch: list[tuple[torch.Tensor, torch.Tensor, str]],
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    """
    Default collate for ASVspoofDataset.

    preprocess() returns [1, T].  Stacking B tensors gives [B, 1, T].
    AASIST.Model.forward expects [B, T] (it does its own unsqueeze(1)
    internally), so we squeeze out the channel dim here.
    """
    waveforms, labels, filenames = zip(*batch)
    # Each waveform: [1, 64600] → squeeze → [64600], then stack → [B, 64600]
    waveforms_stacked = torch.stack([w.squeeze(0) for w in waveforms])  # [B, T]
    return waveforms_stacked, torch.stack(labels), list(filenames)
