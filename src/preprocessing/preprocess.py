"""
preprocess.py — GOLDEN RULE MODULE
===================================
This module is the single source of truth for ALL audio preprocessing.
It is imported and used identically by:
  - train.py  (with mode='train' to enable RawBoost)
  - infer.py  (with mode='infer', no augmentation)
  - eval.py   (with mode='infer')

NEVER duplicate this logic. Any mismatch silently destroys performance.
"""

import torch
import torchaudio
import numpy as np
from pathlib import Path

# ── Constants (lock these in — changing them breaks train/infer parity) ──────
TARGET_SR   = 16000          # model was trained at 16 kHz
TARGET_LEN  = 64600          # 4.025 seconds at 16 kHz (AASIST standard)
# ─────────────────────────────────────────────────────────────────────────────


def load_audio(path: str) -> tuple[torch.Tensor, int]:
    """
    Decode any audio file to a float32 waveform.
    Supports: wav, flac, mp3, ogg, m4a (via torchaudio / soundfile).
    Returns: (waveform [C, T], sample_rate)
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")
    waveform, sr = torchaudio.load(str(path))
    return waveform, sr


def to_mono(waveform: torch.Tensor) -> torch.Tensor:
    """
    Downmix to mono by averaging channels.
    Input:  [C, T]  (C channels)
    Output: [1, T]
    """
    if waveform.shape[0] == 1:
        return waveform
    return waveform.mean(dim=0, keepdim=True)


def resample(waveform: torch.Tensor, orig_sr: int) -> torch.Tensor:
    """
    Resample to TARGET_SR (16 kHz) using a polyphase resampler.
    Never use naive sample dropping — it causes aliasing artifacts.
    """
    if orig_sr == TARGET_SR:
        return waveform
    resampler = torchaudio.transforms.Resample(
        orig_freq=orig_sr,
        new_freq=TARGET_SR,
        resampling_method="sinc_interp_hann",
    )
    return resampler(waveform)


def normalize(waveform: torch.Tensor) -> torch.Tensor:
    """
    Peak-normalize to [-1, 1].
    Prevents the model keying on absolute loudness.
    """
    peak = waveform.abs().max()
    if peak > 0:
        waveform = waveform / peak
    return waveform


def fix_length(waveform: torch.Tensor) -> torch.Tensor:
    """
    Crop or pad to exactly TARGET_LEN samples.
    - Longer clips: crop from the start (deterministic for inference).
    - Shorter clips: repeat-pad (wrap-around) to fill.
    Output: [1, TARGET_LEN]
    """
    wav = waveform.squeeze(0)       # [T]
    length = wav.shape[0]

    if length >= TARGET_LEN:
        wav = wav[:TARGET_LEN]
    else:
        # Repeat-pad: tile the signal until long enough, then crop
        repeats = (TARGET_LEN // length) + 1
        wav = wav.repeat(repeats)[:TARGET_LEN]

    return wav.unsqueeze(0)         # [1, TARGET_LEN]


def preprocess(
    path: str,
    mode: str = "infer",
    rawboost_type: int = 5,         # see src/features/rawboost.py ALGO NUMBERING
) -> torch.Tensor:
    """
    Full preprocessing pipeline. Returns shape [1, 64600] float32 tensor.

    Parameters
    ----------
    path : str
        Path to the audio file.
    mode : str
        'train' — applies RawBoost augmentation (training only).
        'infer' — no augmentation (inference and evaluation).
    rawboost_type : int
        Which RawBoost noise process to apply (training mode only).
        Numbering is upstream RawBoost's, NOT arbitrary:
        5 = convolutive + impulsive (best for LA/microphone condition) <- default
        3 = SSI coloured additive  (better for DF/codec condition, Phase 7)

    Returns
    -------
    torch.Tensor of shape [1, 64600], dtype float32.
    """
    if mode not in ("train", "infer"):
        raise ValueError(f"mode must be 'train' or 'infer', got '{mode}'")

    # ── Stage 1: Decode ──────────────────────────────────────────────────────
    waveform, sr = load_audio(path)

    # ── Stage 2: Mono ────────────────────────────────────────────────────────
    waveform = to_mono(waveform)

    # ── Stage 3: Resample ────────────────────────────────────────────────────
    waveform = resample(waveform, sr)

    # ── Stage 4: Normalize ───────────────────────────────────────────────────
    waveform = normalize(waveform)

    # ── Stage 5: Fix length ──────────────────────────────────────────────────
    waveform = fix_length(waveform)

    # ── Stage 6: RawBoost (training only) ───────────────────────────────────
    if mode == "train":
        # Deliberately NOT wrapped in try/except. A silently skipped
        # augmentation is exactly the train/infer mismatch the GOLDEN RULE
        # exists to prevent — training would run un-augmented and look fine.
        from src.features.rawboost import process_Rawboost_feature
        wav_np = waveform.squeeze(0).numpy()
        wav_np = process_Rawboost_feature(wav_np, TARGET_SR, algo=rawboost_type)
        waveform = torch.from_numpy(wav_np).unsqueeze(0).float()
        # Re-normalize after augmentation (RawBoost does not bound amplitude)
        waveform = normalize(waveform)

    return waveform.float()
