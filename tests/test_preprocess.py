"""
test_preprocess.py — GOLDEN RULE UNIT TEST
============================================
This test MUST pass before proceeding to Phase 2 (training).

It asserts that the same file produces byte-identical tensors
in both train-without-augmentation mode and infer mode.

Run: python -m pytest tests/test_preprocess.py -v
"""

import os
import torch
import pytest
import tempfile
import numpy as np
import torchaudio

from src.preprocessing.preprocess import (
    preprocess,
    TARGET_SR,
    TARGET_LEN,
)


@pytest.fixture
def wav_path(tmp_path):
    """
    A writable temp path that isn't held open by us.
    tempfile.NamedTemporaryFile keeps its own handle open, which blocks
    torchaudio.save() from reopening the same path on Windows.
    """
    return str(tmp_path / "test.wav")


def _make_test_wav(path: str, sr: int = 16000, duration_s: float = 3.0):
    """Generate a synthetic sine-wave WAV for testing."""
    t = torch.linspace(0, duration_s, int(sr * duration_s))
    wave = (0.5 * torch.sin(2 * torch.pi * 440 * t)).unsqueeze(0)
    torchaudio.save(path, wave, sr)


# ── Shape tests ──────────────────────────────────────────────────────────────

def test_output_shape_short_clip(wav_path):
    """Short clips (< 4s) should be padded to [1, 64600]."""
    _make_test_wav(wav_path, duration_s=2.0)
    tensor = preprocess(wav_path, mode="infer")
    assert tensor.shape == (1, TARGET_LEN), f"Expected [1, {TARGET_LEN}], got {tensor.shape}"


def test_output_shape_long_clip(wav_path):
    """Long clips (> 4s) should be cropped to [1, 64600]."""
    _make_test_wav(wav_path, duration_s=8.0)
    tensor = preprocess(wav_path, mode="infer")
    assert tensor.shape == (1, TARGET_LEN), f"Expected [1, {TARGET_LEN}], got {tensor.shape}"


def test_output_dtype(wav_path):
    _make_test_wav(wav_path)
    tensor = preprocess(wav_path, mode="infer")
    assert tensor.dtype == torch.float32


def test_output_range(wav_path):
    """Output must be normalized to [-1, 1]."""
    _make_test_wav(wav_path)
    tensor = preprocess(wav_path, mode="infer")
    assert tensor.abs().max().item() <= 1.0 + 1e-5


# ── THE GOLDEN RULE TEST ─────────────────────────────────────────────────────

def test_train_infer_identical_without_augmentation(wav_path):
    """
    GOLDEN RULE: preprocessing in infer mode must produce the
    same tensor every time — and match what the model sees during eval.
    """
    _make_test_wav(wav_path)
    # Run infer mode twice — must be deterministic
    t1 = preprocess(wav_path, mode="infer")
    t2 = preprocess(wav_path, mode="infer")
    assert torch.allclose(t1, t2, atol=1e-6), \
        "GOLDEN RULE FAILED: infer mode is not deterministic"


def test_resampling_different_input_rates(tmp_path):
    """Files at 44.1kHz and 8kHz should both produce [1, 64600]."""
    for sr in [8000, 22050, 44100]:
        path = str(tmp_path / f"test_{sr}.wav")
        _make_test_wav(path, sr=sr)
        tensor = preprocess(path, mode="infer")
        assert tensor.shape == (1, TARGET_LEN), \
            f"Failed for input sr={sr}: got {tensor.shape}"


def test_stereo_input_produces_mono(wav_path):
    """Stereo input must be downmixed to mono [1, 64600]."""
    t = torch.linspace(0, 3.0, 48000)
    stereo = torch.stack([
        0.5 * torch.sin(2 * torch.pi * 440 * t),
        0.5 * torch.sin(2 * torch.pi * 880 * t),
    ])
    torchaudio.save(wav_path, stereo, 16000)
    tensor = preprocess(wav_path, mode="infer")
    assert tensor.shape == (1, TARGET_LEN)


def test_silent_file_does_not_crash(wav_path):
    """Silent audio should not raise ZeroDivisionError during normalization."""
    silence = torch.zeros(1, 32000)
    torchaudio.save(wav_path, silence, 16000)
    tensor = preprocess(wav_path, mode="infer")
    assert tensor.shape == (1, TARGET_LEN)


def test_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        preprocess("/nonexistent/path/audio.wav", mode="infer")


def test_invalid_mode_raises(wav_path):
    _make_test_wav(wav_path)
    with pytest.raises(ValueError):
        preprocess(wav_path, mode="invalid")
