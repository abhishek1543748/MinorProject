"""
test_phase6.py — Grad-CAM + /analyze endpoint
"""

import base64
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torchaudio

ROOT = Path(__file__).resolve().parent.parent
WEIGHTS = ROOT / "outputs/weights/AASIST-L.pth"
CONFIG = ROOT / "configs/aasist_L.json"

needs_weights = pytest.mark.skipif(not WEIGHTS.exists(), reason="AASIST-L.pth missing")


@pytest.fixture(scope="module")
def model():
    from infer import load_model
    return load_model(WEIGHTS, CONFIG, torch.device("cpu"))


@pytest.fixture
def wav_path(tmp_path):
    torch.manual_seed(0)
    p = tmp_path / "noise.wav"
    torchaudio.save(str(p), 0.1 * torch.randn(1, 16000 * 2), 16000)
    return p


@needs_weights
def test_gradcam_shape_range_and_score(model, wav_path):
    from src.preprocessing.preprocess import preprocess
    from src.explainability.gradcam import GradCAM

    w = preprocess(str(wav_path), mode="infer")
    r = GradCAM(model)(w)

    assert r.cam.ndim == 2 and r.time_profile.shape == (r.cam.shape[1],)
    assert r.cam.min() >= 0 and r.cam.max() <= 1
    with torch.no_grad():
        _, logits = model(w)
    assert r.spoof_score == pytest.approx((logits[0, 0] - logits[0, 1]).item(), abs=1e-4)


@needs_weights
def test_gradcam_deterministic(model, wav_path):
    from src.preprocessing.preprocess import preprocess
    from src.explainability.gradcam import GradCAM

    w = preprocess(str(wav_path), mode="infer")
    cam = GradCAM(model)
    assert np.allclose(cam(w).cam, cam(w).cam)


@needs_weights
def test_render_png(model, wav_path):
    from src.preprocessing.preprocess import preprocess
    from src.explainability.gradcam import GradCAM, render_png

    w = preprocess(str(wav_path), mode="infer")
    png = render_png(w, GradCAM(model)(w))
    assert png[:8] == b"\x89PNG\r\n\x1a\n"


@needs_weights
def test_analyze_endpoint(wav_path):
    from fastapi.testclient import TestClient
    from src.api.server import app

    with TestClient(app) as client:
        with open(wav_path, "rb") as f:
            r = client.post("/analyze", files={"file": ("noise.wav", f, "audio/wav")})
        assert r.status_code == 200
        d = r.json()
        assert d["band"] in ("authentic", "uncertain", "spoof")
        assert "label" not in d                      # no bare binary verdict
        assert 0.0 <= d["prob_spoof"] <= 1.0
        assert base64.b64decode(d["heatmap_png"])[:4] == b"\x89PNG"

        with open(wav_path, "rb") as f:
            r2 = client.post("/analyze?explain=false", files={"file": ("noise.wav", f, "audio/wav")})
        assert r2.json()["heatmap_png"] is None
        assert r2.json()["raw_score"] == pytest.approx(d["raw_score"], abs=1e-3)


# ── Sliding windows ─────────────────────────────────────────────────────────

@pytest.fixture
def long_wav(tmp_path):
    torch.manual_seed(1)
    p = tmp_path / "long.wav"
    torchaudio.save(str(p), 0.1 * torch.randn(1, 22050 * 10), 22050)   # 10 s, 22.05 kHz
    return p


def test_windows_first_matches_preprocess(long_wav, wav_path):
    """GOLDEN RULE: window 0 must equal the single-window preprocess output."""
    from src.preprocessing.preprocess import preprocess, preprocess_windows, TARGET_LEN

    for p in (long_wav, wav_path):          # long clip and short (padded) clip
        w = preprocess_windows(str(p))
        assert w.windows.shape[1] == TARGET_LEN
        assert torch.allclose(w.windows[0:1], preprocess(str(p), mode="infer"))


def test_windows_cover_whole_clip(long_wav):
    from src.preprocessing.preprocess import preprocess_windows, TARGET_LEN, WINDOW_HOP

    w = preprocess_windows(str(long_wav))
    T = w.clip.shape[1]
    assert w.starts[0] == 0
    assert w.starts[-1] + TARGET_LEN == T                    # tail covered
    assert all(b - a <= WINDOW_HOP for a, b in zip(w.starts, w.starts[1:]))  # no gaps
    assert not w.truncated


def test_windows_truncate(long_wav):
    from src.preprocessing.preprocess import preprocess_windows

    w = preprocess_windows(str(long_wav), max_windows=2)
    assert len(w.starts) == 2 and w.truncated


@needs_weights
def test_gradcam_clip_stitch(model, long_wav):
    from src.preprocessing.preprocess import preprocess_windows
    from src.explainability.gradcam import GradCAM

    w = preprocess_windows(str(long_wav))
    cam = GradCAM(model)
    res, scores = cam.clip(w.windows, w.starts, w.clip.shape[1])
    assert len(scores) == len(w.starts)
    assert res.cam.shape[1] == int(np.ceil(10 / 0.1))       # 0.1 s frames over 10 s
    assert 0 <= res.cam.min() and res.cam.max() <= 1
    assert scores[0] == pytest.approx(cam(w.windows[0:1]).spoof_score, abs=1e-4)


@needs_weights
def test_analyze_endpoint_windows(long_wav):
    from fastapi.testclient import TestClient
    from src.api.server import app

    with TestClient(app) as client:
        with open(long_wav, "rb") as f:
            d = client.post("/analyze", files={"file": ("long.wav", f, "audio/wav")}).json()
        with open(long_wav, "rb") as f:
            d2 = client.post("/analyze?explain=false",
                             files={"file": ("long.wav", f, "audio/wav")}).json()

    assert len(d["windows"]) >= 4
    assert d["duration_s"] == pytest.approx(10, abs=0.01)
    assert d["analysed_s"] == pytest.approx(10, abs=0.01)
    assert d["agreement"] in ("consistent", "mixed")
    if d["agreement"] == "mixed":
        assert d["band"] == "uncertain"
    speech = [x["raw_score"] for x in d["windows"] if not x["silent"]]
    assert d["raw_score"] == pytest.approx(sum(speech) / len(speech), abs=1e-3)
    # Grad-CAM path and batched no-grad path agree
    for a, b in zip(d["windows"], d2["windows"]):
        assert a["raw_score"] == pytest.approx(b["raw_score"], abs=1e-3)


# ── VAD + selective calibration ─────────────────────────────────────────────

def test_speech_fraction_silence_vs_tone(tmp_path):
    from src.preprocessing.preprocess import preprocess_windows, speech_fraction

    sr = 16000
    t = torch.arange(sr * 4) / sr
    tone = 0.5 * torch.sin(2 * np.pi * 220 * t)
    sig = torch.cat([tone, 1e-4 * torch.randn(sr * 6)])     # 4 s tone, 6 s near-silence
    p = tmp_path / "half.wav"
    torchaudio.save(str(p), sig.unsqueeze(0), sr)

    w = preprocess_windows(str(p))
    frac = speech_fraction(w.clip, w.starts)
    assert frac[0] > 0.9          # window over the tone
    assert frac[-1] < 0.05        # last window is all near-silence


def test_tune_selective_interior_and_target():
    from src.calibration.calibrate import Calibrator

    rng = np.random.default_rng(0)
    # overlapping classes: bonafide ~ N(+3, 3), spoof ~ N(-3, 3)
    s = np.r_[rng.normal(3, 3, 2000), rng.normal(-3, 3, 2000)]
    y = np.r_[np.zeros(2000, int), np.ones(2000, int)]
    cal = Calibrator().fit_arrays(s.tolist(), y.tolist())
    lo, hi = cal.tune_selective(s, y, target_acc=0.97)
    assert 0 < lo < 0.5 < hi < 1                     # interior, not a grid edge
    p = cal._probs(s.tolist())
    cov = (p < lo) | (p > hi)
    acc = (((p < lo) & (y == 0)) | ((p > hi) & (y == 1)))[cov].mean()
    assert acc >= 0.97


def test_load_cli_pickled_calibrator(tmp_path):
    """A pickle saved from the CLI records __main__.Calibrator; load() must still work."""
    import pickle, types
    from src.calibration import calibrate

    cal = calibrate.Calibrator().fit_arrays([5.0, 4.0, -5.0, -4.0], [0, 0, 1, 1])
    fake_main = types.ModuleType("__main__")
    fake_main.Calibrator = calibrate.Calibrator
    orig_mod = calibrate.Calibrator.__module__
    calibrate.Calibrator.__module__ = "__main__"
    real_main = sys.modules["__main__"]
    sys.modules["__main__"] = fake_main
    try:
        path = tmp_path / "c.pkl"
        path.write_bytes(pickle.dumps(cal))
    finally:
        calibrate.Calibrator.__module__ = orig_mod
        sys.modules["__main__"] = real_main
    assert b"__main__" in path.read_bytes()
    assert isinstance(calibrate.Calibrator.load(str(path)), calibrate.Calibrator)
