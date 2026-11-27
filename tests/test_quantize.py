"""
test_quantize.py — INT8 SSL-AASIST, lifespan server, shared clip scoring
"""

import importlib
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
import torchaudio

ROOT = Path(__file__).resolve().parent.parent
SSL_WEIGHTS = ROOT / "outputs/weights/SSL_AASIST_DF.pth"
SAMPLES = ROOT / "outputs/samples"
needs_ssl = pytest.mark.skipif(not SSL_WEIGHTS.exists() or not SAMPLES.exists(),
                               reason="SSL_AASIST_DF.pth (1.2 GB) or outputs/samples missing")


# ── quantize_ssl on a toy model (fast, always runs) ─────────────────────────

class _Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.ssl_model = nn.Sequential(nn.Linear(16, 32), nn.GELU(), nn.Linear(32, 8))
        self.out_layer = nn.Linear(8, 2)

    def forward(self, x):
        h = self.ssl_model(x)
        return h, self.out_layer(h)


def test_quantize_ssl_only_front_end():
    from src.model.ssl_aasist import quantize_ssl
    from torch.ao.nn.quantized.dynamic import Linear as QLinear

    torch.manual_seed(0)
    m = _Toy().eval()
    x = torch.randn(64, 16)
    with torch.no_grad():
        ref = m(x)[1]
    quantize_ssl(m)

    assert all(isinstance(l, QLinear) for l in (m.ssl_model[0], m.ssl_model[2]))
    assert type(m.out_layer) is nn.Linear          # back-end untouched (fp32, has grads)
    assert m.quantized
    with torch.no_grad():
        q = m(x)[1]
    assert torch.allclose(ref, q, atol=0.05)
    # gradient still reaches the fp32 back-end (what Grad-CAM relies on)
    m.out_layer.weight.grad = None
    m(x)[1].sum().backward()
    assert m.out_layer.weight.grad is not None


def test_server_rejects_quantize_without_ssl(monkeypatch):
    import src.api.server as server
    monkeypatch.setenv("DFD_MODEL", "aasist_l")
    monkeypatch.setenv("DFD_QUANTIZE", "1")
    try:
        with pytest.raises(ValueError, match="DFD_QUANTIZE"):
            importlib.reload(server)
    finally:
        monkeypatch.delenv("DFD_QUANTIZE")
        importlib.reload(server)


# ── shared clip scoring ─────────────────────────────────────────────────────

def test_aggregate_ignores_silent_windows():
    from src.api.clip_scoring import aggregate, MIN_SPEECH

    c = aggregate([4.0, -2.0, 10.0], [0.9, 0.5, MIN_SPEECH / 2])
    assert c.silent == [False, False, True]
    assert c.score == pytest.approx(1.0)
    # every window silent -> fall back to all windows
    assert aggregate([1.0, 3.0], [0.0, 0.0]).score == pytest.approx(2.0)


def test_window_scores_batching_invariant():
    """Chunked forward passes must equal one big batch (eval-mode BN)."""
    from infer import load_model
    from src.api.clip_scoring import window_scores

    model = load_model(ROOT / "outputs/weights/AASIST-L.pth", ROOT / "configs/aasist_L.json",
                       torch.device("cpu"))
    torch.manual_seed(0)
    w = 0.1 * torch.randn(5, 64600)
    assert np.allclose(window_scores(model, w, batch=2), window_scores(model, w, batch=8),
                       atol=1e-4)


# ── real SSL-AASIST weights (slow; skipped without the 1.2 GB checkpoint) ───

@needs_ssl
def test_int8_preserves_sign_and_order():
    from src.model.ssl_aasist import load_ssl_aasist, quantize_ssl
    from src.preprocessing.preprocess import preprocess
    from src.explainability.gradcam import GradCAM

    files = sorted(SAMPLES.glob("*.wav"))
    x = torch.cat([preprocess(str(f), mode="infer") for f in files])

    model = load_ssl_aasist(SSL_WEIGHTS)
    with torch.inference_mode():
        _, lg = model(x)
    fp32 = (lg[:, 1] - lg[:, 0]).numpy()

    quantize_ssl(model)
    with torch.inference_mode():
        _, lg = model(x)
    int8 = (lg[:, 1] - lg[:, 0]).numpy()

    assert np.array_equal(np.sign(fp32), np.sign(int8)), (fp32, int8)
    assert np.array_equal(np.argsort(fp32), np.argsort(int8)), (fp32, int8)
    assert np.abs(fp32 - int8).max() < 1.0, (fp32, int8)

    r = GradCAM(model)(x[:1])                       # back-end fp32 -> CAM still works
    assert np.isfinite(r.cam).all() and r.cam.max() > 0


@needs_ssl
def test_analyze_endpoint_ssl_int8(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient
    import src.api.server as server

    f = sorted(SAMPLES.glob("bonafide*.wav"))[0]
    monkeypatch.setenv("DFD_MODEL", "ssl")
    monkeypatch.setenv("DFD_QUANTIZE", "1")
    try:
        importlib.reload(server)
        with TestClient(server.app) as client:          # lifespan loads the INT8 model
            assert client.get("/health").json()["model_loaded"]
            with open(f, "rb") as fh:
                d = client.post("/analyze", files={"file": (f.name, fh, "audio/wav")}).json()
        assert d["model"] == "ssl" and d["quantized"] is True
        assert d["band"] in ("authentic", "uncertain", "spoof")
        assert d["raw_score"] > 0                           # bonafide sample
        assert d["heatmap_png"]                             # Grad-CAM on INT8 model
    finally:
        monkeypatch.delenv("DFD_MODEL")
        monkeypatch.delenv("DFD_QUANTIZE")
        importlib.reload(server)
