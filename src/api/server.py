"""
server.py — FastAPI inference server (Phase 6)
===============================================
Run: python -m uvicorn src.api.server:app --port 8000

Model: env DFD_MODEL = "aasist_l" (default, 85K params, ~0.5 s/window CPU)
                     | "ssl"      (wav2vec2 XLS-R + AASIST, 317M params)
      env DFD_QUANTIZE = "1"  (ssl only) dynamic INT8 on the wav2vec2 Linears.
          Measured on this 8 GB CPU laptop: ~1.7 GB private RAM vs ~3.1 GB fp32,
          ~0.9 s vs ~1.4 s per 4 s window; sample scores move < 0.35 LLR.
Each model has its own calibrator; scores are NOT on the same scale, so a
calibrator fitted for one model must never be used with the other.
Windowed clip scoring lives in src/api/clip_scoring.py (shared with
scripts/refit_windowed_calibrator.py so calibration sees the same scores).

POST /analyze   multipart: audio file   (?explain=false skips Grad-CAM)
  -> JSON: band, prob_spoof, confidence, raw_score, agreement, windows[],
           duration_s, analysed_s, truncated, thresholds, model_version, heatmap_png
"""

import base64
import os
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import torch

ROOT = Path(__file__).parent.parent.parent
MODEL = os.environ.get("DFD_MODEL", "aasist_l")
if MODEL not in ("aasist_l", "ssl"):
    raise ValueError(f"DFD_MODEL must be 'aasist_l' or 'ssl', got {MODEL!r}")
QUANTIZE = os.environ.get("DFD_QUANTIZE", "0").lower() in ("1", "true", "yes")
if QUANTIZE and MODEL != "ssl":
    raise ValueError("DFD_QUANTIZE=1 is only supported with DFD_MODEL=ssl "
                     "(AASIST-L is 85K params; nothing to gain)")

if MODEL == "ssl":
    WEIGHTS = ROOT / "outputs/weights/SSL_AASIST_DF.pth"
    # INT8 scores differ slightly from fp32: prefer a calibrator refit on the
    # quantized model (refit_windowed_calibrator.py --model ssl --quantize).
    CAL = ROOT / os.environ.get("DFD_CALIBRATOR", "outputs/weights/calibrator_ssl_int8.pkl"
                                if QUANTIZE else "outputs/weights/calibrator_ssl.pkl")
    if not CAL.exists():
        CAL = ROOT / "outputs/weights/calibrator_ssl.pkl"
else:
    DEFAULT_WEIGHTS = ROOT / "outputs/weights/best_model.pth"
    FALLBACK_WEIGHTS = ROOT / "outputs/weights/AASIST-L.pth"
    WEIGHTS = DEFAULT_WEIGHTS if DEFAULT_WEIGHTS.exists() else FALLBACK_WEIGHTS
    # Deployment calibrator: Platt fitted on 2019 LA dev + half the 2021 DF sample
    # (speaker-disjoint from the DF half it is reported on), selective band at 98%
    # accuracy — see scripts/domain_calibration.py. calibrator.pkl (LA dev only) is
    # the in-domain paper baseline and is overconfident on unseen speakers.
    # Override with env DFD_CALIBRATOR=outputs/weights/calibrator.pkl
    CAL = ROOT / os.environ.get("DFD_CALIBRATOR", "outputs/weights/calibrator_mixed.pkl")
    if not CAL.exists():
        CAL = ROOT / "outputs/weights/calibrator.pkl"
CONFIG = ROOT / "configs/aasist_L.json"

# ── Startup: load model once, keep in memory ─────────────────────────────────
_model = None
_calibrator = None
_gradcam = None


def load_resources():
    global _model, _calibrator, _gradcam
    from infer import load_model
    from src.calibration.calibrate import Calibrator
    from src.explainability.gradcam import GradCAM

    device = torch.device("cpu")
    if WEIGHTS.exists():
        if MODEL == "ssl":
            from src.model.ssl_aasist import load_ssl_aasist
            _model = load_ssl_aasist(WEIGHTS, device, quantize=QUANTIZE)
        else:
            _model = load_model(WEIGHTS, CONFIG, device)
        # Both models expose .encoder; quantize_ssl keeps the back-end fp32,
        # so Grad-CAM gradients still flow on the INT8 model.
        _gradcam = GradCAM(_model)
        print(f"Model ({MODEL}{', int8' if QUANTIZE else ''}) loaded from {WEIGHTS}")
    else:
        print(f"WARN: weights not found at {WEIGHTS} — /analyze will fail.")

    if CAL.exists():
        _calibrator = Calibrator.load(str(CAL))
        print("Calibrator loaded.")
    else:
        print(f"WARN: calibrator {CAL.name} not found — uncalibrated verdicts.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_resources()
    yield


app = FastAPI(title="Audio Deepfake Detector", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],    # tighten in production
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _model is not None}


def _band(prob: float) -> str:
    lo, hi = (_calibrator.lower, _calibrator.upper) if _calibrator else (0.5, 0.5)
    return "authentic" if prob < lo else "spoof" if prob > hi else "uncertain"


def _prob(raw_score: float) -> float:
    if _calibrator is not None:
        return _calibrator.predict(raw_score).prob_spoof
    # Uncalibrated fallback: softmax over [spoof, bonafide] = sigmoid(-raw_score)
    return torch.sigmoid(torch.tensor(-raw_score)).item()


@app.post("/analyze")
async def analyze(file: UploadFile = File(...),
                  explain: bool = Query(True, description="Include Grad-CAM heatmap")):
    """
    Upload an audio file, receive a deepfake verdict over the WHOLE clip.

    The clip is split into 4.04 s windows with 50% overlap (preprocess_windows).
    Each window is scored; silent windows are skipped; the clip verdict uses
    the mean bonafide LLR of the remaining windows. If windows confidently
    disagree (some AUTHENTIC, some SPOOF) the verdict is forced to UNCERTAIN.

    Returns:
        band          : "authentic" | "uncertain" | "spoof"
        prob_spoof    : calibrated P(spoof) of the mean score
        confidence    : "high" | "low" | "uncalibrated"
        raw_score     : mean bonafide LLR over speech windows
        agreement     : "consistent" | "mixed"
        windows       : [{start_s, end_s, raw_score, prob_spoof, band, speech, silent}]
        duration_s, analysed_s, truncated
        thresholds    : [lower, upper] (if calibrated)
        model_version : weight file name
        heatmap_png   : base64 PNG of Grad-CAM over the clip (None if explain=false)

    No bare binary label is returned: the band is the verdict (CLAUDE.md rule).
    NOTE: calibrator_mixed.pkl / calibrator.pkl were fitted on single 4 s
    windows, not window means; scripts/refit_windowed_calibrator.py refits
    on clip scores from src/api/clip_scoring.py (the code used below).
    """
    if _model is None:
        raise HTTPException(503, "Model not loaded. Run setup_env.sh first.")

    # ── Save upload to temp file ──────────────────────────────────────────────
    suffix = Path(file.filename).suffix or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        contents = await file.read()
        tmp.write(contents)
        tmp_path = tmp.name

    try:
        from src.preprocessing.preprocess import (preprocess_windows, speech_fraction,
                                                  TARGET_SR, TARGET_LEN)
        from src.api.clip_scoring import window_scores, aggregate
        from src.explainability.gradcam import render_png, SPECTRAL_LABEL

        try:
            w = preprocess_windows(tmp_path)
        except Exception as e:
            raise HTTPException(400, f"Could not decode audio: {e}")

        heatmap_png = None
        if explain:
            cam, spoof_scores = _gradcam.clip(w.windows, w.starts, w.clip.shape[1])
            scores = [-s for s in spoof_scores]       # bonafide LLR = logits[1] - logits[0]
            analysed = w.clip[:, :w.starts[-1] + TARGET_LEN]
            label = "feature channel" if MODEL == "ssl" else SPECTRAL_LABEL
            heatmap_png = base64.b64encode(render_png(analysed, cam, label)).decode("ascii")
        else:
            scores = window_scores(_model, w.windows)

        clip = aggregate(scores, speech_fraction(w.clip, w.starts))
        windows = []
        for s, score, sf, silent in zip(w.starts, clip.window_scores, clip.speech, clip.silent):
            prob = _prob(score)
            windows.append({
                "start_s":    round(s / TARGET_SR, 3),
                "end_s":      round(min(s + TARGET_LEN, w.clip.shape[1]) / TARGET_SR, 3),
                "raw_score":  round(score, 4),
                "prob_spoof": round(prob, 4),
                "band":       _band(prob),
                "speech":     round(sf, 3),
                "silent":     silent,
            })

        speech = [x for x in windows if not x["silent"]] or windows
        raw_score = clip.score
        prob_spoof = _prob(raw_score)
        bands = {x["band"] for x in speech}
        mixed = {"authentic", "spoof"} <= bands

        if _calibrator is None:
            band, confidence = ("authentic" if raw_score > 0 else "spoof"), "uncalibrated"
        elif mixed:
            band, confidence = "uncertain", "low"
        else:
            band = _band(prob_spoof)
            confidence = "low" if band == "uncertain" else "high"

        result = {
            "band":          band,
            "prob_spoof":    round(prob_spoof, 4),
            "confidence":    confidence,
            "raw_score":     round(raw_score, 4),
            "agreement":     "mixed" if mixed else "consistent",
            "windows":       windows,
            "duration_s":    round(w.clip.shape[1] / TARGET_SR, 3),
            "analysed_s":    windows[-1]["end_s"],
            "truncated":     w.truncated,
            "model_version": WEIGHTS.name,
            "model":         MODEL,
            "quantized":     QUANTIZE,
            "heatmap_png":   heatmap_png,
        }
        if _calibrator is not None:
            result["thresholds"] = [_calibrator.lower, _calibrator.upper]
            result["calibrator"] = CAL.name
        return result

    finally:
        Path(tmp_path).unlink(missing_ok=True)


# Serve frontend if built
ui_dir = ROOT / "src/ui"
if (ui_dir / "index.html").exists():
    app.mount("/", StaticFiles(directory=str(ui_dir), html=True), name="ui")
