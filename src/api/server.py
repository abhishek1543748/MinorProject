"""
server.py — FastAPI inference server (Phase 6)
===============================================
Run: uvicorn src.api.server:app --reload --port 8000

POST /analyze   multipart: audio file   (?explain=false skips Grad-CAM)
  -> JSON: band, prob_spoof, confidence, raw_score, agreement, windows[],
           duration_s, analysed_s, truncated, thresholds, model_version, heatmap_png
"""

import base64
import os
import tempfile
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import torch

ROOT = Path(__file__).parent.parent.parent
DEFAULT_WEIGHTS = ROOT / "outputs/weights/best_model.pth"
FALLBACK_WEIGHTS = ROOT / "outputs/weights/AASIST-L.pth"
WEIGHTS = DEFAULT_WEIGHTS if DEFAULT_WEIGHTS.exists() else FALLBACK_WEIGHTS
CONFIG  = ROOT / "configs/aasist_L.json"
# Deployment calibrator: Platt fitted on 2019 LA dev + half the 2021 DF sample
# (speaker-disjoint from the DF half it is reported on), selective band at 98%
# accuracy — see scripts/domain_calibration.py. calibrator.pkl (LA dev only) is
# the in-domain paper baseline and is overconfident on unseen speakers.
# Override with env DFD_CALIBRATOR=outputs/weights/calibrator.pkl
CAL = ROOT / os.environ.get("DFD_CALIBRATOR", "outputs/weights/calibrator_mixed.pkl")
if not CAL.exists():
    CAL = ROOT / "outputs/weights/calibrator.pkl"

app = FastAPI(title="Audio Deepfake Detector", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],    # tighten in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Startup: load model once, keep in memory ─────────────────────────────────
_model = None
_calibrator = None
_gradcam = None

@app.on_event("startup")
def load_resources():
    global _model, _calibrator, _gradcam
    from infer import load_model
    from src.calibration.calibrate import Calibrator
    from src.explainability.gradcam import GradCAM

    device = torch.device("cpu")
    if WEIGHTS.exists() and CONFIG.exists():
        _model = load_model(WEIGHTS, CONFIG, device)
        _gradcam = GradCAM(_model)
        print(f"Model loaded from {WEIGHTS}")
    else:
        print(f"WARN: weights not found at {WEIGHTS} — /analyze will fail.")

    if CAL.exists():
        _calibrator = Calibrator.load(str(CAL))
        print("Calibrator loaded.")
    else:
        print("WARN: calibrator not found — raw scores will be used.")


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _model is not None}


MIN_SPEECH = 0.2     # window with < 20% speech frames (energy VAD) = silence, ignored


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
    NOTE: the calibrator was fitted on single 4 s windows, not window means.
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
        from src.explainability.gradcam import render_png

        try:
            w = preprocess_windows(tmp_path)
        except Exception as e:
            raise HTTPException(400, f"Could not decode audio: {e}")

        heatmap_png = None
        if explain:
            cam, spoof_scores = _gradcam.clip(w.windows, w.starts, w.clip.shape[1])
            scores = [-s for s in spoof_scores]       # bonafide LLR = logits[1] - logits[0]
            analysed = w.clip[:, :w.starts[-1] + TARGET_LEN]
            heatmap_png = base64.b64encode(render_png(analysed, cam)).decode("ascii")
        else:
            with torch.no_grad():
                _, output = _model(w.windows)         # [N, 2]
            scores = (output[:, 1] - output[:, 0]).tolist()

        speech_frac = speech_fraction(w.clip, w.starts)
        windows = []
        for s, score, sf in zip(w.starts, scores, speech_frac):
            prob = _prob(score)
            windows.append({
                "start_s":    round(s / TARGET_SR, 3),
                "end_s":      round(min(s + TARGET_LEN, w.clip.shape[1]) / TARGET_SR, 3),
                "raw_score":  round(score, 4),
                "prob_spoof": round(prob, 4),
                "band":       _band(prob),
                "speech":     round(sf, 3),
                "silent":     sf < MIN_SPEECH,
            })

        speech = [x for x in windows if not x["silent"]] or windows
        raw_score = float(sum(x["raw_score"] for x in speech) / len(speech))
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
