"""
server.py — FastAPI inference server (Phase 6)
===============================================
Run: uvicorn src.api.server:app --reload --port 8000

POST /analyze   multipart: audio file
  -> JSON: band, prob_spoof, confidence, raw_score, segments, model_version
"""

import io
import tempfile
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import torch

ROOT = Path(__file__).parent.parent.parent
WEIGHTS = ROOT / "outputs/weights/best_model.pth"
CONFIG  = ROOT / "configs/aasist_L.json"
CAL     = ROOT / "outputs/weights/calibrator.pkl"

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

@app.on_event("startup")
def load_resources():
    global _model, _calibrator
    from infer import load_model
    from src.calibration.calibrate import Calibrator

    device = torch.device("cpu")
    if WEIGHTS.exists() and CONFIG.exists():
        _model = load_model(WEIGHTS, CONFIG, device)
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


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    """
    Upload an audio file, receive a deepfake verdict.

    Returns:
        band          : "authentic" | "uncertain" | "spoof"
        prob_spoof    : calibrated probability (or None if not calibrated)
        raw_score     : log-likelihood ratio from model
        label         : "bonafide" | "spoof" (binary, no abstain)
        model_version : weight file name
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
        from src.preprocessing.preprocess import preprocess
        import torch.nn.functional as F

        device = torch.device("cpu")
        tensor = preprocess(tmp_path, mode="infer").to(device)  # already [1, 64600]

        with torch.no_grad():
            _, output = _model(tensor)   # AASIST returns (last_hidden, logits)

        raw_score  = (output[0, 1] - output[0, 0]).item()   # bonafide log-likelihood ratio
        prob_spoof = F.softmax(output, dim=1)[0, 0].item()  # idx 0 = spoof

        result = {
            "raw_score":     round(raw_score, 4),
            "prob_spoof":    round(prob_spoof, 4),
            "label":         "spoof" if raw_score > 0 else "bonafide",
            "model_version": WEIGHTS.name,
        }

        # Apply calibration + 3-band verdict if available
        if _calibrator is not None:
            verdict = _calibrator.predict(raw_score)
            result["band"]       = verdict.band
            result["confidence"] = verdict.confidence
            result["prob_spoof"] = verdict.prob_spoof
        else:
            result["band"]       = result["label"]
            result["confidence"] = "uncalibrated"

        return result

    finally:
        Path(tmp_path).unlink(missing_ok=True)


# Serve frontend if built
ui_dir = ROOT / "src/ui"
if (ui_dir / "index.html").exists():
    app.mount("/", StaticFiles(directory=str(ui_dir), html=True), name="ui")
