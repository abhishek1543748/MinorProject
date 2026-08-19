"""
infer.py — Single-file inference entry point
=============================================
Usage:
    python infer.py --input path/to/audio.wav
    python infer.py --input path/to/audio.wav --weights outputs/weights/AASIST-L.pth

Returns: JSON with label (bonafide/spoof), raw score, and probability.

Phase 0: this file is the ONLY deliverable needed to move to Phase 1.
"""

import sys
import json
import argparse
import torch
import torch.nn.functional as F
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
DEFAULT_WEIGHTS = ROOT / "outputs/weights/AASIST-L.pth"
DEFAULT_CONFIG  = ROOT / "configs/aasist_L.json"


def load_model(weights_path: Path, config_path: Path, device: torch.device):
    """Load AASIST-L model with pretrained weights."""
    import json
    from src.model.AASIST import Model

    with open(config_path) as f:
        config = json.load(f)

    model = Model(config["model_config"]).to(device)
    state = torch.load(weights_path, map_location=device)

    # Handle different checkpoint formats
    if "model" in state:
        state = state["model"]
    if "state_dict" in state:
        state = state["state_dict"]

    model.load_state_dict(state)
    model.eval()
    return model


def infer_single(audio_path: str, weights_path: Path, config_path: Path) -> dict:
    """
    Run inference on a single audio file.

    Returns dict with:
        label    : "bonafide" or "spoof"
        score    : raw log-likelihood ratio (float)
        prob_spoof: probability of being spoof (0-1, uncalibrated)
    """
    from src.preprocessing.preprocess import preprocess

    device = torch.device("cpu")   # CPU-first for local deployment

    # ── Preprocess ───────────────────────────────────────────────────────────
    tensor = preprocess(audio_path, mode="infer")  # [1, 64600] == [batch=1, time]
    tensor = tensor.to(device)                     # Model.forward does its own unsqueeze(1)

    # ── Load model ───────────────────────────────────────────────────────────
    model = load_model(weights_path, config_path, device)

    # ── Forward pass ─────────────────────────────────────────────────────────
    with torch.no_grad():
        _, output = model(tensor)   # AASIST.Model returns (last_hidden, logits[1,2])

    # Official AASIST convention: out_layer index 0 = spoof, index 1 = bonafide
    # (verified empirically on 6 ASVspoof19 LA dev files: bonafide files score
    # +11 to +13 on (idx1 - idx0), spoof files score -10 to -15 — this is the
    # opposite of what a naive "0=bonafide,1=spoof" reading would assume).
    score = (output[0, 1] - output[0, 0]).item()   # bonafide log-likelihood ratio
    probs = F.softmax(output, dim=1)[0]
    prob_bonafide = probs[1].item()

    label = "bonafide" if score > 0 else "spoof"

    return {
        "label": label,
        "score": round(score, 4),
        "prob_bonafide": round(prob_bonafide, 4),
        "prob_spoof": round(1 - prob_bonafide, 4),
        "weights": str(weights_path),
        "audio": str(audio_path),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Audio deepfake detector — single file inference"
    )
    parser.add_argument("--input",   required=True,  help="Path to audio file")
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS), help="Path to .pth weights")
    parser.add_argument("--config",  default=str(DEFAULT_CONFIG),  help="Path to model config JSON")
    args = parser.parse_args()

    audio_path   = Path(args.input)
    weights_path = Path(args.weights)
    config_path  = Path(args.config)

    # ── Validation ───────────────────────────────────────────────────────────
    if not audio_path.exists():
        print(f"ERROR: Audio file not found: {audio_path}", file=sys.stderr)
        sys.exit(1)
    if not weights_path.exists():
        print(f"ERROR: Weights not found: {weights_path}", file=sys.stderr)
        print("Run: bash scripts/setup_env.sh to copy the pretrained weights.", file=sys.stderr)
        sys.exit(1)
    if not config_path.exists():
        print(f"ERROR: Config not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    # ── Run ──────────────────────────────────────────────────────────────────
    result = infer_single(str(audio_path), weights_path, config_path)

    # Pretty print
    print(json.dumps(result, indent=2))

    # Exit code: 0 = bonafide, 1 = spoof
    sys.exit(0 if result["label"] == "bonafide" else 1)


if __name__ == "__main__":
    main()
