"""
clip_scoring.py — whole-clip scoring shared by the server and calibration
=========================================================================
One implementation of "sliding windows -> per-window bonafide LLR -> energy
VAD -> mean over speech windows", imported by BOTH src/api/server.py and
scripts/refit_windowed_calibrator.py. A calibrator fitted on these clip
scores therefore sees exactly the score the server feeds it (same rule as
the GOLDEN RULE for preprocessing: one implementation, imported everywhere).

Score convention: bonafide LLR = logits[:, 1] - logits[:, 0]
(AASIST class 0 = spoof, 1 = bonafide).
"""

from dataclasses import dataclass

import torch

from src.preprocessing.preprocess import preprocess_windows, speech_fraction

MIN_SPEECH = 0.2     # window with < 20% speech frames (energy VAD) = silence, ignored
BATCH = 8            # windows per forward pass (bounds SSL activation memory)


@dataclass
class ClipScore:
    window_scores: list[float]   # bonafide LLR per window
    speech: list[float]          # speech fraction per window
    silent: list[bool]           # speech < MIN_SPEECH
    score: float                 # mean LLR over non-silent windows (all if every window silent)


@torch.inference_mode()
def window_scores(model: torch.nn.Module, windows: torch.Tensor,
                  batch: int = BATCH) -> list[float]:
    """Bonafide LLR for each row of windows [N, TARGET_LEN] (no grad)."""
    device = next(model.parameters()).device
    out = []
    for i in range(0, windows.shape[0], batch):
        _, logits = model(windows[i:i + batch].to(device))
        out += (logits[:, 1] - logits[:, 0]).tolist()
    return out


def aggregate(scores: list[float], speech: list[float]) -> ClipScore:
    """Clip score = mean LLR over speech windows (all windows if none pass VAD)."""
    silent = [sf < MIN_SPEECH for sf in speech]
    keep = [s for s, q in zip(scores, silent) if not q] or list(scores)
    return ClipScore(list(scores), list(speech), silent, float(sum(keep) / len(keep)))


def score_file(model: torch.nn.Module, path: str) -> ClipScore:
    """Score a whole audio file exactly as POST /analyze (explain=false) does."""
    w = preprocess_windows(path)
    return aggregate(window_scores(model, w.windows), speech_fraction(w.clip, w.starts))
