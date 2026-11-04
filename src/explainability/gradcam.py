"""
gradcam.py — Grad-CAM for AASIST-L (Phase 6)
=============================================
Target layer: model.encoder output e  (#bs, #filt, #spec, #seq)
  - #spec = spectral nodes (pooled SincNet filter bands, low → high freq)
  - #seq  = time frames spanning the 64600-sample input

Target score: spoof LLR = logits[:, 0] - logits[:, 1]
  (AASIST class 0 = spoof, 1 = bonafide; negative of the bonafide score
   used in infer.py / eval.py). So the map shows where the model found
   evidence of SPOOF, whatever the final verdict.

Usage:
    from src.explainability.gradcam import GradCAM, render_png
    cam = GradCAM(model)
    result = cam(waveform)             # waveform [1, 64600] from preprocess()
    png = render_png(waveform, result) # bytes

    # whole clip, from preprocess_windows():
    result, window_scores = cam.clip(w.windows, w.starts, w.clip.shape[1])
    png = render_png(w.clip, result)
"""

import io
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

SAMPLE_RATE = 16000


@dataclass
class CAMResult:
    cam: np.ndarray           # [spec, seq] in [0, 1]
    time_profile: np.ndarray  # [seq] in [0, 1], mean over spec
    spoof_score: float        # logits[0] - logits[1]


class GradCAM:
    """Grad-CAM on AASIST's encoder output. Model must be in eval mode."""

    def __init__(self, model: torch.nn.Module):
        self.model = model
        self._act = None
        self._grad = None
        model.encoder.register_forward_hook(self._save)

    def _save(self, module, inp, out):
        self._act = out
        if out.requires_grad:
            out.register_hook(lambda g: setattr(self, "_grad", g))

    def raw(self, waveform: torch.Tensor) -> tuple[torch.Tensor, float]:
        """Unnormalised CAM [S, T] and spoof LLR for one [1, TARGET_LEN] window."""
        self.model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            _, logits = self.model(waveform)
            spoof_score = logits[0, 0] - logits[0, 1]
            spoof_score.backward()

        act, grad = self._act.detach()[0], self._grad.detach()[0]   # [C, S, T]
        weights = grad.mean(dim=(1, 2), keepdim=True)               # [C, 1, 1]
        cam = F.relu((weights * act).sum(dim=0))                    # [S, T]

        self._act = self._grad = None
        return cam, spoof_score.item()

    def __call__(self, waveform: torch.Tensor) -> CAMResult:
        """waveform: [1, T] float32 (output of preprocess(mode='infer'))."""
        cam, score = self.raw(waveform)
        return _finish(cam, score)

    def clip(self, windows: torch.Tensor, starts: list[int],
             n_samples: int, frame_s: float = 0.1) -> tuple[CAMResult, list[float]]:
        """
        Grad-CAM over a whole clip from overlapping windows.

        windows : [N, L] from preprocess_windows()
        starts  : window starts in samples
        Each window's raw CAM is resampled onto a shared time grid
        (frame_s seconds per frame) and overlaps are averaged. Normalised
        once at the end, so a weak window stays weak next to a strong one.
        Returns (stitched result, per-window spoof LLRs).
        """
        L = windows.shape[1]
        n_frames = int(np.ceil(n_samples / SAMPLE_RATE / frame_s))
        acc = cnt = None
        scores = []
        for w, s in zip(windows, starts):
            cam, score = self.raw(w.unsqueeze(0))
            scores.append(score)
            a = int(round(s / SAMPLE_RATE / frame_s))
            b = min(int(round((s + L) / SAMPLE_RATE / frame_s)), n_frames)
            if acc is None:
                acc = torch.zeros(cam.shape[0], n_frames)
                cnt = torch.zeros(n_frames)
            acc[:, a:b] += F.interpolate(cam[None], size=b - a, mode="linear",
                                         align_corners=False)[0]
            cnt[a:b] += 1
        stitched = acc / cnt.clamp(min=1)
        return _finish(stitched, float(np.mean(scores))), scores


def _finish(cam: torch.Tensor, score: float) -> CAMResult:
    cam = cam / cam.max() if cam.max() > 0 else cam
    prof = cam.mean(dim=0)
    prof = prof / prof.max() if prof.max() > 0 else prof
    return CAMResult(cam=cam.numpy(), time_profile=prof.numpy(), spoof_score=score)


def render_png(waveform: torch.Tensor, result: CAMResult) -> bytes:
    """
    Two panels, shared time axis (seconds):
      top    — input spectrogram with spoof-evidence time profile overlaid
      bottom — 2D Grad-CAM (spectral node × time)
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    wav = waveform.detach().reshape(-1)
    dur = len(wav) / SAMPLE_RATE

    spec = torch.stft(wav, n_fft=512, hop_length=160,
                      window=torch.hann_window(512), return_complex=True)
    spec_db = 20 * torch.log10(spec.abs() + 1e-6).numpy()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(max(8, min(16, 2 * dur)), 4.5), sharex=True,
                                   gridspec_kw={"height_ratios": [3, 2]})
    ax1.imshow(spec_db, origin="lower", aspect="auto", cmap="gray_r",
               extent=[0, dur, 0, SAMPLE_RATE / 2000],
               vmin=spec_db.max() - 80, vmax=spec_db.max())
    ax1.set_ylabel("kHz")
    t = np.linspace(0, dur, len(result.time_profile))
    ax1b = ax1.twinx()
    ax1b.fill_between(t, result.time_profile, step="mid", alpha=0.35, color="tab:red")
    ax1b.set_ylim(0, 1.05)
    ax1b.set_yticks([])
    ax1.set_title(f"Spoof evidence (Grad-CAM)   mean spoof LLR = {result.spoof_score:+.2f}",
                  fontsize=10)

    ax2.imshow(result.cam, origin="lower", aspect="auto", cmap="inferno",
               extent=[0, dur, 0, result.cam.shape[0]], vmin=0, vmax=1)
    ax2.set_ylabel("spectral node\n(low → high)")
    ax2.set_xlabel("time (s)")

    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100)
    plt.close(fig)
    return buf.getvalue()
