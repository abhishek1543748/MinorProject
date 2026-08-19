# Audio Deepfake Detection — Project Brief for Claude Code

## What this project is
A university research project combining signal/speech processing with machine learning
to detect AI-generated (deepfake) audio. The goal is a product-grade, fully-local
voice spoof detector — not just an academic exercise.

Supervised by a professor. Small team. Ideation complete. Now building.

---

## Core architecture decisions (already made — do not change without asking)

| Decision | Choice | Why |
|---|---|---|
| Backbone | AASIST-L (85K params) | Lightweight, runs on CPU, official pretrained weights available |
| Front-end | Raw waveform (SincNet inside AASIST-L) | No separate LFCC/FFT stage needed for MVP |
| Augmentation | RawBoost (conv+impulsive) | Training-only, no external data, free robustness |
| Training set | ASVspoof 2019 LA | Standard benchmark, labelled real+fake |
| Stress test | ASVspoof 2021 DF | Codec-degraded, tests generalization |
| Contribution | Calibrated 3-band abstain verdict | No commercial tool publishes this |
| Deployment | Fully local, no API key, no internet at inference | 330KB weight file is the entire model |

**Do NOT use the HuggingFace AASIST port — it is unmaintained.**
**Use the official repo: github.com/clovaai/aasist**

---

## The GOLDEN RULE (highest priority constraint in the entire codebase)
`src/preprocessing/preprocess.py` must be used IDENTICALLY in both
`train.py` and `infer.py`. Never duplicate preprocessing logic.
Any mismatch between training and inference preprocessing silently
destroys performance. Build it once, import it everywhere.

---

## Repository structure

```
deepfake_detector/
├── CLAUDE.md                  ← you are here
├── README.md
├── requirements.txt
├── configs/
│   ├── aasist_L.json          ← AASIST-L model config (from clovaai repo)
│   └── train_config.json      ← training hyperparameters
├── data/
│   ├── raw/                   ← ASVspoof 2019 LA (downloaded separately)
│   └── processed/             ← cached tensors (optional)
├── src/
│   ├── preprocessing/
│   │   └── preprocess.py      ← GOLDEN RULE MODULE — shared by train + infer
│   ├── features/
│   │   └── rawboost.py        ← RawBoost augmentation (training only)
│   ├── model/
│   │   ├── AASIST.py          ← copied from clovaai/aasist/models/AASIST.py
│   │   └── loss.py            ← LMCL + CE loss
│   ├── calibration/
│   │   └── calibrate.py       ← Platt scaling + 3-band thresholds
│   ├── explainability/
│   │   └── gradcam.py         ← Grad-CAM on AASIST-L final layer
│   ├── api/
│   │   └── server.py          ← FastAPI POST /analyze endpoint
│   └── ui/
│       └── index.html         ← simple upload + result card
├── scripts/
│   ├── download_data.sh       ← instructions to get ASVspoof 2019 LA
│   └── setup_env.sh           ← conda env + pip installs
├── tests/
│   └── test_preprocess.py     ← CRITICAL: asserts train==infer tensor equality
├── train.py                   ← training entry point
├── infer.py                   ← single-file inference entry point
├── eval.py                    ← batch EER + tDCF scoring
└── outputs/
    ├── weights/               ← model checkpoints (.pth files)
    ├── scores/                ← per-file score CSVs
    └── plots/                 ← EER curves, score histograms
```

---

## Phase execution order

### PHASE 0 — Setup & first inference (Week 1) ← START HERE
**Goal:** one audio file in, real/fake score out, on this machine.

Steps:
1. `bash scripts/setup_env.sh` — create conda env, install deps
2. `git clone https://github.com/clovaai/aasist` into a temp folder
3. Copy `models/AASIST.py` → `src/model/AASIST.py`
4. Copy `models/weights/AASIST-L.pth` → `outputs/weights/AASIST-L.pth`
5. Copy `config/AASIST-L.conf` → `configs/aasist_L.json`
6. Run `python infer.py --input test.wav` → should print a score
7. Test on 3 real files and 3 known-fake files — confirm direction of scores

**Success criterion:** scores < 0 for bonafide, > 0 for spoof (or vice versa —
just confirm consistent direction before moving on).

### PHASE 1 — Signal processing core (Weeks 2–3)
**Goal:** build `src/preprocessing/preprocess.py` and make `tests/test_preprocess.py` pass.

The preprocessing module must:
- Accept any audio file path (wav, flac, mp3, ogg)
- Decode via torchaudio.load()
- Resample to 16000 Hz using polyphase resampler
- Downmix to mono (average channels)
- Peak-normalize to [-1, 1]
- Pad or crop to exactly 64600 samples (4.025 seconds at 16kHz)
- Return a torch.Tensor of shape [1, 64600] dtype float32
- Accept a `mode` parameter: 'train' applies RawBoost, 'infer' does not

The test must assert:
```python
train_tensor = preprocess(path, mode='infer')  # no augmentation
infer_tensor = preprocess(path, mode='infer')
assert torch.allclose(train_tensor, infer_tensor)
```

### PHASE 2 — Training pipeline (Weeks 3–5)
**Goal:** train AASIST-L on ASVspoof 2019 LA, save best checkpoint.

Key decisions:
- Loss: LMCL (large margin cosine loss) preferred over CE
- Optimizer: Adam, lr=0.0001, weight_decay=1e-4
- Epochs: 100, save best by dev EER
- Batch size: 24 (or 12 if RAM-constrained)
- RawBoost: type 1+2 (convolutive + impulsive) for LA condition
- Data: use official train/dev/eval splits from ASVspoof 2019 LA protocol files

### PHASE 3 — Evaluation & baseline EER (Weeks 5–6)
**Goal:** EER on ASVspoof 2019 LA eval set.
Target: ~0.83% EER matching the AASIST-L paper.
If >2%: bug is in preprocessing or data loading, NOT the model.
Output: `outputs/scores/asvspoof19_LA_eval_scores.csv`

### PHASE 4 — Cross-domain stress test (Weeks 6–7)
**Goal:** run frozen 2019-trained model on ASVspoof 2021 DF eval.
Expected: EER jumps from ~0.8% to ~15%. That IS the finding.
Do NOT retrain. Do NOT try to fix it. Measure and report it.
Output: `outputs/scores/asvspoof21_DF_eval_scores.csv`

### PHASE 5 — Calibration + abstain band (Weeks 7–8)
**Goal:** Platt calibration + 3-band verdict. This is the research contribution.
Steps:
1. Collect raw scores on dev set
2. Fit sklearn LogisticRegression on (score, label) pairs
3. Set lower threshold at 0.35, upper at 0.65 (tune on dev EER)
4. Report abstain% in-domain vs cross-domain
5. That comparison = the paper-quality finding

### PHASE 6 — Explainability + web UI (Weeks 8–10)
**Goal:** Grad-CAM overlay + FastAPI endpoint + simple web page.
Stack: FastAPI + uvicorn, plain HTML or React.
Key endpoint: POST /analyze (multipart audio) → JSON with band, prob, heatmap_png.
Export model to ONNX for 2-5x faster CPU inference if needed.

### PHASE 7 — Codec robustness study (Weeks 10–11)
**Goal:** EER vs bitrate table across Opus/AMR/MP3.
Tool: ffmpeg for re-encoding.
Experiment: original vs codec-degraded vs stationary-noise RawBoost retrain.
This is the Codecfake open problem (2024/2025 papers).

### PHASE 8 — Write-up + professor review (Weeks 11–12)
**Goal:** written report + live demo + 5-min pitch.
Upper-bound experiment: wav2vec2+AASIST (run once, report vs AASIST-L).

---

## Key sources (exact URLs)

| Resource | URL |
|---|---|
| AASIST code + weights | https://github.com/clovaai/aasist |
| RawBoost code | https://github.com/TakHemlata/RawBoost |
| ASVspoof 2019 LA dataset | https://datashare.ed.ac.uk/handle/10283/3336 |
| ASVspoof 2021 DF eval | https://zenodo.org/record/4835108 |
| AASIST paper | https://arxiv.org/pdf/2110.01200 |
| RawBoost paper | https://arxiv.org/pdf/2111.04433 |
| wav2vec2 front-end paper | https://arxiv.org/pdf/2202.12233 |

---

## What NOT to do

- Do not use HuggingFace AASIST port (unmaintained)
- Do not implement LFCC/FFT feature extraction for the MVP (AASIST-L takes raw audio)
- Do not compute CMVN stats from a single test file (use training-set stats)
- Do not apply RawBoost during inference (training only)
- Do not aggressively denoise input audio (noise is evidence)
- Do not retrain when running the Phase 4 cross-domain test (frozen model only)
- Do not output a binary real/fake label without the abstain band (Phase 5+)

---

## Research framing (for the professor)
"We train a standard AASIST-L backbone on ASVspoof 2019 LA.
Our contribution is not the architecture — it is the calibrated
3-band abstain mechanism, and the open study of how that mechanism
absorbs cross-domain generalization error. We quantify what no
commercial tool publishes: how much degradation lands in the abstain
zone vs a confident wrong answer."

---

## Current status
Phase 0 — infra done, inference pipeline verified working end-to-end.
Phase 1 preprocessing module also done + test suite passing (see below,
pulled forward since Phase 0 needed it to run infer.py).

Env: no conda on this machine — using `.venv` (--system-site-packages,
reuses system torch 2.6.0+cpu) + `pip install torchaudio==2.6.0` from the
CPU wheel index. torch-geometric skipped — AASIST.py has zero dependency
on it (verified: only imports numpy/torch/random/typing).

Done:
- src/model/AASIST.py copied from clovaai/aasist (unmodified)
- outputs/weights/AASIST-L.pth, AASIST.pth copied
- configs/aasist_L.json, configs/aasist.json copied from clovaai .conf files
- src/features/rawboost.py copied from TakHemlata/RawBoost-antispoofing,
  with a process_Rawboost_feature() wrapper appended (upstream's version
  takes an argparse Namespace; ours takes explicit kwargs so preprocess.py
  doesn't need argparse). Algo numbering is upstream's — 5 = convolutive+
  impulsive (our LA default), NOT 3 (that's coloured additive, saved for
  Phase 7 DF/codec work). preprocess.py default fixed from 3 to 5 accordingly.
- src/preprocessing/preprocess.py: RawBoost import no longer wrapped in
  try/except ImportError — a silently skipped augmentation during training
  is exactly the train/infer mismatch the GOLDEN RULE forbids; better to
  crash loudly than train un-augmented without noticing.
- infer.py: fixed two bugs found running Phase 0's own success criterion
  (python infer.py --input test.wav):
  1. extra `.unsqueeze(0)` before calling model — preprocess() already
     returns [1, 64600] = [batch, time], which is exactly what
     AASIST.Model.forward expects (it does its own unsqueeze(1) internally).
     The extra dim produced [1,1,64600] and broke SincConv's conv1d.
  2. `Model.forward` returns `(last_hidden, logits)`, not just logits —
     infer.py was indexing the tuple as if it were the logits tensor.
- tests/test_preprocess.py: rewrote temp-file handling. Windows locks an
  open NamedTemporaryFile handle, so torchaudio.save() couldn't reopen the
  same path inside the `with` block (LibsndfileError). Switched to
  pytest's tmp_path fixture. All 10 tests pass.
- Verified inference runs end-to-end on synthetic sine-wave WAVs — pipeline
  is mechanically correct, direction consistent (score > 0 = spoof).
  NOT yet verified against real bonafide/spoof speech (Phase 0's actual
  success criterion) — need real ASVspoof-style audio samples for that,
  sine tones aren't speech and don't exercise the SincNet front-end
  meaningfully.

PHASE 0 COMPLETE. Real speech test done using the actual ASVspoof2019 LA
dataset found in C:\Users\BIT\Downloads\archive (8).zip (full LA+PA
corpus + cm protocols — not used wholesale, just pulled 6 dev files via
the cm.dev.trl.txt protocol: 3 bonafide, 3 spoof/A01-TTS).

Result: 6/6 correct separation, large margin.
  bonafide (LA_D_1047731/1105538/1125976): score +11.2 / +13.4 / +12.1
  spoof    (LA_D_1008730/1034049/1048723): score -10.5 / -13.5 / -14.6

This caught a real bug: infer.py assumed out_layer index 0=bonafide,
1=spoof. Empirically it's the opposite — AASIST's convention is
index 0=spoof, index 1=bonafide. All 6 samples were mislabeled (but
with perfect score separation) until this was fixed. infer.py now
computes score = output[1] - output[0] as the BONAFIDE log-likelihood
ratio, label = "bonafide" if score > 0. Fixed and reverified — all 6
samples now labeled correctly.

Still open (non-blocking for Phase 1):
- ffmpeg not installed on this machine — needed later for mp3/ogg support
  in preprocess.py's torchaudio.load() backend and for Phase 7 codec work.
- The ASVspoof2019 LA+PA zip sitting in Downloads (archive (8).zip) is the
  real training corpus Phase 2 needs. Should be unzipped into
  data/raw/ when Phase 2 starts (~7GB, not extracted yet — only 6 sample
  flacs were pulled out for this smoke test).
