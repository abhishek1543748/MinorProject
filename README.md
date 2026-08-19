# Audio Deepfake Detection

Lightweight, fully-local voice spoof detector.  
**AASIST-L backbone · calibrated 3-band verdict · no API key · runs on a CPU laptop.**

University research project. Supervised by a professor.

---

## Quick start

```bash
# 1. Create virtual environment and install dependencies
python -m venv .venv --system-site-packages
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux / macOS

pip install torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# 2. Run inference on one file
python infer.py --input path/to/audio.wav
```

Example output:
```json
{
  "label": "bonafide",
  "score": 11.1875,
  "prob_bonafide": 1.0,
  "prob_spoof": 0.0,
  "weights": "outputs/weights/AASIST-L.pth",
  "audio": "path/to/audio.wav"
}
```

> **Note:** Model weights (`outputs/weights/AASIST-L.pth`) are included in the repo.  
> Raw audio data is NOT committed — see [`data/README.md`](data/README.md) for download instructions.

---

## Phase checklist

| Phase | Goal | Status |
|---|---|---|
| 0 | Setup + first inference on real ASVspoof files | ✅ Complete |
| 1 | Shared preprocessing module + 10/10 tests passing | ✅ Complete |
| 2 | Training pipeline on ASVspoof 2019 LA | ⏳ Next |
| 3 | Baseline EER (target: ~0.83%) | ⏳ Pending |
| 4 | Cross-domain stress test on ASVspoof 2021 DF | ⏳ Pending |
| 5 | Calibration + 3-band abstain verdict | ⏳ Pending |
| 6 | Grad-CAM + FastAPI + web UI | ⏳ Pending |
| 7 | Codec robustness study | ⏳ Pending |
| 8 | Write-up + professor review | ⏳ Pending |

---

## Repository structure

```
deepfake_detector/
├── configs/
│   ├── aasist_L.json          ← AASIST-L model config (from clovaai/aasist)
│   ├── aasist.json            ← Full AASIST config
│   └── train_config.json      ← Training hyperparameters + data paths
├── data/
│   ├── README.md              ← Dataset download instructions
│   ├── raw/                   ← ASVspoof 2019 LA/PA (gitignored — ~24 GB)
│   └── processed/             ← Cached tensors (gitignored, auto-generated)
├── docs/
│   ├── ASVspoof2019_dataset_readme.txt
│   ├── asvspoof2019_evaluation_plan.pdf
│   └── asvspoof2019_Interspeech2019_submission.pdf
├── outputs/
│   ├── weights/               ← AASIST-L.pth (pretrained) + best_model.pth (trained)
│   ├── scores/                ← Per-file score CSVs (gitignored)
│   └── plots/                 ← EER curves, histograms (gitignored)
├── scripts/
│   ├── setup_env.sh           ← Environment setup instructions
│   └── download_data.sh       ← Dataset download instructions
├── src/
│   ├── api/server.py          ← FastAPI POST /analyze endpoint (Phase 6)
│   ├── calibration/calibrate.py ← Platt scaling + 3-band verdict (Phase 5)
│   ├── explainability/        ← Grad-CAM (Phase 6)
│   ├── features/rawboost.py   ← RawBoost augmentation (training only)
│   ├── model/AASIST.py        ← AASIST-L model (from clovaai/aasist, unmodified)
│   └── preprocessing/preprocess.py ← GOLDEN RULE module (shared by train + infer)
├── tests/
│   └── test_preprocess.py     ← 10 tests, all passing
├── eval.py                    ← Batch EER + tDCF scoring
├── infer.py                   ← Single-file inference entry point
├── train.py                   ← Training entry point (Phase 2)
├── requirements.txt
└── CLAUDE.md                  ← Full project brief for AI assistants
```

---

## Architecture

```
Audio file (.wav / .flac / .mp3)
    ↓
[preprocess.py]  16kHz · mono · normalize · pad/crop to 64,600 samples
    ↓                       ← GOLDEN RULE: identical in train + infer
[AASIST-L]  85K params · raw waveform → 2 logits (SincNet + graph attention)
    ↓
score = logit[1] - logit[0]   (bonafide log-likelihood ratio)
    ↓
[Calibrator]  Platt scaling → 3-band verdict
    ↓
{ band: "authentic" | "uncertain" | "spoof", prob_spoof, confidence }
```

---

## Research contribution

Not a new architecture. The contribution is:
- **Calibrated 3-band verdict**: authentic / uncertain (abstain) / spoof
- **Generalization study**: quantifying how much cross-domain error the abstain zone absorbs
- **Open + reproducible**: standard published architecture + full degradation curves

No commercial tool publishes abstain-zone characterization under codec degradation.

---

## Key sources

| Resource | URL |
|---|---|
| AASIST (official) | https://github.com/clovaai/aasist |
| RawBoost | https://github.com/TakHemlata/RawBoost |
| ASVspoof 2019 LA | https://datashare.ed.ac.uk/handle/10283/3336 |
| ASVspoof 2021 DF | https://zenodo.org/record/4835108 |
| AASIST paper | https://arxiv.org/pdf/2110.01200 |
| RawBoost paper | https://arxiv.org/pdf/2111.04433 |

> **Do NOT use the HuggingFace AASIST port — it is unmaintained.**  
> Always use `github.com/clovaai/aasist`.
