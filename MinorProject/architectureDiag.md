# Architecture Diagram — Analysis, Corrections & Full Explanation

> **File:** `architectureDiag.md`  
> **Project:** Audio Deepfake Detection — 5th Semester Minor Project  
> **Reference:** `phase_plan.pdf` ? Figure 1: System Overview

---

## The Two Diagrams: What Changed

### ? Diagram 1 (Wrong — AI-Hallucinated)

The first generated diagram was **incorrect** because it was inferred from the HuggingFace model card (`lab260/Spectra-AASIST`) instead of reading the actual `phase_plan.pdf`.

**Errors in Diagram 1:**

| Component | What Was Generated (Wrong) | Why It Was Wrong |
|---|---|---|
| Frontend | `wav2vec2-xls-r-300M` SSL Transformer (300M params, ~1.2GB) | Your plan uses bare **AASIST-L** — no SSL frontend |
| MLP Projection Bridge | Present (as a separate stage) | **Does not exist** in the phase plan |
| CMVN step | Missing from preprocessing | **CMVN is required** — it was omitted entirely |
| Train/Eval paths | Not differentiated | Plan explicitly shows **separate TRAIN PATH and EVAL PATH** into AASIST-L |
| Model size | "300M params, 24 transformer layers" | Actual model is **330 KB, fully on-device** |
| GAT-S / GAT-T labels | Shown explicitly as separate external stages | These are internal to AASIST-L, not separate pipeline stages |
| Architecture style | Complex 6-layer vertical breakdown | Should be a **simple 5-column left-to-right pipeline** |

**Root Cause:** The Spectra-AASIST HuggingFace variant (a research-grade, heavy model) was confused with the phase_plan's actual target model (AASIST-L, a lightweight 330KB classifier).

---

### ? Diagram 2 (Corrected — Matches phase_plan.pdf Figure 1)

The corrected diagram faithfully reproduces **Figure 1: System Overview** from `phase_plan.pdf` with 6 columns:

```
INPUT ? PREPROCESSING ? MODEL (AASIST-L) ? SCORING & CROSS-DOMAIN TEST ? CALIBRATION ? OUTPUT
```

**Changes made in Diagram 2 vs Diagram 1:**

1. **Removed** `wav2vec2-xls-r-300M` SSL frontend entirely
2. **Removed** MLP Projection Bridge
3. **Added** `CMVN` as the 5th preprocessing step
4. **Added** explicit `TRAIN PATH` and `EVAL PATH` arrows going into AASIST-L
5. **Changed** model label to `AASIST-L` with "330KB, on-device, no API"
6. **Changed** scoring section to show the exact two-box layout: 2019 LA (green) vs 2021 DF (orange) with `? ~20x gap`
7. **Added** speedometer/gauge icon for the 3-band verdict (REAL / ABSTAIN / FAKE)
8. **Added** `FastAPI endpoint` + `Grad-CAM spectrogram` in the Output column
9. **Added** bottom legend bar: Open & Reproducible · Calibrated with Honest Abstain · Fully Local · Cross-Domain Study
10. **Simplified** overall layout to match the clean, minimal academic style of Figure 1

---

## Full Architecture Explanation (Component by Component)

### Stage 1 — INPUT

```
Raw Audio
(any format / any sample rate)
```

- The system accepts audio in **any format** (WAV, MP3, FLAC, OGG, M4A, etc.)
- Audio can be **any sample rate** — the preprocessing stage handles normalization
- This is intentional for real-world deployment flexibility

---

### Stage 2 — PREPROCESSING

```
+--------------------------------------------------+
¦                  Preprocessing                    ¦
¦                                                   ¦
¦   1. Resample ? 16 kHz                           ¦
¦   2. Mono downmix (stereo ? 1 channel)           ¦
¦   3. Amplitude normalization                     ¦
¦   4. Pad / Crop ? exactly 64,600 samples         ¦
¦   5. CMVN (Cepstral Mean-Variance Normalization) ¦
¦                                                   ¦
¦   ? identical preprocessing, verified by         ¦
¦     unit test                                     ¦
+--------------------------------------------------+
         ¦                    ¦
    TRAIN PATH           EVAL PATH
         +---------------------+
                    ?
               AASIST-L
```

**What each step does:**

| Step | Purpose |
|------|---------|
| **Resample ? 16 kHz** | AASIST-L was trained on 16 kHz audio; all inputs must match this sample rate |
| **Mono downmix** | Converts stereo/multi-channel to single channel; the model expects 1D input |
| **Amplitude normalization** | Scales waveform amplitude to a fixed range; prevents loudness-induced bias |
| **Pad / Crop ? 64,600 samples** | Fixed-length input required by the model. 64,600 ÷ 16,000 Hz = ~4.04 seconds |
| **CMVN** | Cepstral Mean-Variance Normalization. Normalizes feature mean and variance per utterance so the model is robust to channel/microphone conditions |

**Why identical TRAIN/EVAL paths matter:**  
A common mistake in ML pipelines is applying slightly different transforms at training vs. evaluation time. The plan mandates a **unit test** to assert that the exact same function is called on both paths, preventing data leakage and evaluation errors.

---

### Stage 3 — MODEL: AASIST-L

```
+----------------------------------+
¦           AASIST-L               ¦
¦                                  ¦
¦   • 330 KB on disk               ¦
¦   • Fully on-device              ¦
¦   • No external API calls        ¦
¦                                  ¦
¦   [Graph Attention Network]      ¦
¦   Spectro-Temporal heterogeneous ¦
¦   graph attention mechanism      ¦
+----------------------------------+
```

**What AASIST-L is:**
- **AASIST** = Audio Anti-Spoofing using Integrated Spectro-Temporal graph attention networks
- **L** = "Large" variant (vs. the smaller AASIST-S)
- Despite being called "large", it is only **330 KB** — extraordinarily compact
- Internal architecture: heterogeneous graph attention that simultaneously processes spectral (frequency-domain) and temporal (time-domain) features
- No SSL transformer frontend; it operates directly on the processed waveform

**Why no API / fully local:**  
The design principle is *on-device inference with no cloud dependency*. Any audio file stays on the local machine — critical for privacy-sensitive applications.

---

### Stage 4 — SCORING & CROSS-DOMAIN TEST

```
+---------------------------------+  ? GREEN border
¦  ASVspoof 2019 LA               ¦
¦  (in-domain)                    ¦
¦  — EER ~0.83%                   ¦
+---------------------------------+
              ?
         ? ~20x gap
              ?
+---------------------------------+  ? ORANGE border
¦  ASVspoof 2021 DF               ¦
¦  (cross-domain, codec-degraded) ¦
¦  — EER ~15%                     ¦
+---------------------------------+
```

**What EER means:**  
EER = Equal Error Rate. The point where False Acceptance Rate == False Rejection Rate. Lower is better.
- EER 0.83% ? the model almost perfectly distinguishes real vs. fake speech on 2019 data
- EER 15% ? the model fails significantly on 2021 codec-degraded data

**The ? ~20x gap explained:**  
The **same AASIST-L model**, with **no retraining**, tested on two datasets:
- **2019 LA** = clean, uncompressed spoofed speech (in-distribution)
- **2021 DF** = spoofed speech that passed through telephony codecs (MP3, Opus, AMR compression)

The codec compression **destroys the subtle artifact patterns** that AASIST-L learned — causing a 20× performance collapse. This gap is the **core research problem** this project investigates.

**Datasets:**

| Dataset | Type | Purpose |
|---------|------|---------|
| ASVspoof 2019 LA | Logical Access — synthetic TTS/VC speech | Training + in-domain eval |
| ASVspoof 2021 DF | Deepfake — 2019 data + codec compression | Cross-domain stress test |

---

### Stage 5 — CALIBRATION: Platt Scaling + 3-Band Verdict

```
Raw logits (AASIST-L output)
           ¦
           ?
    Platt Scaling
    (sigmoid fit on dev set scores)
           ¦
           ?
   Calibrated probabilities
           ¦
           ?
+-----------------------------+
¦  REAL  ¦ ABSTAIN  ¦  FAKE   ¦
¦  (?)  ¦   (?)   ¦  (?)   ¦
+-----------------------------+
```

**What Platt Scaling is:**  
A post-hoc calibration technique that fits a logistic regression on top of the model's raw output scores to convert them into well-calibrated probabilities. Applied using the **development set** (not the test set).

**Why a 3-band verdict (not binary):**  
Instead of hard REAL/FAKE decisions, a middle "ABSTAIN" band is defined where the model's confidence is too low to commit. On cross-domain inputs (codec-degraded audio), the model is uncertain, and it should say so rather than making a confidently wrong prediction.

---

### Stage 6 — OUTPUT: Explainability + Deployment

```
+----------------------------------------------+
¦          Explainability + Deployment         ¦
¦                                              ¦
¦  [Grad-CAM spectrogram heatmap]  [FastAPI]  ¦
¦                                              ¦
¦  fully local inference.                      ¦
+----------------------------------------------+
```

**Grad-CAM Explainability:**  
- Gradient-weighted Class Activation Mapping applied to AASIST-L's graph attention layers
- Generates a **heatmap over the spectrogram**: highlights which time-frequency regions were most influential in the SPOOF/REAL decision
- Answers: *"Why did the model say FAKE?"*

**FastAPI Endpoint:**  
- A `/predict` HTTP endpoint accepting audio file uploads, returning the 3-band verdict + score
- Runs entirely on localhost — no cloud API, no GPU required for inference

---

## Bottom Legend — 4 Design Principles

| Principle | Meaning |
|-----------|---------|
| ?? **Open & Reproducible** | All code, weights, and eval scripts are open-source; results are independently reproducible |
| ?? **Calibrated with Honest Abstain** | Platt scaling + 3-band verdict; the model admits uncertainty rather than guessing |
| ?? **Fully Local** | No cloud inference, no API keys, no internet required at runtime |
| ?? **Cross-Domain Study** | Explicitly measures the 20× EER gap between ASVspoof 2019 and 2021 |

---

## Summary: Correct Complete Pipeline

```
Raw Audio (any format/rate)
    ¦
    ?
Preprocessing:
    1. resample ? 16kHz
    2. mono downmix
    3. amplitude norm
    4. pad/crop ? 64,600 samples
    5. CMVN
    ¦           ¦
 TRAIN PATH   EVAL PATH   ? identical, verified by unit test
    +----------+
          ?
     AASIST-L (330KB, on-device, no API)
          ¦
          +--? ASVspoof 2019 LA eval ? EER ~0.83%   (in-domain)
          ¦              ?  ? ~20x gap
          +--? ASVspoof 2021 DF eval ? EER ~15%    (cross-domain)
          ¦
          ?
    Platt Scaling calibration
          ¦
          ?
    +-------------------------+
    ¦ REAL  ¦ ABSTAIN ¦ FAKE  ¦
    +-------------------------+
          ¦
          ?
    Grad-CAM spectrogram heatmap  +  FastAPI /predict endpoint
    (fully local inference)
```

---

*Document: `architectureDiag.md`*  
*Audio Deepfake Detection — 5th Semester Minor Project*  
*Reference: `phase_plan.pdf` ? Figure 1: System Overview*
