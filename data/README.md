# Data Directory

Raw audio data is **not committed** to this repository (too large: ~24 GB).

## Required datasets

### ASVspoof 2019 LA (required — training + evaluation)

**Download:** https://datashare.ed.ac.uk/handle/10283/3336

After downloading, extract so that the structure is:

```
data/raw/ASVspoof2019_LA/
├── ASVspoof2019_LA_cm_protocols/
│   ├── ASVspoof2019.LA.cm.train.trn.txt
│   ├── ASVspoof2019.LA.cm.dev.trl.txt
│   └── ASVspoof2019.LA.cm.eval.trl.txt
├── ASVspoof2019_LA_train/flac/    ← 25,380 files
├── ASVspoof2019_LA_dev/flac/      ← 24,986 files
└── ASVspoof2019_LA_eval/flac/     ← 71,933 files
```

### ASVspoof 2021 DF (optional — Phase 4 cross-domain stress test)

**Download:** https://zenodo.org/record/4835108

Extract to: `data/raw/ASVspoof2021_DF/`

### ASVspoof 2019 PA (optional — Physical Access condition, not used in main study)

Already in your local copy under `data/raw/ASVspoof2019_PA/`.
Not used by any current phase — reserved for future work.

## Processed data

`data/processed/` — cached tensors (optional, auto-generated during training).
Also gitignored.
