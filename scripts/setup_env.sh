#!/bin/bash
# Audio Deepfake Detection — Environment Setup
# Run once: bash scripts/setup_env.sh

set -e

echo "==> Creating conda environment: deepfake"
conda create -n deepfake python=3.10 -y
conda activate deepfake || source activate deepfake

echo "==> Installing PyTorch (CPU build — change for GPU)"
# CPU-only (works on any laptop):
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu

# If you have an NVIDIA GPU, use this instead:
# pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118

echo "==> Installing all dependencies"
pip install -r requirements.txt

echo "==> Cloning clovaai/aasist (official source)"
git clone https://github.com/clovaai/aasist /tmp/aasist_repo

echo "==> Copying AASIST model and weights into project"
cp /tmp/aasist_repo/models/AASIST.py src/model/AASIST.py
mkdir -p outputs/weights
cp /tmp/aasist_repo/models/weights/AASIST-L.pth outputs/weights/AASIST-L.pth
cp /tmp/aasist_repo/models/weights/AASIST.pth  outputs/weights/AASIST.pth

echo "==> Cloning RawBoost (augmentation)"
git clone https://github.com/TakHemlata/RawBoost /tmp/rawboost_repo
cp /tmp/rawboost_repo/RawBoost.py src/features/rawboost.py

echo ""
echo "==> Done. Verify with:"
echo "    conda activate deepfake"
echo "    python infer.py --input <any_audio_file.wav>"
echo ""
echo "REMINDER: Never use the HuggingFace AASIST port. Always use clovaai/aasist."
