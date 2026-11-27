"""
test_refit_calibrator.py — windowed calibration refit (scripts/refit_windowed_calibrator.py)
"""

import csv
import sys
from pathlib import Path

import pytest
import torch
import torchaudio

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
WEIGHTS = ROOT / "outputs/weights/AASIST-L.pth"
needs_weights = pytest.mark.skipif(not WEIGHTS.exists(), reason="AASIST-L.pth missing")


@pytest.fixture
def mini_corpus(tmp_path):
    """6 files (3 'bonafide', 3 'spoof'), mixed lengths so some have >1 window."""
    audio = tmp_path / "flac"
    audio.mkdir()
    lines = []
    torch.manual_seed(0)
    for i in range(6):
        sec = [2, 6, 9][i % 3]
        x = 0.1 * torch.randn(1, 16000 * sec)
        torchaudio.save(str(audio / f"F_{i}.wav"), x, 16000)
        lines.append(f"SPK{i} F_{i} - - {'bonafide' if i < 3 else 'spoof'}")
    proto = tmp_path / "proto.txt"
    proto.write_text("\n".join(lines) + "\n")
    return proto, audio


def _argv(proto, audio, tmp_path, *extra):
    return ["--sets", "dev", "--dev-protocol", str(proto), "--dev-audio", str(audio),
            "--weights", str(WEIGHTS), "--scores-csv", str(tmp_path / "s.csv"),
            "--out", str(tmp_path / "cal.pkl"), *extra]


@needs_weights
def test_refit_matches_server_and_resumes(mini_corpus, tmp_path, monkeypatch):
    import refit_windowed_calibrator as refit
    from fastapi.testclient import TestClient
    from src.api.server import app
    from src.calibration.calibrate import Calibrator

    proto, audio = mini_corpus
    refit.main(_argv(proto, audio, tmp_path))
    rows = list(csv.DictReader(open(tmp_path / "s.csv")))
    assert len(rows) == 6 and any(int(r["n_windows"]) > 1 for r in rows)
    assert isinstance(Calibrator.load(str(tmp_path / "cal.pkl")), Calibrator)

    # The score the calibrator is fitted on == the raw_score the server computes
    with TestClient(app) as client:
        for r in rows:
            with open(audio / f"{r['filename']}.wav", "rb") as f:
                d = client.post("/analyze?explain=false",
                                files={"file": ("x.wav", f, "audio/wav")}).json()
            assert d["raw_score"] == pytest.approx(float(r["score"]), abs=1e-3)

    # Second run: everything cached -> model must not even be loaded
    monkeypatch.setattr(refit, "load", lambda a: (_ for _ in ()).throw(AssertionError("rescored")))
    refit.main(_argv(proto, audio, tmp_path))
    assert len(list(csv.DictReader(open(tmp_path / "s.csv")))) == 6


def test_subsample_balanced_and_seeded():
    import refit_windowed_calibrator as refit

    items = [{"filename": f"f{i}", "label": int(i >= 10), "speaker": "s"} for i in range(100)]
    a = refit.subsample(items, 20, seed=0)
    assert len(a) == 20 and sum(x["label"] == 0 for x in a) == 10
    assert a == refit.subsample(items, 20, seed=0)
    assert refit.subsample(items, None, 0) is items


def test_read_items_formats_and_df_speaker_split(tmp_path):
    import refit_windowed_calibrator as refit
    from domain_calibration import is_cal_speaker

    p = tmp_path / "p.txt"
    p.write_text("LA_0079 LA_T_1 - - bonafide\n"
                 "LA_0034 DF_E_2 high_mp3 asvspoof A07 spoof notrim progress x - - - -\n"
                 "TFM1 DF_E_3 mp3m4a vcc2020 Task2 bonafide notrim eval bonafide\n")
    items = refit.read_items(str(p))
    assert [x["label"] for x in items] == [0, 1, 0]
    cal = refit.read_items(str(p), only_cal_speakers=True)
    assert all(is_cal_speaker(x["speaker"]) for x in cal)
