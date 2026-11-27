"""
test_phase2.py — Verification of Phase 2 Training Components
=============================================================
Tests:
1. LMCLLoss (CosFace) forward + backward gradient flow + build_loss factory.
2. ASVspoofDataset & collate_fn shape assertions ([B, 64600]).
3. Complete forward + loss + backward + optimizer step through AASIST-L.
"""

import json
import pytest
import torch
from pathlib import Path

from src.model.loss import LMCLLoss, CELoss, build_loss
from src.data.dataset import collate_fn
from src.model.AASIST import Model


def test_lmcl_loss_forward_backward():
    """Verify LMCLLoss produces finite loss and valid gradient flow."""
    criterion = LMCLLoss(num_classes=2, s=30.0, m=0.4)
    # Seeded: with s=30, some unseeded draws make CE underflow to exactly 0.0
    # in float32 and the "loss > 0" check failed depending on test order.
    torch.manual_seed(0)
    logits = torch.randn(4, 2, requires_grad=True)
    labels = torch.tensor([0, 1, 0, 1], dtype=torch.long)

    loss = criterion(logits, labels)
    assert torch.isfinite(loss), "LMCL loss returned non-finite value"
    assert loss.item() > 0, "Loss should be positive"

    loss.backward()
    assert logits.grad is not None, "Gradient did not flow back to logits"
    assert torch.all(torch.isfinite(logits.grad)), "Gradients contain NaN/Inf"


def test_build_loss_factory():
    """Test loss factory builder."""
    lmcl = build_loss("LMCL", num_classes=2)
    assert isinstance(lmcl, LMCLLoss)

    ce = build_loss("CE")
    assert isinstance(ce, CELoss)

    with pytest.raises(ValueError):
        build_loss("UNKNOWN_LOSS")


def test_collate_fn_shapes():
    """Verify collate_fn outputs 2D tensor [B, 64600] matching AASIST input."""
    item1 = (torch.randn(1, 64600), torch.tensor(0), "file1")
    item2 = (torch.randn(1, 64600), torch.tensor(1), "file2")
    batch = [item1, item2]

    waveforms, labels, filenames = collate_fn(batch)
    assert waveforms.shape == (2, 64600), f"Expected [2, 64600], got {waveforms.shape}"
    assert labels.shape == (2,)
    assert filenames == ["file1", "file2"]


def test_model_training_step():
    """Verify a complete forward + loss + backward + step pass through AASIST-L."""
    config_path = Path("configs/aasist_L.json")
    assert config_path.exists(), "configs/aasist_L.json not found"

    with open(config_path) as f:
        cfg = json.load(f)

    model = Model(cfg["model_config"])
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = LMCLLoss(num_classes=2, s=30.0, m=0.4)

    # Batch of 2 clips with shape [B, 64600]
    dummy_input = torch.randn(2, 64600)
    dummy_labels = torch.tensor([0, 1], dtype=torch.long)

    optimizer.zero_grad()
    _, logits = model(dummy_input)
    assert logits.shape == (2, 2), f"Expected logits [2, 2], got {logits.shape}"

    loss = criterion(logits, dummy_labels)
    assert torch.isfinite(loss)

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
    optimizer.step()


# ── Fine-tuning (train.py --resume_weights + mixed bonafide protocols) ──────

def test_finetune_resume_weights_smoke(tmp_path):
    """2 tiny epochs from AASIST-L.pth with an extra bonafide protocol mixed in."""
    import json
    from pathlib import Path
    import pytest
    import torchaudio
    import train

    root = Path(__file__).resolve().parent.parent
    weights = root / "outputs/weights/AASIST-L.pth"
    if not weights.exists():
        pytest.skip("AASIST-L.pth missing")

    torch.manual_seed(0)
    def corpus(name, labels):
        d = tmp_path / name
        d.mkdir()
        lines = []
        for i, lab in enumerate(labels):
            torchaudio.save(str(d / f"{name}_{i}.wav"), 0.1 * torch.randn(1, 16000 * 2), 16000)
            lines.append(f"S{i} {name}_{i} - - {lab}")
        (d / "p.txt").write_text("\n".join(lines) + "\n")
        return d / "p.txt", d

    tr_p, tr_d = corpus("tr", ["bonafide", "spoof"] * 2)
    dv_p, dv_d = corpus("dv", ["bonafide", "spoof"])
    ex_p, ex_d = corpus("ex", ["bonafide"] * 2)
    cfg = json.loads((root / "configs/finetune_config.json").read_text())
    cfg["data"].update(train_protocol=str(tr_p), train_audio_dir=str(tr_d),
                       dev_protocol=str(dv_p), dev_audio_dir=str(dv_d))
    cfg["model_config"] = str(root / "configs/aasist_L.json")
    cfg["paths"] = {"weights_dir": str(tmp_path / "w"), "scores_dir": str(tmp_path / "s")}
    (tmp_path / "cfg.json").write_text(json.dumps(cfg))

    train.main(["--config", str(tmp_path / "cfg.json"), "--epochs", "2", "--batch-size", "2",
                "--resume_weights", str(weights),
                "--extra-train-protocol", str(ex_p), "--extra-train-audio-dir", str(ex_d),
                "--extra-dev-protocol", str(ex_p), "--extra-dev-audio-dir", str(ex_d),
                "--select-protocol", str(dv_p), "--select-audio-dir", str(dv_d),
                "--save-name", "ft.pth"])
    log = (tmp_path / "s" / "ft_log.csv").read_text().splitlines()
    assert len(log) == 3                                    # header + 2 epochs
    assert float(log[1].split(",")[3]) == pytest.approx(1e-5)   # fine-tune default lr
    assert log[0].endswith("select_eer") and log[1].split(",")[5] != ""
    last = torch.load(tmp_path / "w" / "ft_last.pth", weights_only=False)   # always written
    assert last["epoch"] == 1 and last["select_eer"] is not None


def test_make_df_protocols(tmp_path):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    from make_df_protocols import convert
    from domain_calibration import is_cal_speaker
    from src.data.dataset import ASVspoofDataset

    spk = ["LA_0001", "LA_0002", "LA_0003", "LA_0004", "LA_0005", "LA_0006"]
    rows = [f"{s} DF_E_{i} nocodec asvspoof {'-' if i % 2 else 'A07'} "
            f"{'bonafide' if i % 2 else 'spoof'} notrim eval x - - - -" for i, s in enumerate(spk)]
    (tmp_path / "s.txt").write_text("\n".join(rows) + "\n")
    n = convert(tmp_path / "s.txt", tmp_path)
    assert n["cal"] + n["test"] == 6
    for name in ("cal", "test"):
        lines = (tmp_path / f"df_{name}_protocol.txt").read_text().split()
        for i, s in enumerate(spk):
            assert (f"DF_E_{i}" in lines) == (is_cal_speaker(s) == (name == "cal"))
    # ASVspoofDataset must parse it (label in column 5)
    items = ASVspoofDataset._parse_protocol(ASVspoofDataset.__new__(ASVspoofDataset),
                                            tmp_path / "df_cal_protocol.txt")
    assert {it["label"] for it in items} <= {0, 1} and len(items) == n["cal"]


def test_train_args_validation():
    import pytest
    import train
    with pytest.raises(SystemExit):
        train.get_args(["--resume", "a.pth", "--resume_weights", "b.pth"])
    with pytest.raises(SystemExit):
        train.get_args(["--extra-train-protocol", "p.txt"])    # missing audio dir
    a = train.get_args(["--resume_weights", "x.pth"])
    assert a.lr is None and a.save_name == "best_model.pth"
