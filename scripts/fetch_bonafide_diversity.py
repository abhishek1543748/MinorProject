"""
fetch_bonafide_diversity.py — real-world bonafide speech for AASIST-L fine-tuning
=================================================================================
Why: AASIST-L confidently calls unseen REAL speakers spoof (25.7% of 2021 DF
bonafide, 65.6% of vcc2018 bonafide; see outputs/scores/phase5_abstain_analysis.txt).
ASVspoof 2019 LA bonafide is ~2.6k utterances from VCTK studio recordings, so
"unfamiliar microphone / room" looks like an artifact. This script collects
2-3k bonafide clips from other corpora and writes them as an extra protocol
that train.py mixes into the LA train set (--extra-train-protocol).

Sources (all streamed — nothing larger than one clip is held in memory):
  librispeech : OpenSLR LibriSpeech tar.gz (default dev-other: 33 speakers,
                varied LibriVox home microphones), read member by member over HTTP.
  hf          : any Hugging Face audio dataset in streaming mode, e.g.
                --hf-name mozilla-foundation/common_voice_17_0 --hf-config en
                --hf-split validation --hf-speaker client_id
                (Common Voice needs `huggingface-cli login` + accepting its terms;
                requires `pip install datasets`)
  dir         : a local folder of audio files (any format preprocess.load_audio
                reads), e.g. your own phone/laptop recordings. Speaker = subfolder.

Output (default data/raw/bonafide_diversity/):
  flac/<id>.flac          16 kHz mono 16-bit, NOT peak-normalised (preprocess() does that)
  protocol_train.txt      "<speaker> <id> - - bonafide"   (ASVspoofDataset format)
  protocol_dev.txt        speaker-disjoint hold-out (--dev-frac of speakers)

Resampling/downmix use src.preprocessing.preprocess.resample / to_mono (GOLDEN
RULE: no second resampler). Clips are cut to --max-seconds because train-mode
preprocess() only sees the first 4.04 s anyway.

Usage:
  python scripts/fetch_bonafide_diversity.py --n 2500                  # LibriSpeech dev-other
  python scripts/fetch_bonafide_diversity.py --n 20 --out /tmp/bd      # quick check
  python scripts/fetch_bonafide_diversity.py --source dir --input my_recordings/
"""

import argparse
import io
import random
import re
import sys
import tarfile
import urllib.request
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
import torch  # noqa: E402

from src.preprocessing.preprocess import TARGET_SR, load_audio, resample, to_mono  # noqa: E402

LIBRISPEECH_URL = "https://www.openslr.org/resources/12/{subset}.tar.gz"
AUDIO_EXT = {".wav", ".flac", ".mp3", ".ogg", ".m4a", ".opus", ".webm"}


def get_args(argv=None):
    p = argparse.ArgumentParser(description="Collect diverse bonafide speech for fine-tuning")
    p.add_argument("--source", choices=["librispeech", "hf", "dir"], default="librispeech")
    p.add_argument("--n", type=int, default=2500, help="total clips to write")
    p.add_argument("--per-speaker", type=int, default=None,
                   help="cap clips per speaker (default: spread --n evenly, min 10)")
    p.add_argument("--max-seconds", type=float, default=8.0)
    p.add_argument("--min-seconds", type=float, default=1.0)
    p.add_argument("--dev-frac", type=float, default=0.1, help="share of SPEAKERS held out")
    p.add_argument("--out", default=str(ROOT / "data/raw/bonafide_diversity"))
    p.add_argument("--seed", type=int, default=0)
    # librispeech
    p.add_argument("--subset", default="dev-other",
                   help="LibriSpeech subset: dev-other, dev-clean, test-other, train-other-500 ...")
    p.add_argument("--url", default=None, help="override tarball URL (mirror)")
    # hf
    p.add_argument("--hf-name", default="mozilla-foundation/common_voice_17_0")
    p.add_argument("--hf-config", default="en")
    p.add_argument("--hf-split", default="validation")
    p.add_argument("--hf-speaker", default="client_id", help="speaker id column")
    # dir
    p.add_argument("--input", default=None, help="--source dir: folder of audio files")
    a = p.parse_args(argv)
    if a.source == "dir" and not a.input:
        p.error("--source dir needs --input")
    return a


# ── Sources: yield (speaker, clip_id, waveform [C, T] float32, sample_rate) ──

def iter_librispeech(args):
    url = args.url or LIBRISPEECH_URL.format(subset=args.subset)
    print(f"[fetch] streaming {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "deepfake-detector-fetch/1.0"})
    with urllib.request.urlopen(req) as resp, tarfile.open(fileobj=resp, mode="r|gz") as tar:
        for m in tar:
            if not (m.isfile() and m.name.endswith(".flac")):
                continue
            utt = Path(m.name).stem                       # <spk>-<chapter>-<utt>
            data = tar.extractfile(m).read()
            x, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
            yield f"LS{utt.split('-')[0]}", f"LS_{utt.replace('-', '_')}", \
                torch.from_numpy(x.T.copy()), sr


def iter_hf(args):
    from datasets import load_dataset
    ds = load_dataset(args.hf_name, args.hf_config, split=args.hf_split,
                      streaming=True, trust_remote_code=True)
    tag = re.sub(r"[^A-Za-z0-9]", "", args.hf_name.split("/")[-1])[:8].upper()
    for i, ex in enumerate(ds):
        a = ex["audio"]
        spk = re.sub(r"[^A-Za-z0-9]", "", str(ex.get(args.hf_speaker, "unk")))[:16] or "unk"
        x = torch.as_tensor(np.asarray(a["array"], dtype=np.float32))
        yield f"{tag}{spk}", f"{tag}_{i:07d}", x.reshape(1, -1) if x.ndim == 1 else x.T, \
            a["sampling_rate"]


def iter_dir(args):
    root = Path(args.input)
    for f in sorted(root.rglob("*")):
        if f.suffix.lower() not in AUDIO_EXT:
            continue
        rel = f.relative_to(root)
        spk = re.sub(r"[^A-Za-z0-9]", "", rel.parts[0] if len(rel.parts) > 1 else "own") or "own"
        cid = re.sub(r"[^A-Za-z0-9_]", "_", "_".join(rel.with_suffix("").parts))
        x, sr = load_audio(str(f))
        yield f"DIR{spk}", f"DIR_{cid}", x, sr


SOURCES = {"librispeech": iter_librispeech, "hf": iter_hf, "dir": iter_dir}


# ── Main ─────────────────────────────────────────────────────────────────────

def to_16k_mono(x: torch.Tensor, sr: int) -> torch.Tensor:
    """Same downmix + polyphase resampler as preprocess() (GOLDEN RULE)."""
    return resample(to_mono(x.float()), sr)


def write_protocols(out: Path, entries: list[tuple[str, str]], dev_frac: float, seed: int):
    speakers = sorted({s for s, _ in entries})
    rng = random.Random(seed)
    rng.shuffle(speakers)
    n_dev = int(round(len(speakers) * dev_frac)) if len(speakers) > 1 else 0
    dev_spk = set(speakers[:n_dev])
    for name, keep in (("train", lambda s: s not in dev_spk), ("dev", lambda s: s in dev_spk)):
        lines = [f"{s} {c} - - bonafide" for s, c in entries if keep(s)]
        (out / f"protocol_{name}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
        print(f"[fetch] protocol_{name}.txt: {len(lines)} clips, "
              f"{len({s for s, _ in entries if keep(s)})} speakers")


def main(argv=None):
    args = get_args(argv)
    out = Path(args.out)
    (out / "flac").mkdir(parents=True, exist_ok=True)
    cap = args.per_speaker or max(10, args.n // 30)

    per_spk = defaultdict(int)
    entries, skipped = [], 0
    for spk, cid, x, sr in SOURCES[args.source](args):
        if len(entries) >= args.n:
            break
        if per_spk[spk] >= cap:
            continue
        y = to_16k_mono(x, sr)[:, :int(args.max_seconds * TARGET_SR)]
        if y.shape[1] < args.min_seconds * TARGET_SR or y.abs().max() == 0:
            skipped += 1
            continue
        peak = y.abs().max().item()
        if peak > 1.0:                                 # keep FLAC from clipping; scale is
            y = y / peak                               # irrelevant (preprocess peak-normalises)
        sf.write(str(out / "flac" / f"{cid}.flac"), y[0].numpy(), TARGET_SR, subtype="PCM_16")
        entries.append((spk, cid))
        per_spk[spk] += 1
        if len(entries) % 250 == 0:
            print(f"  {len(entries)}/{args.n} clips, {len(per_spk)} speakers", flush=True)

    if not entries:
        raise SystemExit("[fetch] no clips written")
    if len(entries) < args.n:
        print(f"[fetch] WARN: source exhausted at {len(entries)} clips "
              f"(per-speaker cap {cap}; raise --per-speaker or pick a larger subset)")
    print(f"[fetch] wrote {len(entries)} clips from {len(per_spk)} speakers "
          f"({skipped} too short/silent skipped) -> {out / 'flac'}")
    write_protocols(out, entries, args.dev_frac, args.seed)
    return entries


if __name__ == "__main__":
    main()
