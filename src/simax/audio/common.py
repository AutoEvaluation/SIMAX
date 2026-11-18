"""Shared paths, voices, chunk planning and dialogue assembly for the audio steps."""
from __future__ import annotations

import hashlib
import json
import math
import os
import random
import re
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
VOICE_DIR = Path(os.environ.get("SIMAX_VOICE_DIR", "voices"))
VOICES_JSON = REPO_ROOT / "configs" / "voices.json"
QC_CACHE = Path(os.environ.get("SIMAX_QC_CACHE", ".cache/simax_qc"))

EXAM_CUE = re.compile(
    r"\b(examin\w*|take a (quick )?look|have a (quick )?look|let me (check|feel|listen|press|look)|"
    r"lie back|hop up|breathe|listen to|measure|blood pressure|heartbeat|doppler|bend|straighten|"
    r"squeeze|press(ing)? (here|on|down))", re.IGNORECASE)


def wjson(path, obj) -> None:
    path = Path(path)
    tmp = path.with_name(f"{path.name}.tmp{os.getpid()}")
    tmp.write_text(json.dumps(obj, ensure_ascii=False))
    os.replace(tmp, path)


def rjson(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return default


def voice_sentences() -> dict[str, str]:
    return {v["wav"]: v["sentence"] for v in json.loads(VOICES_JSON.read_text())["voices"]}


def voice_path(voice: str) -> Path:
    path = VOICE_DIR / "wav16k" / voice
    if not path.exists():
        raise FileNotFoundError(f"{path}: set SIMAX_VOICE_DIR to the folder that holds wav16k/")
    return path


def natural_gaps(turns, sample_id: str) -> list[float]:
    rng = random.Random(int(hashlib.sha256(sample_id.encode()).hexdigest()[:12], 16))
    gaps = []
    for turn in turns:
        gap = min(3.0, max(0.25, rng.lognormvariate(math.log(1.0), 0.5)))
        if turn["speaker"] == "doctor":
            if EXAM_CUE.search(turn["text"]):
                gap = rng.uniform(6.0, 20.0)
            elif rng.random() < 0.2:
                gap = rng.uniform(3.0, 8.0)
        gaps.append(gap)
    return gaps


def chunk_turns(turns, gaps, max_words: int = 700, min_words: int = 250) -> list[list[int]]:
    chunks, start, words = [], 0, 0
    for i, turn in enumerate(turns):
        words += len(turn["text"].split())
        last = i == len(turns) - 1
        following = 0 if last else len(turns[i + 1]["text"].split())
        if last or words + following > max_words or (words >= min_words and gaps[i] >= 5.0):
            chunks.append([start, i + 1])
            start, words = i + 1, 0
    return chunks


def prepare_folder(row: dict, root) -> Path:
    folder = Path(root) / f"{row['sample_id']}__{row['family']}_{row['label']}"
    folder.mkdir(parents=True, exist_ok=True)
    spec = folder / "dialogue_input.json"
    if not spec.exists():
        gaps = natural_gaps(row["turns"], row["sample_id"])
        chunks = chunk_turns(row["turns"], gaps)
        wjson(spec, {**row, "chunks": chunks, "gap_after_s": [round(gaps[b - 1], 2) for _, b in chunks]})
    return folder


def dialogue_spec(folder) -> dict | None:
    return rjson(Path(folder) / "dialogue_input.json")


def chunk_sig(folder, k: int):
    try:
        st = (Path(folder) / f"chunk_{k:02d}.wav").stat()
    except FileNotFoundError:
        return None
    return [st.st_size, int(st.st_mtime)]


def rebuild_full(folder) -> None:
    import soundfile as sf

    folder = Path(folder)
    spec = dialogue_spec(folder)
    turns, pieces, meta, sr = spec["turns"], [], [], None
    for k, ((a, b), gap) in enumerate(zip(spec["chunks"], spec["gap_after_s"])):
        audio, rate = sf.read(str(folder / f"chunk_{k:02d}.wav"), dtype="float32")
        audio = audio.mean(1) if audio.ndim > 1 else audio
        sr = sr or rate
        if rate != sr:
            raise ValueError("chunk sample rates differ")
        seconds = len(audio) / sr
        words = sum(len(t["text"].split()) for t in turns[a:b])
        meta.append({"turns": [a, b], "words": words, "seconds": round(seconds, 2),
                     "words_per_minute": round(words / max(seconds, 1e-6) * 60, 1), "gap_after_s": gap})
        pieces.append(audio)
        if k < len(spec["chunks"]) - 1:
            pieces.append(np.zeros(int(gap * sr), dtype=np.float32))
    full = np.concatenate(pieces)
    peak = float(np.abs(full).max()) or 1.0
    tmp = folder / "dialogue_full.part.wav"
    sf.write(str(tmp), (0.95 * full / peak) if peak > 1 else full, sr)
    tmp.replace(folder / "dialogue_full.wav")
    wjson(folder / "dialogue.json", {**{k: v for k, v in spec.items() if k not in ("chunks", "gap_after_s")},
                                     "sample_rate": sr, "chunks": meta, "minutes": round(len(full) / sr / 60, 2)})
