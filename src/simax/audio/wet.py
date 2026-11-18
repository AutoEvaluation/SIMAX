"""Add clinic room acoustics to dry dialogue audio."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal

ASSETS = Path(os.environ.get("SIMAX_AUDIO_ASSETS", Path(__file__).resolve().parents[3] / "resources" / "audio_assets"))
IR = ASSETS / "ir" / "FourPointsRoom270.wav"
BED = ASSETS / "201638__bassboybg__hospital-waiting-room.wav"
KEYBOARD = ASSETS / "331428__m4taiori__mechanical-keyboard-typing.mp3"
MOUSE = ASSETS / "545960__metrowned__mouse_click.wav"
DOOR = ASSETS / "126044__mhtaylor67__office-door-closing.wav"


def mono(x: np.ndarray) -> np.ndarray:
    return x.mean(axis=1) if x.ndim == 2 else x


def resample(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return x
    g = np.gcd(sr_in, sr_out)
    return signal.resample_poly(x, sr_out // g, sr_in // g).astype(np.float32)


def load(path: Path, sr: int) -> np.ndarray:
    x, s = sf.read(str(path), dtype="float32", always_2d=False)
    return resample(mono(x), s, sr)


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x ** 2)) + 1e-9)


def pauses(dry: np.ndarray, sr: int, min_gap_s: float) -> list[tuple[int, int]]:
    hop = int(0.05 * sr)
    csum = np.concatenate([[0.0], np.cumsum(dry.astype(np.float64) ** 2)])
    idx = np.arange(len(dry))
    lo, hi = np.clip(idx - hop // 2, 0, len(dry)), np.clip(idx + hop // 2 + 1, 0, len(dry))
    env = np.sqrt((csum[hi] - csum[lo]) / np.maximum(hi - lo, 1))
    speech = env > max(np.percentile(env, 95) * 0.05, 1e-4)
    runs, start = [], None
    for i, on in enumerate(speech):
        if not on and start is None:
            start = i
        elif on and start is not None:
            if (i - start) / sr >= min_gap_s:
                runs.append((start, i))
            start = None
    if start is not None and (len(speech) - start) / sr >= min_gap_s:
        runs.append((start, len(speech)))
    return runs


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--reverb-mix", type=float, default=0.25)
    ap.add_argument("--bed-db", type=float, default=-26.0, help="background level relative to speech RMS")
    ap.add_argument("--office-db", type=float, default=-14.0, help="office sounds relative to speech RMS")
    ap.add_argument("--gap-s", type=float, default=1.5)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args(argv)

    rng = np.random.default_rng(args.seed)
    x, sr = sf.read(args.dry, dtype="float32", always_2d=False)
    dry = mono(x).astype(np.float32)
    speech_rms = rms(dry[np.abs(dry) > 1e-3]) if np.any(np.abs(dry) > 1e-3) else rms(dry)
    n = len(dry)

    ir = load(IR, sr)
    ir = ir[: int(0.4 * sr)] / (np.max(np.abs(ir)) + 1e-9)
    reverb = signal.fftconvolve(dry, ir)[:n]
    reverb *= speech_rms / (rms(reverb) + 1e-9)
    out = (1 - args.reverb_mix) * dry + args.reverb_mix * reverb

    bed = load(BED, sr)
    reps = int(np.ceil(n / len(bed)))
    bed = np.tile(bed, reps)[:n]
    bed *= speech_rms * 10 ** (args.bed_db / 20) / rms(bed)
    out += bed

    kb = load(KEYBOARD, sr)[: int(3.0 * sr)]
    mouse = load(MOUSE, sr)
    door = load(DOOR, sr)
    office_gain = speech_rms * 10 ** (args.office_db / 20)
    placed = {"keyboard": 0, "mouse": 0, "door": 0}
    for start, end in pauses(dry, sr, args.gap_s):
        length = end - start
        if rng.random() < 0.6 and length > 2.5 * sr:
            clip = kb[: min(len(kb), length // 2)]
            s = start + int(rng.uniform(0, length - len(clip)))
            out[s:s + len(clip)] += clip * office_gain / rms(clip) * 0.5
            placed["keyboard"] += 1
        elif rng.random() < 0.5 and length > 0.6 * sr:
            s = start + int(rng.uniform(0, length - len(mouse)))
            out[s:s + len(mouse)] += mouse * office_gain / rms(mouse) * 0.5
            placed["mouse"] += 1
    door_at = int(rng.uniform(1.0, 6.0) * sr)
    if door_at + len(door) < n:
        out[door_at:door_at + len(door)] += door * office_gain / rms(door) * 0.7
        placed["door"] = 1

    peak = float(np.max(np.abs(out))) or 1.0
    out = 0.95 * out / peak if peak > 0.95 else out
    sf.write(args.out, out.astype(np.float32), sr)
    print({"out": args.out, "seconds": round(n / sr, 1), "office_placed": placed}, flush=True)
    return 0
