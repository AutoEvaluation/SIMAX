#!/usr/bin/env python3
"""Transcribe the release with NVIDIA Parakeet-TDT 0.6B v2."""
import argparse, json, os, subprocess, sys, time, zlib
from pathlib import Path
REL = Path(os.environ.get("SIMAX_RELEASE", "release")); OUT = Path("runs/eval/parakeet")
MODEL = os.environ.get("PARAKEET_MODEL", "nvidia/parakeet-tdt-0.6b-v2")
NEMO = os.environ.get("PARAKEET_NEMO", "")
FFMPEG = os.environ.get("FFMPEG", "ffmpeg")
ap = argparse.ArgumentParser(); ap.add_argument("shard", type=int); ap.add_argument("nshards", type=int)
ap.add_argument("--device", default="cuda"); ap.add_argument("--out", default=str(OUT)); ap.add_argument("--limit", type=int, default=0); ap.add_argument("--max-seconds", type=float, default=0)
a = ap.parse_args(); out = Path(a.out); out.mkdir(exist_ok=True, parents=True); os.umask(0o002)
import numpy as np, torch
items = [(p, d.name) for p in ("Public-Short", "Internal-Extended") for d in sorted((REL / p).glob("*")) if (d / "dry.mp3").exists()]
items = [x for x in items if zlib.crc32(f"{x[0]}/{x[1]}".encode()) % a.nshards == a.shard]
if a.limit: items = items[:a.limit]
f = out / f"hyp_{a.shard:03d}.jsonl"; done = set()
if f.exists():
    for l in open(f):
        if l.startswith("{"): done.add(json.loads(l)["key"])
todo = [x for x in items if f"{x[0]}/{x[1]}" not in done]
if todo:
    import nemo.collections.asr as nemo_asr
    import logging; logging.getLogger("nemo_logger").setLevel(logging.ERROR)
    dev = a.device if (a.device == "cpu" or torch.cuda.is_available()) else "cpu"
    model = (nemo_asr.models.ASRModel.restore_from(NEMO, map_location=dev) if NEMO and os.path.exists(NEMO) else nemo_asr.models.ASRModel.from_pretrained(MODEL, map_location=dev)).eval()
    model.change_attention_model(self_attention_model="rel_pos_local_attn", att_context_size=[256, 256])
    model.change_subsampling_conv_chunking_factor(1)
    if dev == "cpu": torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
t0 = time.time()
with open(f, "a") as fo:
    for n, (p, i) in enumerate(todo):
        key = f"{p}/{i}"
        raw = subprocess.run([FFMPEG, "-v", "error", "-i", str(REL / p / i / "dry.mp3"), "-ac", "1", "-ar", "16000", "-f", "f32le", "-"], capture_output=True, check=True).stdout
        audio = np.frombuffer(raw, dtype=np.float32).copy()
        if a.max_seconds: audio = audio[: int(a.max_seconds * 16000)]
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=(dev != "cpu")):
            r = model.transcribe([audio], batch_size=1, verbose=False)
        h = r[0]; text = h.text if hasattr(h, "text") else str(h)
        fo.write(json.dumps({"key": key, "part": p, "id": i, "dur_s": round(len(audio) / 16000, 1), "hyp": text.strip(), "segments": [], "asr": MODEL}, ensure_ascii=False) + "\n"); fo.flush()
        print(n + 1, "/", len(todo), key, f"{time.time() - t0:.0f}s", flush=True)
        if dev != "cpu": torch.cuda.empty_cache()
print("PARAKEET_SHARD_DONE")
