#!/usr/bin/env python3
"""Transcribe the release with Whisper medium."""
import argparse, json, os, sys, time, zlib
from pathlib import Path
REL = Path(os.environ.get("SIMAX_RELEASE", "release")); OUT = Path("runs/eval/whisper")
ap = argparse.ArgumentParser(); ap.add_argument("shard", type=int); ap.add_argument("nshards", type=int)
ap.add_argument("--limit", type=int, default=0); ap.add_argument("--max-seconds", type=float, default=0); ap.add_argument("--device", default="cuda"); ap.add_argument("--out", default=str(OUT))
ap.add_argument("--par", type=int, default=1, help="run this many processes on the same GPU (each takes every par-th dialogue of the shard)")
ap.add_argument("--sub", type=int, default=0); ap.add_argument("--nsub", type=int, default=1)
a = ap.parse_args(); out = Path(a.out); out.mkdir(exist_ok=True, parents=True); os.umask(0o002)
if a.par > 1:
    import subprocess
    base = [sys.executable, os.path.abspath(__file__), str(a.shard), str(a.nshards), "--device", a.device, "--out", str(out), "--nsub", str(a.par)]
    if a.limit: base += ["--limit", str(a.limit)]
    if a.max_seconds: base += ["--max-seconds", str(a.max_seconds)]
    procs = [subprocess.Popen(base + ["--sub", str(k)]) for k in range(a.par)]
    rc = [q.wait() for q in procs]
    if any(rc): sys.exit(max(rc))
    print("WHISPER_SHARD_DONE"); sys.exit(0)
import whisper, numpy as np
items = [(p, d.name) for p in ("Public-Short", "Internal-Extended") for d in sorted((REL / p).glob("*")) if (d / "dry.mp3").exists()]
items = [x for x in items if zlib.crc32(f"{x[0]}/{x[1]}".encode()) % a.nshards == a.shard]
if a.limit: items = items[:a.limit]
items = items[a.sub::a.nsub]
f = out / (f"hyp_{a.shard:03d}.jsonl" if a.nsub == 1 else f"hyp_{a.shard:03d}_{a.sub}.jsonl"); done = set()
for g in out.glob(f"hyp_{a.shard:03d}*.jsonl"):
    for l in open(g):
        if l.startswith("{"):
            try: done.add(json.loads(l)["key"])
            except ValueError: pass
items = [x for x in items if f"{x[0]}/{x[1]}" not in done]
model = whisper.load_model(os.environ.get("WHISPER_MODEL", "medium"), device=a.device); t0 = time.time()
with open(f, "a") as fo:
    for n, (p, i) in enumerate(items):
        key = f"{p}/{i}"
        if key in done: continue
        audio = whisper.load_audio(str(REL / p / i / "dry.mp3"))
        if a.max_seconds: audio = audio[: int(a.max_seconds * 16000)]
        r = model.transcribe(audio, language="en", fp16=(a.device != "cpu"), verbose=False)
        fo.write(json.dumps({"key": key, "part": p, "id": i, "dur_s": round(len(audio) / 16000, 1), "hyp": r["text"].strip(), "segments": [[round(x["start"], 2), round(x["end"], 2), x["text"].strip()] for x in r["segments"]]}, ensure_ascii=False) + "\n"); fo.flush()
        print(n + 1, "/", len(items), key, f"{time.time() - t0:.0f}s", flush=True)
print("WHISPER_SHARD_DONE")
