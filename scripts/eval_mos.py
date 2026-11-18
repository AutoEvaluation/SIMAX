#!/usr/bin/env python3
"""Compute UTMOSv2 or WV-MOS on 10-second windows of the dry audio."""
import sys, os, json, glob, time, zlib, argparse
import numpy as np, librosa
ap = argparse.ArgumentParser()
ap.add_argument("metric"); ap.add_argument("shard", type=int); ap.add_argument("nshards", type=int); ap.add_argument("out")
ap.add_argument("--kinds", default="dry"); ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--whole", type=int, default=1); ap.add_argument("--win", type=float, default=10.0)
ap.add_argument("--device", default="cpu"); ap.add_argument("--files", default=None); ap.add_argument("--max-win", type=int, default=0); ap.add_argument("--ids-file", default=None); ap.add_argument("--root", default=os.environ.get("SIMAX_RELEASE", "release"))
ap.add_argument("--sub", type=int, default=0); ap.add_argument("--nsub", type=int, default=1)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
items = []
if a.files:
    for l in open(a.files):
        k, p, g = l.rstrip("\n").split("\t"); items.append((g, k, "ref", p))
for part in ([] if a.files else ("Public-Short", "Internal-Extended")):
    for d in sorted(glob.glob(f"{a.root}/{part}/*/")):
        for k in a.kinds.split(","):
            items.append((part, os.path.basename(d.rstrip("/")), k, f"{d}{k}.mp3"))
if a.ids_file:
    _ids = {l.strip() for l in open(a.ids_file) if l.strip()}
    items = [i for i in items if f"{i[0]}/{i[1]}" in _ids]
items = [i for i in items if zlib.crc32(f"{i[1]}/{i[2]}".encode()) % a.nshards == a.shard]
if a.limit: items = items[:a.limit]
items = items[a.sub::a.nsub]
if a.metric == "utmos":
    import utmosv2
    model = utmosv2.create_model(pretrained=True, device=a.device)
    def score_many(ws, bs=16): return [float(model.predict(data=w, sr=16000, device=a.device, num_workers=0, verbose=False)) for w in ws]
else:
    from wvmos import get_wvmos
    import torch
    model = get_wvmos(cuda=(a.device != "cpu"))
    def score_many(ws, bs=16):
        out = []
        for i in range(0, len(ws), bs):
            b = ws[i:i+bs]
            if len(set(len(w) for w in b)) > 1: out += [x for w in b for x in score_many([w], 1)]; continue
            x = model.processor(list(b), return_tensors="pt", padding=True, sampling_rate=16000).input_values
            if a.device != "cpu": x = x.cuda()
            with torch.no_grad(): out += model.forward(x).flatten().float().cpu().tolist()
        return out
outf = f"{a.out}/{a.metric}_{a.shard:03d}.jsonl" if a.nsub == 1 else f"{a.out}/{a.metric}_{a.shard:03d}_{a.sub}.jsonl"
done = set()
for g in glob.glob(f"{a.out}/{a.metric}_{a.shard:03d}*.jsonl"):
    for l in open(g):
        try: done.add(json.loads(l)["key"])
        except ValueError: pass
fo = open(outf, "a"); t0 = time.time()
for n, (part, did, kind, p) in enumerate(items):
    key = f"{part}/{did}/{kind}" if not a.files else did
    if key in done: continue
    if p.endswith(".m4a"):
        import subprocess
        y = np.frombuffer(subprocess.run([os.environ.get("FFMPEG", "ffmpeg"), "-v", "error", "-i", p, "-ac", "1", "-ar", "16000", "-f", "f32le", "-"], capture_output=True, check=True).stdout, dtype=np.float32)
    else:
        y, _ = librosa.load(p, sr=16000)
    W = int(a.win * 16000)
    wins = [y[i:i + W] for i in range(0, len(y) - W // 2, W)]
    if not wins: wins = [y]
    if a.max_win and len(wins) > a.max_win: wins = [wins[int(i)] for i in np.linspace(0, len(wins) - 1, a.max_win)]
    ws = score_many([w.astype(np.float32) for w in wins])
    rec = dict(key=key, part=part, id=(did if not a.files else key), kind=kind, dur_s=len(y) / 16000, n_win=len(ws), win_scores=[round(s, 3) for s in ws])
    if a.whole and a.metric == "utmos" and zlib.crc32(key.encode()) % 7 == 0: rec["whole"] = score_many([y.astype(np.float32)], 1)[0]
    fo.write(json.dumps(rec) + "\n"); fo.flush()
    print(n, key, f"{time.time()-t0:.0f}s", flush=True)
