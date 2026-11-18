#!/usr/bin/env python3
"""Compute CLAP text-audio similarity for the release."""
import argparse, os, csv, json, re, statistics, sys, zlib
from pathlib import Path
REL = Path(os.environ.get("SIMAX_RELEASE", "release")); OUT = Path("runs/eval/clap"); PARTS = ("Public-Short", "Internal-Extended")

def text_of(d):
    t = (d / "transcript.txt").read_text(encoding="utf-8", errors="ignore")
    t = re.sub(r"\[\d+\]\s*", "", t)
    return " ".join(t.split())

def cmd_score(a):
    global OUT
    OUT = Path(a.out) if a.out else (OUT.with_name("out_" + a.window) if a.window != "rand" else OUT)
    import numpy as np, torch, librosa
    from transformers import ClapModel, ClapProcessor
    name = "laion/clap-htsat-unfused"; dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ClapModel.from_pretrained(name).to(dev).eval(); proc = ClapProcessor.from_pretrained(name); sr = int(proc.feature_extractor.sampling_rate)
    items = [(p, d) for p in PARTS for d in sorted((REL / p).glob("*")) if (d / "dry.mp3").exists()]
    items = [x for n, x in enumerate(items) if n % a.num_shards == a.shard]; OUT.mkdir(exist_ok=True, parents=True)
    with open(OUT / f"scores_{a.shard}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, ["part", "id", "codebook", "label", "target", "cosine", "audio_s", "error"]); w.writeheader()
        for n, (p, d) in enumerate(items):
            m = json.load(open(d / "metadata.json")); tg = m["target"]; tg = tg.get("value") if isinstance(tg, dict) else tg
            row = dict(part=p, id=d.name, codebook=m["codebook"], label=m["label"], target=tg, cosine="", audio_s="", error="")
            try:
                np.random.seed(zlib.crc32(d.name.encode()) & 0x7fffffff)
                au, _ = librosa.load(str(d / "dry.mp3"), sr=sr, mono=True, duration=(10.0 if a.window == "head" else None)); row["audio_s"] = f"{len(au) / sr:.1f}"
                ia = proc.feature_extractor([au], sampling_rate=sr, return_tensors="pt", padding=True, **({"truncation": "truncate"} if a.window == "head" else {}))
                it = proc.tokenizer([text_of(d)], return_tensors="pt", padding=True, truncation=True)
                ia = {k: v.to(dev) for k, v in ia.items()}; it = {k: v.to(dev) for k, v in it.items()}
                with torch.no_grad():
                    te = model.text_projection(model.text_model(**it).pooler_output); ae = model.audio_projection(model.audio_model(**ia).pooler_output)
                    row["cosine"] = f"{float(torch.sum(torch.nn.functional.normalize(te, dim=-1) * torch.nn.functional.normalize(ae, dim=-1)).item()):.8f}"
            except Exception as e:
                row["error"] = str(e)[:200]
            w.writerow(row); f.flush()
            if (n + 1) % 25 == 0: print(f"shard {a.shard}: {n + 1}/{len(items)}", flush=True)

def cmd_summarize(a):
    global OUT
    OUT = Path(a.out) if a.out else (OUT.with_name("out_" + a.window) if a.window != "rand" else OUT)
    rows = []
    for f in sorted(OUT.glob("scores_*.csv")): rows += list(csv.DictReader(open(f)))
    ok = [r for r in rows if r["cosine"]]
    with open(OUT / "clap_scores.csv", "w", newline="") as f:
        w = csv.DictWriter(f, list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    def fam(r): return "Global" if r["codebook"] == "Global" else "WISER-family"
    groups = {"overall": lambda r: "all", "part": lambda r: r["part"], "batch (Global vs rest)": fam, "codebook": lambda r: r["codebook"], "part x batch": lambda r: f"{r['part']} / {fam(r)}", "target": lambda r: f"{r['codebook']} / {r['target']}"}
    with open(OUT / "clap_summary.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["group", "value", "n", "mean", "std", "min", "max"])
        for g, key in groups.items():
            by = {}
            for r in ok: by.setdefault(key(r), []).append(float(r["cosine"]))
            for k in sorted(by, key=str):
                v = by[k]; w.writerow([g, k, len(v), f"{statistics.mean(v):.4f}", f"{statistics.stdev(v) if len(v) > 1 else 0:.4f}", f"{min(v):.4f}", f"{max(v):.4f}"])
    print(f"dialogues {len(rows)}, scored {len(ok)}, errors {len(rows) - len(ok)}")
    v = [float(r["cosine"]) for r in ok]; print(f"overall mean {statistics.mean(v):.4f} std {statistics.stdev(v):.4f}")
    print((OUT / "clap_summary.csv").read_text())

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["score", "summarize"]); ap.add_argument("--shard", type=int, default=0); ap.add_argument("--num-shards", type=int, default=1); ap.add_argument("--window", choices=["rand", "head"], default="rand"); ap.add_argument("--out", default=None)
    a = ap.parse_args(); {"score": cmd_score, "summarize": cmd_summarize}[a.cmd](a)
