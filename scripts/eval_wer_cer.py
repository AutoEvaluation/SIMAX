#!/usr/bin/env python3
"""Compute WER and CER of ASR transcripts against the source text."""
import csv, glob, json, os, re, statistics, sys
from pathlib import Path
import jiwer
HYP = Path(sys.argv[1]); OUTD = Path(sys.argv[2]); OUTD.mkdir(parents=True, exist_ok=True); REL = Path(os.environ.get("SIMAX_RELEASE", "release"))
def norm(t): return " ".join(re.sub(r"[^a-z0-9\s]", "", t.lower()).split())
def ref_of(d):
    t = (d / "transcript.txt").read_text(encoding="utf-8", errors="ignore")
    t = re.sub(r"\[\d+\]\s*", "", t); t = re.sub(r"\b(Doctor|Patient):\s*", "", t)
    return norm(t)
rows = []; seen = set()
for f in sorted(glob.glob(str(HYP / "hyp_*.jsonl"))):
    for l in open(f):
        if not l.startswith("{"): continue
        try: r = json.loads(l)
        except ValueError: continue
        if r["key"] in seen: continue
        seen.add(r["key"]); d = REL / r["part"] / r["id"]; ref, hyp = ref_of(d), norm(r["hyp"])
        m = json.load(open(d / "metadata.json"))
        rows.append(dict(part=r["part"], id=r["id"], codebook=m["codebook"], label=m["label"], dur_s=r["dur_s"], ref_words=len(ref.split()), hyp_words=len(hyp.split()),
                         wer=jiwer.wer(ref, hyp) if hyp else 1.0, cer=jiwer.cer(ref, hyp) if hyp else 1.0))
with open(OUTD / "wer_cer_scores.csv", "w", newline="") as f:
    w = csv.DictWriter(f, list(rows[0])); w.writeheader(); w.writerows(rows)
def fam(r): return "Global" if r["codebook"] == "Global" else "WISER-family"
with open(OUTD / "wer_cer_summary.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["group", "value", "n", "wer_mean", "wer_median", "wer_std", "wer_max", "cer_mean", "cer_median", "cer_std", "cer_max"])
    for g, key in {"overall": lambda r: "all", "part": lambda r: r["part"], "batch": fam, "codebook": lambda r: r["codebook"]}.items():
        by = {}
        for r in rows: by.setdefault(key(r), []).append(r)
        for k, v in sorted(by.items()):
            wv, cv = [x["wer"] for x in v], [x["cer"] for x in v]
            w.writerow([g, k, len(v), *(f"{fn(x):.4f}" for x in (wv, cv) for fn in (statistics.mean, statistics.median, lambda z: statistics.stdev(z) if len(z) > 1 else 0, max))][:1] if False else [g, k, len(v)] + [f"{fn(x):.4f}" for x in (wv, cv) for fn in (statistics.mean, statistics.median, (lambda z: statistics.stdev(z) if len(z) > 1 else 0.0), max)])
print(open(OUTD / "wer_cer_summary.csv").read()); print("dialogues", len(rows), "WER>1 (hallucination?):", sum(r["wer"] > 1 for r in rows))
