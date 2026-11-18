#!/usr/bin/env python3
"""Synthesize dialogues, check every chunk and regenerate failed chunks."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from simax.audio.common import dialogue_spec, prepare_folder, rebuild_full
from simax.audio.qc import QC, verdict
from simax.audio.tts import TTS

CHUNK_FILES = ("wav", "qc.json", "qcemb.npy", "words.json", "attempts.json")


def drop_chunk(folder: Path, k: int, keep_in: Path | None = None) -> None:
    for suffix in CHUNK_FILES:
        path = folder / f"chunk_{k:02d}.{suffix}"
        if path.exists():
            if keep_in is not None:
                keep_in.mkdir(exist_ok=True)
                shutil.move(str(path), keep_in / path.name)
            else:
                path.unlink()


def repair(folder: Path, tts: TTS, qc: QC, max_redos: int) -> dict:
    tts.fill_folder(folder)
    _, result = qc.check_dialogue(folder)
    redos: dict[int, int] = {}
    for _ in range(max_redos):
        failed = [c["k"] for c in result["chunks"] if c["status"] == "fail" and redos.get(c["k"], 0) < max_redos]
        if not failed:
            break
        for k in failed:
            redos[k] = redos.get(k, 0) + 1
            drop_chunk(folder, k)
        tts.fill_folder(folder)
        _, result = qc.check_dialogue(folder)
    for chunk in [c for c in result["chunks"] if c["status"] == "suspect"]:
        k, old = chunk["k"], folder / "previous"
        drop_chunk(folder, k, keep_in=old)
        tts.fill_folder(folder)
        _, result = qc.check_dialogue(folder)
        if next(c for c in result["chunks"] if c["k"] == k)["status"] != "pass":
            drop_chunk(folder, k)
            for path in old.glob(f"chunk_{k:02d}.*"):
                shutil.move(str(path), folder / path.name)
            rebuild_full(folder)
            result = verdict(folder)
        shutil.rmtree(old, ignore_errors=True)
    return result


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--max-redos", type=int, default=6)
    args = ap.parse_args(argv)
    rows = [json.loads(line) for line in open(args.input, encoding="utf-8")][args.shard::args.num_shards]
    tts, qc = TTS(), QC()
    out = Path(args.output)
    for row in rows:
        folder = prepare_folder(row, out / row["part"])
        result = repair(folder, tts, qc, args.max_redos)
        bad = [c for c in result["chunks"] if c["status"] == "fail"]
        print(f"{row['sample_id']}: {len(result['chunks']) - len(bad)}/{len(result['chunks'])} chunks passed", flush=True)
        if bad:
            with open(out / "needs_review.jsonl", "a", encoding="utf-8") as fh:
                fh.write(json.dumps({"sample_id": dialogue_spec(folder)["sample_id"], "chunks": bad}, ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
