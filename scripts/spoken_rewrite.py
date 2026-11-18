#!/usr/bin/env python3
"""Rewrite generated dialogues as word-for-word speech."""

from __future__ import annotations

import argparse
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for extra in (ROOT / "src", ROOT / "scripts"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from generate_dialogues import write_summary

from simax.generation.generator import (
    GenerationConfig,
    OpenAIChatClient,
    UsageLedger,
    _extract_json_object,
    git_commit_info,
)
from simax.generation.long import (
    SPOKEN_TEMPLATE_VERSION,
    check_long_dialogue,
    check_rewritten_chunk,
    compare_to_profile,
    compile_spoken_rewrite_prompt,
    dialogue_stats,
    load_profile,
    split_into_chunks,
)
from simax.jsonl_io import utc_now_iso


def parse_chunk(text: str) -> list[dict[str, str]]:
    payload = json.loads(_extract_json_object(text))
    turns = payload["turns"] if isinstance(payload, dict) else payload
    out = []
    for raw in turns:
        speaker, body = str(raw["speaker"]).strip().lower(), str(raw["text"]).strip()
        if speaker not in ("doctor", "patient") or not body:
            raise ValueError(f"bad turn: {raw!r}")
        out.append({"speaker": speaker, "text": body})
    return out


def usd(paths: list[str], budget: dict[str, Any]) -> float:
    total = 0.0
    for path in paths:
        for line in open(path, encoding="utf-8"):
            row = json.loads(line)
            total += ((row.get("prompt_tokens") or 0) * budget["usd_per_1m_input_tokens"]
                      + (row.get("completion_tokens") or 0) * budget["usd_per_1m_output_tokens"])
    return budget["price_safety_factor"] * total / 1e6


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--generation", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--config", default=str(ROOT / "configs" / "generation.json"))
    parser.add_argument("--select", default=None, help="comma-separated label:target subset")
    parser.add_argument("--chunk-turns", type=int, default=30)
    parser.add_argument("--max-tokens", type=int, default=12000)
    parser.add_argument("--workers", type=int, default=60)
    args = parser.parse_args(argv)

    config = json.loads(Path(args.config).read_text())
    budget = config["budget"]
    profile = load_profile(ROOT / config["design"]["profile"])
    gen_cfg = GenerationConfig.from_config(config)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    gen_path = out / "generation.jsonl"
    ledger = UsageLedger(out / "token_usage.jsonl")
    client = OpenAIChatClient(usage_ledger=ledger, timeout_s=600.0, call_timeout_s=600.0)
    commit, dirty = git_commit_info()

    wanted = None
    if args.select:
        wanted = {tuple(item.strip().split(":")) for item in args.select.split(",")}
    done = set()
    if gen_path.exists():
        done = {json.loads(line)["sample_id"] for line in open(gen_path, encoding="utf-8")}
    records = [r for r in map(json.loads, open(args.generation, encoding="utf-8"))
               if r["status"] == "ok" and r["sample_id"] not in done
               and (wanted is None or (r["focal_construct"], str(r["target_level"])) in wanted)]

    lock = threading.Lock()
    state = {"reserved": 0.0, "stopped": False}

    def all_ledgers() -> list[str]:
        return [str(ledger.path)] if ledger.path.exists() else []

    def rewrite_chunk(record: dict, chunk: list[dict]) -> tuple[list[dict], bool]:
        prompt = compile_spoken_rewrite_prompt(chunk, profile)
        reserve = 2 * budget["price_safety_factor"] * (
            len(prompt.user) / 3 * budget["usd_per_1m_input_tokens"]
            + args.max_tokens * budget["usd_per_1m_output_tokens"]) / 1e6
        with lock:
            if state["stopped"]:
                return chunk, False
            spent = usd(all_ledgers(), budget)
            if spent + state["reserved"] + reserve > budget["limit_usd"]:
                state["stopped"] = True
                print(f"BUDGET STOP at ${spent:.4f} (cap ${budget['limit_usd']:.2f})", flush=True)
                return chunk, False
            state["reserved"] += reserve
        try:
            for _ in range(2):
                reply = client.chat(
                    system=prompt.system, user=prompt.user, model=gen_cfg.model,
                    temperature=gen_cfg.temperature, seed=gen_cfg.generation_seed,
                    max_tokens=args.max_tokens,
                    context={"sample_id": record["sample_id"], "method": "spoken-rewrite",
                             "construct": record["focal_construct"],
                             "target_level": record["target_level"]})
                try:
                    rewritten = parse_chunk(reply.content)
                except (ValueError, KeyError, TypeError):
                    continue
                if not check_rewritten_chunk(chunk, rewritten):
                    return rewritten, True
            return chunk, False
        finally:
            with lock:
                state["reserved"] -= reserve

    def rewrite_dialogue(record: dict) -> None:
        chunks = split_into_chunks(record["dialogue"]["turns"], chunk_turns=args.chunk_turns)
        with ThreadPoolExecutor(max_workers=len(chunks)) as inner:
            results = list(inner.map(lambda c: rewrite_chunk(record, c), chunks))
        if state["stopped"]:
            print(f"{record['focal_construct']} {record['target_level']}: not written "
                  "(budget stop)", flush=True)
            return
        turns = [{"turn_index": i, "speaker": t["speaker"], "text": t["text"]}
                 for i, t in enumerate(t for chunk, _ in results for t in chunk)]
        stats = dialogue_stats(turns)
        loose_plan = {**record["plan"], "min_turns": 0, "max_turns": 10**6}
        new = {**record}
        new.update(
            dialogue={"turns": turns}, turn_count=len(turns),
            stats={k: round(float(v), 4) for k, v in stats.items()},
            profile_comparison=compare_to_profile(stats, profile),
            contract_problems=check_long_dialogue(turns, loose_plan),
            rewrite={"template_version": SPOKEN_TEMPLATE_VERSION, "model": gen_cfg.model,
                     "chunks": len(chunks), "chunks_rewritten": sum(ok for _, ok in results),
                     "source_generation": str(args.generation),
                     "source_turn_count": record["turn_count"],
                     "source_stats": record["stats"],
                     "git_commit": commit, "git_dirty": dirty},
            created_at=utc_now_iso(),
        )
        with lock:
            with open(gen_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(new, ensure_ascii=False) + "\n")
            print(f"{record['focal_construct']} {record['target_level']}: "
                  f"{new['rewrite']['chunks_rewritten']}/{len(chunks)} chunks rewritten, "
                  f"{record['turn_count']} -> {len(turns)} turns, fillers/100w "
                  f"{record['stats']['fillers_per_100_words']:.1f} -> "
                  f"{stats['fillers_per_100_words']:.1f}, short turns "
                  f"{record['stats']['short_turn_fraction']:.2f} -> "
                  f"{stats['short_turn_fraction']:.2f}", flush=True)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for future in [pool.submit(rewrite_dialogue, r) for r in records]:
            future.result()

    if gen_path.exists():
        write_summary(out, config, ledger)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
