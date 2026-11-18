#!/usr/bin/env python3
"""Fill cases and generate the full-visit dialogues."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from simax.generation.generator import (
    GenerationConfig,
    OpenAIChatClient,
    UsageLedger,
    generate_dialogue,
    git_commit_info,
)
from simax.generation.long import (
    LONG_METHOD,
    check_long_dialogue,
    compare_to_profile,
    compile_long_prompt,
    dialogue_stats,
    load_profile,
    load_topic_profile,
    plan_long_dialogue,
    plan_small_talk,
    plan_visit_mix,
    render_long_count_block,
)
from simax.generation.short import render_short_scale_block
from simax.generation.case_template import render_people, resolve_template
from simax.jsonl_io import utc_now_iso

SCHEMA_VERSION = "simax-dialogue-v1"


def _id(*parts: Any) -> str:
    return hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:16]


def build_blocks(config: dict[str, Any]) -> list[dict[str, Any]]:
    design = config["design"]
    plan = json.loads((ROOT / design["persona_plan"]).read_text())["blocks"]
    wiser = design["wiser"]
    labels = [("global", name, "global", {}) for name in design["constructs"]]
    labels += [("wiser", name, "count", wiser) for name in wiser["labels"]]
    for family, settings in design["count_families"].items():
        labels += [(family, name, "count", settings) for name in settings["labels"]]
    requirements = design.get("case_requirements") or {}

    blocks = []
    for family, label, kind, settings in labels:
        for block_id, entry in plan.items():
            if entry["label"] != label:
                continue
            template = {"patient": dict(entry["patient"]), "clinician": dict(entry["clinician"]),
                        "case": {"department": entry["department"], **entry["case"]}}
            blocks.append({"block_id": block_id, "family": family, "label": label, "kind": kind,
                           "settings": settings, "template": template,
                           "case_seed": int(_id("long-case", block_id), 16) % (2**31),
                           "requirements": list(requirements.get(label, [])) +
                                           list(requirements.get(family, []))})
    return blocks


def resolve_cases(blocks, out: Path, complete) -> dict[str, dict[str, Any]]:
    path = out / "resolved_cases.jsonl"
    cache = {}
    if path.exists():
        cache = {r["block_id"]: r["resolved"] for r in map(json.loads, open(path, encoding="utf-8"))}
    def one(block):
        try:
            resolved = resolve_template(block["template"], seed=block["case_seed"],
                                        requirements=block["requirements"], complete=complete,
                                        max_attempts=5)
        except ValueError as exc:
            print(f"case fill failed for {block['label']} {block['block_id']}: {exc}", flush=True)
            return block, None
        case = resolved["case"]
        case.setdefault("case_id", f"{case['department'][:5]}-{block['block_id'][:8]}")
        return block, resolved

    todo = [b for b in blocks if b["block_id"] not in cache]
    with ThreadPoolExecutor(max_workers=8) as pool:
        for block, resolved in pool.map(one, todo):
            if resolved is None:
                continue
            cache[block["block_id"]] = resolved
            with open(path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps({"block_id": block["block_id"], "label": block["label"],
                                     "resolved": resolved}, ensure_ascii=False) + "\n")
    return cache


def build_tasks(config: dict[str, Any], blocks, resolved: dict[str, dict]) -> list[dict[str, Any]]:
    design = config["design"]
    rubric = json.loads((ROOT / design["rubric"]).read_text())
    profile = load_profile(ROOT / design["profile"])
    topic_profile = load_topic_profile(ROOT / design["topic_profile"]) if design.get("topic_profile") else None
    specs = json.loads((ROOT / design["label_specs"]).read_text())["families"]
    tasks = []
    for block in blocks:
        if block["block_id"] not in resolved:
            continue
        family, label, kind, settings = block["family"], block["label"], block["kind"], block["settings"]
        block_id, full = block["block_id"], resolved[block["block_id"]]
        case, people = full["case"], render_people(full)
        persona_seed = int(_id("long-persona", block_id), 16) % (2**31)
        spec = dict(specs[family]["labels"].get(label, {}))
        if settings.get("definitions", {}).get(label):
            spec["definition"] = settings["definitions"][label]
        notes = (design.get("label_notes") or {}).get(label) or {}
        spec["notes"] = notes
        targets = design["levels"] if kind == "global" else settings["bands"]
        for rank, target in enumerate(targets, start=1):
            sample_id = _id("long-sample", block_id, target)
            plan = plan_long_dialogue(
                sample_id=sample_id, block_id=block_id, case=case, focal_construct=label,
                target_level=rank, persona_seed=persona_seed, profile=profile,
            ).to_dict()
            if topic_profile is not None:
                plan["small_talk"] = plan_small_talk(plan, topic_profile=topic_profile,
                                                     patient_age=full["patient"].get("age"))
                plan["visit_mix"] = plan_visit_mix(plan, topic_profile=topic_profile)
                if topic_profile.get("low_arm_guard") and str(plan["target_level"]) == "low":
                    plan["low_arm_guard"] = True
            common = {"case": case, "people": people, "plan": plan, "profile": profile}
            target_count = None
            if kind == "global":
                prompt = compile_long_prompt(focal_construct=label, target_level=target,
                                             rubric=rubric, **common)
            else:
                counts = settings["high_count"]
                target_count = counts.get(label, counts["default"]) if target == "high" else None
                prompt = compile_long_prompt(behavior=render_long_count_block(
                    spec, band=target, target_count=target_count or 0), **common)
            tasks.append({"sample_id": sample_id, "block_id": block_id, "case": case,
                          "persona": {k: full[k] for k in ("patient", "clinician")},
                          "case_provenance": full.get("provenance", {}),
                          "persona_seed": persona_seed, "family": family,
                          "construct": label, "level": target, "rank": rank,
                          "target_count": target_count, "plan": plan, "prompt": prompt,
                          "profile": profile})
    return tasks


def spent_usd(ledger: UsageLedger, budget: dict[str, Any]) -> float:
    total = 0.0
    if ledger.path.exists():
        for line in open(ledger.path, encoding="utf-8"):
            row = json.loads(line)
            total += ((row.get("prompt_tokens") or 0) * budget["usd_per_1m_input_tokens"]
                      + (row.get("completion_tokens") or 0) * budget["usd_per_1m_output_tokens"])
    return budget["price_safety_factor"] * total / 1e6


def worst_case_usd(prompt_chars: int, max_tokens: int, budget: dict[str, Any]) -> float:
    return budget["price_safety_factor"] * (
        prompt_chars / 3 * budget["usd_per_1m_input_tokens"]
        + max_tokens * budget["usd_per_1m_output_tokens"]
    ) / 1e6


def write_summary(out: Path, config: dict[str, Any], ledger: UsageLedger) -> None:
    records = [json.loads(line) for line in open(out / "generation.jsonl", encoding="utf-8")]
    records.sort(key=lambda r: (r["label_family"], r["focal_construct"], r["target_rank"]))
    budget = config["budget"]
    ledger.reload_from_disk()
    totals = ledger.totals()["overall"]
    list_usd = spent_usd(ledger, budget) / budget["price_safety_factor"]
    ok = [r for r in records if r["status"] == "ok"]
    metrics = list(ok[0]["profile_comparison"]) if ok else []
    lines = [
        "# SIMAX dialogues",
        "",
        f"- model: {config['generation']['model']} (default reasoning), {len(ok)}/{len(records)} dialogues ok",
        f"- tokens: {totals['prompt_tokens']} prompt + {totals['completion_tokens']} completion "
        f"({totals['reasoning_tokens']} reasoning) over {totals['calls']} calls",
        f"- spend: ${list_usd:.4f} at list price; ${list_usd * budget['price_safety_factor']:.4f} "
        f"as counted against the ${budget['limit_usd']:.2f} cap (x{budget['price_safety_factor']:g} safety factor)",
        "",
        "Each cell is the dialogue's value; `*` marks a value outside the profile q10-q90 band.",
        "",
        "| label | target | planned turns | " + " | ".join(metrics) + " | contract |",
        "|---|---|---|" + "---|" * (len(metrics) + 1),
    ]
    if ok:
        band = ok[0]["profile_comparison"]
        lines.append("| *profile q10-q90* | | | " + " | ".join(
            f"{band[m]['q10']}-{band[m]['q90']}" for m in metrics) + " | |")
        lines.append("| *profile median* | | | " + " | ".join(
            str(band[m]["median"]) for m in metrics) + " | |")
    for rec in records:
        if rec["status"] != "ok":
            lines.append(f"| {rec['focal_construct']} | {rec['target_level']} | "
                         f"{rec['plan']['turn_count']} | " + " | ".join("-" for _ in metrics)
                         + f" | {rec['status']} |")
            continue
        cmp_ = rec["profile_comparison"]
        cells = [f"{cmp_[m]['value']:g}{'' if cmp_[m]['band'] == 'in' else '*'}" for m in metrics]
        lines.append(f"| {rec['focal_construct']} | {rec['target_level']} | "
                     f"{rec['plan']['turn_count']} | " + " | ".join(cells)
                     + f" | {'ok' if not rec['contract_problems'] else '; '.join(rec['contract_problems'])} |")
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(ROOT / "configs" / "generation.json"))
    parser.add_argument("--out-dir", default=str(ROOT / "runs" / "dialogues"))
    parser.add_argument("--select", default=None,
                        help="comma-separated labels (all targets) or label:target pairs")
    parser.add_argument("--limit", type=int, default=None, help="generate at most N new dialogues")
    parser.add_argument("--dry-run", action="store_true",
                        help="compile prompts and write them out; no provider calls")
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args(argv)

    config = json.loads(Path(args.config).read_text())
    if config["generation"].get("reasoning_effort"):
        os.environ["SIMAX_REASONING_EFFORT"] = config["generation"]["reasoning_effort"]
    budget = config["budget"]
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ledger = UsageLedger(out / "token_usage.jsonl")
    if args.summary_only:
        write_summary(out, config, ledger)
        return 0

    blocks = build_blocks(config)
    if args.select:
        wanted_labels = {item.strip().split(":")[0] for item in args.select.split(",")}
        blocks = [b for b in blocks if b["label"] in wanted_labels]
    gen_cfg = GenerationConfig.from_config(config)
    client = OpenAIChatClient(usage_ledger=ledger, timeout_s=1200.0, call_timeout_s=1200.0)

    def complete(system: str, user: str) -> str:
        return client.chat(system=system, user=user, model=gen_cfg.model,
                           temperature=gen_cfg.temperature, seed=gen_cfg.generation_seed,
                           max_tokens=8000, context={"method": "case-fill"}).content

    resolved = resolve_cases(blocks, out, complete)
    tasks = build_tasks(config, blocks, resolved)
    if args.dry_run:
        with open(out / "prompts_dry_run.jsonl", "w", encoding="utf-8") as fh:
            for task in tasks:
                fh.write(json.dumps({"sample_id": task["sample_id"], "construct": task["construct"],
                                     "level": task["level"], **task["prompt"].to_dict()}) + "\n")
        print(tasks[0]["prompt"].user)
        print(f"\n{len(tasks)} prompts written to {out / 'prompts_dry_run.jsonl'}")
        return 0

    gen_path = out / "generation.jsonl"
    done = set()
    if gen_path.exists():
        done = {json.loads(line)["sample_id"] for line in open(gen_path, encoding="utf-8")}
    commit, dirty = git_commit_info()
    lock = threading.Lock()
    state = {"reserved": 0.0, "stopped": False}
    todo = [task for task in tasks if task["sample_id"] not in done]
    if args.select:
        wanted = {item.strip() for item in args.select.split(",")}
        todo = [t for t in todo if t["construct"] in wanted or f"{t['construct']}:{t['level']}" in wanted]
    if args.limit is not None:
        todo = todo[: args.limit]

    def run_one(task: dict[str, Any]) -> None:
        reserve = (1 + gen_cfg.max_retries) * worst_case_usd(
            len(task["prompt"].system) + len(task["prompt"].user), gen_cfg.max_tokens, budget)
        with lock:
            if state["stopped"]:
                return
            spent = spent_usd(ledger, budget)
            if spent + state["reserved"] + reserve > budget["limit_usd"]:
                state["stopped"] = True
                print(f"BUDGET STOP: spent ${spent:.4f} + in flight ${state['reserved']:.4f} + "
                      f"reserve ${reserve:.4f} > cap ${budget['limit_usd']:.2f}", flush=True)
                return
            state["reserved"] += reserve
        context = {"sample_id": task["sample_id"], "method": LONG_METHOD,
                   "block_id": task["block_id"], "construct": task["construct"],
                   "target_level": task["level"]}
        try:
            result = generate_dialogue(client, compiled_prompt=task["prompt"], config=gen_cfg,
                                       context=context)
        finally:
            with lock:
                state["reserved"] -= reserve
        record: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION, "sample_id": task["sample_id"],
            "block_id": task["block_id"], "case_id": task["case"]["case_id"],
            "department": task["case"]["department"], "label_family": task["family"],
            "focal_construct": task["construct"], "target_level": task["level"],
            "target_rank": task["rank"], "target_count": task["target_count"],
            "method": LONG_METHOD, "status": result.status,
            "persona": task["persona"], "persona_seed": task["persona_seed"], "plan": task["plan"],
            "case": task["case"], "case_provenance": task["case_provenance"],
            "generation": {"model": gen_cfg.model, "attempts": result.attempts,
                           "finish_reasons": result.finish_reasons,
                           "malformed_errors": result.malformed_errors,
                           "latency_ms_total": round(result.latency_ms_total, 1)},
            "lineage": {"git_commit": commit, "git_dirty": dirty,
                        "prompt_sha256": task["prompt"].prompt_hash,
                        "template_version": task["prompt"].template_version,
                        "profile_version": task["profile"]["version"]},
            "created_at": utc_now_iso(),
        }
        if result.turns is not None:
            stats = dialogue_stats(result.turns)
            record.update(
                turn_count=len(result.turns),
                dialogue={"turns": [dict(turn) for turn in result.turns]},
                stats={k: round(float(v), 4) for k, v in stats.items()},
                profile_comparison=compare_to_profile(stats, task["profile"]),
                contract_problems=check_long_dialogue(result.turns, task["plan"]),
            )
        with lock:
            with open(gen_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(f"{task['construct']} {task['level']}: {result.status}, "
                  f"{record.get('turn_count')} turns (planned {task['plan']['turn_count']}), "
                  f"spent so far ${spent_usd(ledger, budget):.4f} "
                  f"(x{budget['price_safety_factor']:g})", flush=True)

    with ThreadPoolExecutor(max_workers=int(config["generation"].get("workers", 1))) as pool:
        for future in [pool.submit(run_one, task) for task in todo]:
            future.result()

    if gen_path.exists():
        write_summary(out, config, ledger)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
