#!/usr/bin/env python3
"""Turn spoken dialogues into audio input rows."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VOICE_SOURCES = {"Public-Short": "v10f5", "Internal-Extended": "v10ru"}
AUTHOR_CLINICIAN_VOICE = "ruio_doctor.wav"


def rows_of(record: dict, voices: dict) -> list[dict]:
    label, target = record["focal_construct"], record["target_level"]
    out = []
    for part, tag in VOICE_SOURCES.items():
        doctor = AUTHOR_CLINICIAN_VOICE if part == "Internal-Extended" else voices["doctor"]
        out.append({"sample_id": f"{tag}-{label}-{target}-{record['block_id'][:8]}", "part": part,
                    "source_sample_id": record["sample_id"], "family": record["label_family"], "label": label,
                    "target": target, "voice_doctor": doctor, "voice_patient": voices["patient"],
                    "turns": record["dialogue"]["turns"]})
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spoken", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--plan", default=str(ROOT / "configs/persona_plan.json"))
    args = ap.parse_args(argv)
    plan = json.loads(Path(args.plan).read_text())["blocks"]
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(out, "w", encoding="utf-8") as fh:
        for line in open(args.spoken, encoding="utf-8"):
            record = json.loads(line)
            if record.get("status") != "ok":
                continue
            for row in rows_of(record, plan[record["block_id"]]["voices"]):
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                n += 1
    print(f"wrote {n} audio rows to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
