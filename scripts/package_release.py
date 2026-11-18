#!/usr/bin/env python3
"""Package checked dialogues into the release format."""
import argparse, json, os, subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FFMPEG = os.environ.get("FFMPEG", "ffmpeg")
CODEBOOK = {"global": "Global", "wiser": "WISER", "patient": "Patient Behavior", "sdoh": "SDOH&weight",
            "intervention": "intervention"}
MOSAIC_DIRS = ("global", "wiser", "patient_behavior", "sdoh_weight", "intervention")
GLOBAL_NAMES = {"flow": "Flow", "concerns": "Concerns", "attentive": "Attentive", "warmth": "Warmth", "respect": "Respect"}


def encode(src: Path, dst: Path):
    tmp = dst.with_name(dst.stem + ".part" + dst.suffix)
    subprocess.run([FFMPEG, "-hide_banner", "-loglevel", "error", "-y", "-i", str(src), "-map_metadata", "-1",
                    "-ac", "1", "-c:a", "libmp3lame", "-b:a", "128k", str(tmp)], check=True)
    tmp.replace(dst)


AUTHOR_DOCTOR = {"age": 28, "gender": "male", "region": "American", "voice": "recording of a paper author (not Common Voice)"}


def clinician_fields(part, text_persona):
    if part == "Internal-Extended":
        return {"clinician": AUTHOR_DOCTOR, "clinician_text_persona": text_persona}
    return {"clinician": text_persona}


def ready(d: Path) -> bool:
    full, wet, verdict = d / "dialogue_full.wav", d / "dialogue_wet.wav", d / "qc_verdict.json"
    if not (full.exists() and wet.exists() and verdict.exists() and (d / "dialogue.json").exists()):
        return False
    return wet.stat().st_mtime >= full.stat().st_mtime and \
        all(c["status"] != "fail" for c in json.load(open(verdict))["chunks"])


def render_transcript(turns) -> str:
    lines = [f"[{t['turn']:03d}] {t['speaker'].capitalize()}: {t['text']}" for t in turns]
    return "\n\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tts", required=True, help="folder with the dialogue folders of this voice source")
    ap.add_argument("--part", required=True, choices=["Public-Short", "Internal-Extended"])
    ap.add_argument("--spoken", required=True, help="generation.jsonl of the spoken rewrite")
    ap.add_argument("--mosaic", default=None, help="optional folder with MOSAIC summary.json files for the source text")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    rows = {}
    for line in open(args.spoken, encoding="utf-8"):
        r = json.loads(line); rows[r["sample_id"]] = r
    plan = json.load(open(ROOT / "configs/persona_plan.json"))["blocks"]
    mosaic = {}
    for name in MOSAIC_DIRS if args.mosaic else ():
        summary = Path(args.mosaic) / name / "summary.json"
        if summary.exists():
            for r in json.load(open(summary))["rows"]:
                mosaic[r["sample_id"]] = r
    prefix = {"Public-Short": "v10f5-", "Internal-Extended": "v10ru-"}[args.part]
    todo = []
    for d in sorted(Path(args.tts).glob("*__*")):
        fam = d.name.split("__")[1].split("_")[0]
        if fam in CODEBOOK and ready(d):
            todo.append((d, fam))
    for d, fam in todo:
        pid = d.name.split("__")[0].replace(prefix, "", 1)
        dst = out / pid
        if (dst / "metadata.json").exists():
            continue
        meta = json.load(open(d / "dialogue.json"))
        row = rows[meta["source_sample_id"]]
        blk = plan[row["block_id"]]
        dst.mkdir(parents=True, exist_ok=True)
        encode(d / "dialogue_full.wav", dst / "dry.mp3")
        encode(d / "dialogue_wet.wav", dst / "wet.mp3")
        turns = [{"turn": t["turn_index"], "speaker": t["speaker"], "text": t["text"]} for t in meta["turns"]]
        checks = {c["k"]: c for c in json.load(open(d / "qc_verdict.json"))["chunks"]}
        chunks, t0 = [], 0.0
        for k, c in enumerate(meta["chunks"]):
            asr = json.load(open(d / f"chunk_{k:02d}.qc.json"))["metrics"]
            chunks.append({"index": k, "turn_range": c["turns"], "start_s": round(t0, 2),
                           "end_s": round(t0 + c["seconds"], 2), "words": c["words"], "pause_after_s": c["gap_after_s"],
                           "asr_missing_rate": asr["missing"], "asr_added_rate": asr["added"],
                           "asr_pass": checks[k]["status"] != "fail"})
            t0 += c["seconds"] + c["gap_after_s"]
        (dst / "transcript.txt").write_text(render_transcript(turns), encoding="utf-8")
        label = meta["label"]
        is_global = fam == "global"
        target = ({"type": "level", "value": row["target_level"]} if is_global else
                  {"type": "count_band", "value": row["target_level"], "target_count": row.get("target_count")})
        m = mosaic.get(meta["source_sample_id"])
        mos = None
        if m is not None:
            mos = {"agent": CODEBOOK[fam], "construct": GLOBAL_NAMES.get(label, label),
                   "value": m.get("mosaic_value"), "matches_target": m.get("correct")}
        st = row["stats"]
        out_meta = {
            "id": pid, "codebook": CODEBOOK[fam], "label": GLOBAL_NAMES.get(label, label), "target": target,
            "department": row["department"], "visit_stage": row["case"]["visit_stage"],
            "condition": {"name": row["case"]["condition"], "icd10": blk.get("icd10")},
            "patient": row["persona"]["patient"], **clinician_fields(args.part, row["persona"]["clinician"]),
            "dialogue_stats": {k: st[k] for k in ("turns", "words", "doctor_word_share", "short_turn_fraction",
                                                  "fillers_per_100_words")},
            "audio": {"dry": "dry.mp3", "wet": "wet.mp3", "format": "mp3 128 kbps CBR mono", "sample_rate_hz": 24000,
                      "duration_min": meta["minutes"], "chunks": chunks},
            "mosaic": mos, "text_model": row["generation"]["model"], "audio_model": "MOSS-TTSD v1.0",
        }
        (dst / "metadata.json").write_text(json.dumps(out_meta, indent=1, ensure_ascii=False))
        print("built", pid, flush=True)


if __name__ == "__main__":
    main()
