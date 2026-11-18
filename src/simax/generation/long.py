"""Full-visit planning, prompts and spoken-style rewrite."""

from __future__ import annotations

import hashlib
import json
import random
import re
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .prompts import CompiledPrompt, assert_no_bracket_markers, assert_no_metadata_leak
from .short import (
    render_short_examples_block,
    render_short_scale_block,
)

LONG_METHOD = "L1"
LONG_PROMPT_TEMPLATE_VERSION = "single-label-long-prompts-v5"
DEFAULT_PROFILE_PATH = Path(__file__).resolve().parents[3] / "configs" / "dialogue_profile.json"

_PHASES: tuple[tuple[str, float, str], ...] = (
    ("opening", 0.06, "greet, settle in, small talk, set the agenda for today"),
    ("history", 0.24, "the story of the problem so far, symptom by symptom"),
    ("medications and daily life", 0.18, "how medicines are actually taken, side effects, work, sleep, home"),
    ("examination", 0.08, "the doctor examines the patient and talks through it; short instructions and short replies"),
    ("results and assessment", 0.14, "go over test results and what the doctor thinks is going on"),
    ("plan", 0.20, "options, what to change, the patient's questions, back-and-forth until settled"),
    ("closing", 0.10, "refills, tests, scheduling, last questions, goodbye"),
)

_SYSTEM_PROMPT = (
    "You write realistic full-length clinic visit transcripts between a doctor "
    "and a patient, as they would look when a real conversation is transcribed "
    "word for word. You follow the required output format exactly, keep every "
    "stated clinical fact consistent, and write natural spoken English."
)

_FILLER_PAT = re.compile(
    r"\b(um+|uh+|mm+-?hm+|mhm+|yeah|okay|ok|right|you know|i mean|like)\b", re.IGNORECASE
)


def _has_blank(value: Any) -> bool:
    if isinstance(value, Mapping):
        return any(_has_blank(v) for v in value.values())
    if isinstance(value, list):
        return any(_has_blank(v) for v in value)
    return value is None


def _require_filled(profile: Mapping[str, Any], path: Path) -> None:
    if _has_blank(profile):
        raise ValueError(f"{path} still has blank values: fill in every null before generating dialogues")


def load_profile(path: Path | str | None = None) -> dict[str, Any]:
    path = Path(path or DEFAULT_PROFILE_PATH)
    profile = json.loads(path.read_text(encoding="utf-8"))
    _require_filled(profile, path)
    return profile


@dataclass(frozen=True)
class LongPhase:
    name: str
    goal: str
    first_turn: int
    last_turn: int
    topics: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "goal": self.goal, "first_turn": self.first_turn,
                "last_turn": self.last_turn, "topics": list(self.topics)}


@dataclass(frozen=True)
class LongPlan:
    sample_id: str
    block_id: str
    case_id: str
    focal_construct: str
    target_level: int
    persona_seed: int
    turn_count: int
    word_count: int
    min_turns: int
    max_turns: int
    phases: tuple[LongPhase, ...]
    plan_hash: str = field(default="")

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id, "block_id": self.block_id, "case_id": self.case_id,
            "focal_construct": self.focal_construct, "target_level": self.target_level,
            "persona_seed": self.persona_seed, "turn_count": self.turn_count,
            "word_count": self.word_count, "min_turns": self.min_turns,
            "max_turns": self.max_turns,
            "phases": [phase.to_dict() for phase in self.phases],
            "plan_hash": self.plan_hash,
        }


def _block_rng(block_id: str, persona_seed: int) -> random.Random:
    digest = int(hashlib.sha256(f"long-plan:{block_id}".encode()).hexdigest()[:12], 16)
    return random.Random(persona_seed ^ digest)


def plan_long_dialogue(
    *,
    sample_id: str,
    block_id: str,
    case: Mapping[str, Any],
    focal_construct: str,
    target_level: int,
    persona_seed: int,
    profile: Mapping[str, Any],
    turn_count: int | None = None,
) -> LongPlan:
    metrics, planning = profile["metrics"], profile["planning"]
    rng = _block_rng(block_id, persona_seed)
    if turn_count is None:
        low = int(metrics["turns"][planning["turn_count_low"]])
        high = int(metrics["turns"][planning["turn_count_high"]])
        turn_count = rng.randint(low, high)
    turn_count = int(turn_count) | 1
    words_per_turn = metrics["words"]["median"] / metrics["turns"]["median"]
    tolerance = float(planning["turn_tolerance"])

    agenda = case.get("agenda", {})
    topics_by_phase: dict[str, list[str]] = {name: [] for name, _, _ in _PHASES}
    if isinstance(agenda, Mapping):
        for name, items in agenda.items():
            name = {"interval history": "history"}.get(name, name)
            if name not in topics_by_phase:
                raise ValueError(f"unknown visit phase in case agenda: {name!r}")
            topics_by_phase[name].extend(str(item) for item in items)
    else:
        body = _PHASES[1:-1]
        for index, item in enumerate(agenda):
            topics_by_phase[body[index * len(body) // len(agenda)][0]].append(str(item))

    phases: list[LongPhase] = []
    cursor = 0
    cumulative = 0.0
    for name, share, goal in _PHASES:
        cumulative += share
        end = turn_count if name == _PHASES[-1][0] else max(cursor + 1, round(turn_count * cumulative))
        phases.append(LongPhase(name, goal, cursor + 1, end, tuple(topics_by_phase[name])))
        cursor = end

    plan = LongPlan(
        sample_id=sample_id, block_id=block_id, case_id=str(case.get("case_id", "")),
        focal_construct=str(focal_construct).strip().lower(), target_level=int(target_level),
        persona_seed=int(persona_seed), turn_count=turn_count,
        word_count=int(round(turn_count * words_per_turn, -2)),
        min_turns=int(turn_count * (1 - tolerance)), max_turns=int(turn_count * (1 + tolerance)),
        phases=tuple(phases),
    )
    payload = plan.to_dict()
    payload.pop("plan_hash")
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return LongPlan(**{**plan.__dict__, "plan_hash": hashlib.sha256(encoded).hexdigest()})


DEFAULT_TOPIC_PROFILE_PATH = DEFAULT_PROFILE_PATH.with_name("topic_profile.json")

_SMALL_TALK_SUBJECTS: tuple[tuple[str, int, int], ...] = (
    ("how their weekend went", 0, 120),
    ("their work or what they do during the day", 18, 70),
    ("a trip they took or are planning", 0, 120),
    ("news about their family", 0, 120),
    ("a hobby or something they enjoy doing", 0, 120),
    ("the weather and getting to the clinic today", 0, 120),
    ("plans for an upcoming holiday or celebration", 0, 120),
    ("a pet at home", 0, 120),
    ("their studies or a course they are taking", 18, 35),
    ("their grandchildren", 55, 120),
    ("a game or sports team they follow", 0, 120),
    ("something they have been cooking or baking", 0, 120),
    ("a show or book they have been enjoying", 0, 120),
    ("their garden or something they are fixing at home", 30, 120),
)


def _from_quantiles(points: Mapping[str, float], u: float) -> float:
    qs = [(0.10, points["q10"]), (0.25, points["q25"]), (0.50, points["median"]),
          (0.75, points["q75"]), (0.90, points["q90"])]
    p = 0.10 + 0.80 * u
    for (p0, v0), (p1, v1) in zip(qs, qs[1:]):
        if p <= p1:
            return v0 + (v1 - v0) * (p - p0) / (p1 - p0)
    return qs[-1][1]


def load_topic_profile(path: Path | str | None = None) -> dict[str, Any]:
    path = Path(path or DEFAULT_TOPIC_PROFILE_PATH)
    profile = json.loads(path.read_text(encoding="utf-8"))
    _require_filled(profile, path)
    return profile


_MAX_STRETCH_TURNS = 10
_SMALL_TALK_WORDS_PER_TURN = 9


def plan_small_talk(
    plan: Mapping[str, Any],
    *,
    topic_profile: Mapping[str, Any],
    patient_age: int | None = None,
) -> dict[str, Any]:
    targets = topic_profile["targets"]
    construct, level = str(plan["focal_construct"]), str(plan["target_level"])
    target = targets["warmth"][level] if construct == "warmth" else targets["default"]
    seed = int(hashlib.sha256(f"small-talk:{plan['block_id']}".encode()).hexdigest()[:12], 16)
    rng = random.Random(seed ^ int(plan["persona_seed"]))
    u = rng.random()
    if "quantiles" in target:
        count = int(round(_from_quantiles(target["quantiles"]["stretches"], u)))
        total = int(round(plan["turn_count"] * _from_quantiles(target["quantiles"]["turn_share"], u)))
    else:
        lo, hi = target["stretches"]
        count = int(round(lo + u * (hi - lo)))
        share_lo, share_hi = target["turn_share"]
        total = int(round(plan["turn_count"] * (share_lo + u * (share_hi - share_lo))))
    if count == 0 or total < 2:
        return {"turns": 0, "stretches": []}

    where = topic_profile["position"]
    n_turns = int(plan["turn_count"])
    subjects = [s for s, a, b in _SMALL_TALK_SUBJECTS if patient_age is None or a <= patient_age <= b]
    rng.shuffle(subjects)
    deciles = [float(x) for x in where["start_deciles"]]
    order = []
    while len(order) < 10:
        pick = rng.choices([d for d in range(10) if d not in order],
                           weights=[deciles[d] + 1e-6 for d in range(10) if d not in order])[0]
        order.append(pick)
    starts = [max(2, min(n_turns - 3, int((d + rng.random()) / 10 * n_turns))) for d in order]
    starters = ["doctor" if rng.random() < float(where["doctor_starts"]) else "patient" for _ in order]
    jitter = [0.6 + 0.8 * rng.random() for _ in order]

    count = min(count, len(order))
    scale = total / sum(jitter[:count])
    stretches = []
    for i in range(count):
        phase = next(p["name"] for p in plan["phases"] if p["first_turn"] <= starts[i] <= p["last_turn"])
        stretches.append({"phase": phase, "near_turn": starts[i],
                          "turns": min(_MAX_STRETCH_TURNS, max(2, int(round(jitter[i] * scale)))),
                          "starter": starters[i], "about": subjects[i % len(subjects)]})
    stretches.sort(key=lambda s: s["near_turn"])
    loose = construct in topic_profile.get("loose_positions_for", [])
    return {"turns": sum(s["turns"] for s in stretches), "stretches": stretches, "loose": loose}


_LOGISTICS_MOMENTS: tuple[str, ...] = (
    "the doctor looks something up in the record and reads it out",
    "the doctor orders a test or a referral on the computer and says what they are clicking",
    "checking which pharmacy the prescription should go to",
    "printing or handing over a leaflet or form",
    "sorting out whether a medicine or test is covered and what it costs",
    "confirming the patient's phone number or how they want to be contacted",
)

_LIFE_TALK_LABELS = frozenset({"sd", "w"})


def plan_visit_mix(plan: Mapping[str, Any], *, topic_profile: Mapping[str, Any]) -> dict[str, Any]:
    mix = topic_profile["visit_mix"]
    construct, level = str(plan["focal_construct"]), str(plan["target_level"])
    seed = int(hashlib.sha256(f"visit-mix:{plan['block_id']}".encode()).hexdigest()[:12], 16)
    rng = random.Random(seed ^ int(plan["persona_seed"]))
    u = rng.random()
    n_turns = int(plan["turn_count"])
    lo, hi = mix["logistics_turn_share"]
    logistics = int(round(n_turns * (lo + u * (hi - lo))))
    middle = int(round(logistics * float(mix["logistics_middle_share"])))
    n_moments = max(1, round(middle / 3))
    deciles = [float(x) for x in mix["logistics_middle_deciles"]]
    picks = sorted(rng.choices(range(len(deciles)), weights=deciles, k=n_moments))
    ideas = list(_LOGISTICS_MOMENTS)
    rng.shuffle(ideas)
    moments = [{"near_turn": max(3, int((d + rng.random()) / 10 * n_turns)),
                "turns": max(2, round(middle / n_moments)), "what": ideas[i % len(ideas)]}
               for i, d in enumerate(picks)]
    life = None
    open_life = {c.lower() for c in mix.get("life_unbudgeted_for", ())} | _LIFE_TALK_LABELS
    if construct.lower() not in open_life:
        band = mix["life_turn_share"]["warmth"][level] if construct == "warmth" else mix["life_turn_share"]["default"]
        life = int(round(n_turns * (band[0] + u * (band[1] - band[0]))))
    return {"logistics_turns": logistics, "logistics_moments": moments,
            "closing_logistics_turns": max(0, logistics - sum(m["turns"] for m in moments)),
            "life_turns": life, "loose": construct in topic_profile.get("loose_positions_for", [])}


_FLOW_TRANSITIONS: dict[str, str] = {
    "1": ("Every change of topic, into and out of the small talk and the logistics, is abrupt: the "
          "doctor cuts in while the patient is still answering, people talk over each other, and "
          "there are awkward gaps where someone trails off (\"so... anyway\") before the next topic "
          "starts. The patient rarely gets to finish a longer account."),
    "2": ("Changes of topic, into and out of the small talk and the logistics, are mostly fine, with "
          "a few awkward jumps or moments where both start talking at once."),
    "3": ("Every change of topic, into and out of the small talk and the logistics, is smooth: the "
          "doctor links it to what was just said, waits for the patient to finish, and lets the "
          "patient tell longer stretches of their story without interrupting, with open questions "
          "that keep them talking."),
}


def render_visit_mix(mix: Mapping[str, Any]) -> str:
    lines = [
        "Everyday business of the visit (real visits are full of it):",
        f"- About {mix['logistics_turns']} turns go to logistics: about "
        f"{mix['closing_logistics_turns']} near the end (scheduling, refills, tests to book, where "
        "to go next) and the rest as short moments in the middle of the visit:",
    ]
    if mix.get("loose"):
        lines.extend(f"    - wherever it fits, about {m['turns']} turns: {m['what']}."
                     for m in mix["logistics_moments"])
    else:
        lines.extend(f"    - around turn {m['near_turn']}: about {m['turns']} turns, {m['what']}."
                     for m in mix["logistics_moments"])
    if mix["life_turns"] is not None:
        lines.append(f"- Only about {mix['life_turns']} turns in the whole visit are about how the "
                     "problem affects work, home, sleep or mood; do not keep coming back to it. The "
                     "rest of the clinical talk stays on the symptoms, the findings and the plan.")
    return "\n".join(lines)


_LOW_ARM_GUARD = (
    "The limit on the target behavior (at most once in the whole visit) also covers the small talk "
    "and the everyday matters above: nothing said in them may count as an instance of it. Keep that "
    "chat to plain statements and short answers in the same way the rest of the visit keeps the "
    "behavior out."
)


def render_small_talk(small_talk: Mapping[str, Any]) -> str:
    if not small_talk["stretches"]:
        return ("Small talk: there is no small talk about the patient's life in this visit. "
                "Greetings and goodbyes stay brief and polite, and the rest is the visit's business.")
    lines = [
        "Small talk about the patient's life (outside the health problem: family, work, plans, "
        "their day). Real visits have it in a few short stretches, not spread thin across every turn:",
        f"- About {small_talk['turns']} turns in total, in {len(small_talk['stretches'])} separate "
        "stretches, not counting greetings and goodbyes:",
    ]
    for s in small_talk["stretches"]:
        start = ("the doctor starts it" if s["starter"] == "doctor" else
                 "the PATIENT opens it: the patient's own line, not an answer to a question, "
                 "volunteers something about their life, and the doctor then picks it up")
        where = ("wherever it fits naturally" if small_talk.get("loose")
                 else f"around turn {s['near_turn']} ({s['phase']})")
        words = s["turns"] * _SMALL_TALK_WORDS_PER_TURN
        lines.append(f"    - {where}: about {s['turns']} turns and no more than about {words} words "
                     f"in all; {start}; for example about {s['about']}; then back to the visit.")
    lines.append("- It is ordinary chat between two people in short turns, "
                 "of about nine words, with the same short replies and backchannels as the rest of "
                 "the visit, not a speech about rapport.")
    return "\n".join(lines)


def _render_long_case(case: Mapping[str, Any]) -> str:
    lines = [
        "Setting",
        f"- Department: {case.get('department', 'unspecified')}",
        f"- Visit: {case.get('visit_stage', 'unspecified')}",
        f"- Reason for visit: {case.get('chief_complaint', 'unspecified')}",
        "",
        "Clinical facts (every one of these must stay true and consistent; you may add "
        "ordinary everyday detail around them, but no new diagnoses, test results or drugs)",
    ]
    lines.extend(f"- {fact}" for fact in case.get("clinical_facts", []))
    return "\n".join(lines)


def render_long_outline(plan: Mapping[str, Any]) -> str:
    lines = ["Visit outline (approximate; let topics overlap and drift the way real visits do)"]
    for phase in plan["phases"]:
        goal = phase["goal"].replace(", small talk", "") if "small_talk" in plan else phase["goal"]
        if "visit_mix" in plan:
            goal = goal.replace(", work, sleep, home", "")
        lines.append(f"- Turns {phase['first_turn']}-{phase['last_turn']}, {phase['name']}: {goal}.")
        lines.extend(f"    - {topic}" for topic in phase["topics"])
    if "small_talk" in plan:
        lines.extend(["", render_small_talk(plan["small_talk"])])
    if "visit_mix" in plan:
        lines.extend(["", render_visit_mix(plan["visit_mix"])])
    if plan.get("low_arm_guard"):
        lines.extend(["", _LOW_ARM_GUARD])
    if plan.get("focal_construct") == "flow" and ("small_talk" in plan or "visit_mix" in plan):
        lines.extend(["", _FLOW_TRANSITIONS[str(plan["target_level"])]])
    return "\n".join(lines)


def render_long_shape(plan: Mapping[str, Any], profile: Mapping[str, Any]) -> str:
    m = profile["metrics"]
    d = profile["definitions"]
    n_turns, n_words = plan["turn_count"], plan["word_count"]
    n_short = round(n_turns * m["short_turn_fraction"]["median"])
    n_long = round(n_turns * m["long_turn_fraction"]["median"])
    doctor_pct = int(round(100 * m["doctor_word_share"]["median"]))
    n_patient_q = max(1, round(n_turns / 2 * m["patient_question_rate"]["median"]))
    n_fillers = round(n_words * m["fillers_per_100_words"]["median"] / 100, -1)
    return "\n".join([
        "Length and shape (this is a full visit, not a summary of one)",
        f"- Write about {n_turns} turns (between {plan['min_turns']} and {plan['max_turns']}). "
        "A turn is everything one person says before the other speaks.",
        f"- Write about {n_words} words in total. Do not fall short: most of the words sit in "
        "the long turns described below.",
        f"- Turn lengths are very uneven, and this matters more than anything else here:",
        f"    - about {n_short} turns are {d['short_turn_max_words']} words or fewer "
        "(\"Yeah.\", \"Okay.\", \"Mm-hmm.\", \"No.\", \"Right, right.\", \"Oh, okay.\");",
        f"    - about {n_long} turns are {d['long_turn_min_words']} words or more. Most of these "
        "are the doctor explaining, reasoning aloud or giving instructions, and several of them "
        "run to 80-150 words; a few are the patient telling what happened in their own "
        "roundabout way;",
        "    - the rest are a sentence or two.",
        f"- Counted in words, the doctor does about {doctor_pct}% of the talking. A typical "
        f"patient turn is about {int(m['patient_median_turn_words']['median'])} words; a typical "
        f"doctor turn about {int(m['doctor_median_turn_words']['median'])}.",
        "- The doctor asks a question in a bit under half of their turns and makes statements, "
        "acknowledgements or instructions in the rest. The patient asks only about "
        f"{n_patient_q} questions in the whole visit.",
        "- It is speech transcribed word for word, not written prose. Roughly one word in every "
        f"fifteen is um, uh, you know, I mean, like, yeah, okay or right (around {int(n_fillers)} "
        "across the visit), from both speakers and inside the long turns too. People start a "
        "sentence, drop it and start again; they repeat words; they trail off; they talk over "
        "each other, so a turn ends mid-sentence and is picked up later.",
        "- To show the texture only (do not reuse these lines or their content): "
        "\"So the, um, the plan today is, I mean, we'll start with, you know, how you've been, "
        "and then, uh, yeah, we'll go from there.\" / \"It was, like, Tuesday? No, Wednesday. "
        "Yeah, Wednesday, 'cause I had the, um, the thing at school.\" / \"Okay. Okay, yeah. "
        "And that's, uh, that's been the same since, or...?\"",
        "- Not every turn carries information. Include the ordinary texture of a visit: checking "
        "what was said, going back to an earlier topic, "
        + ("" if "small_talk" in plan else "small talk, ")
        + ("" if "visit_mix" in plan else "pharmacy and scheduling logistics, ")
        + "the doctor thinking aloud while looking something up, stretches where one person talks and the other only says \"mm-hmm\" or \"yeah\" in "
        "between.",
        "- Do not let the turns settle into a regular question-then-full-answer rhythm, and do "
        "not make every turn a complete, polished sentence.",
    ])


def render_long_count_block(
    label_spec: Mapping[str, Any], *, band: str, target_count: int
) -> str:
    who = {"doctor": "the doctor", "patient": "the patient"}.get(
        str(label_spec.get("side")), "either speaker")
    lines = [
        "Target behavior",
        f"Who: {who}",
        f"Definition: {label_spec['definition']}",
    ]
    if band == "low":
        lines.append(
            "Across the whole visit this behavior appears at most once. Do not replace it "
            "with something similar; the visit simply goes on without it."
        )
    elif band == "high":
        lines.append(
            f"Across the whole visit this behavior appears about {target_count} times, each one "
            "a clear instance, spread through the different parts of the visit rather than "
            "bunched together."
        )
        if label_spec.get("needs_setup"):
            lines.append(
                "Each instance responds to something the other person has just said, so that "
                "has to come first."
            )
    else:
        raise ValueError(f"unknown count band: {band!r}")
    note = (label_spec.get("notes") or {}).get(band)
    if note:
        lines.append(note)
    return "\n".join(lines)


def render_long_ordinal_block(
    spec: Mapping[str, Any], *, target_level: int, who: str
) -> str:
    lines = [
        f"Target behavior — full scale for reference (it describes {who})",
        f"Dimension: {spec['name']}",
    ]
    for level in (1, 2, 3):
        lines.append(f"- Scale point {level}: {spec['anchors'][str(level)]}")
    lines.append(f"Aim for scale point {target_level}: {spec['anchors'][str(target_level)]}")
    return "\n".join(lines)


def _format_long_contract(plan: Mapping[str, Any]) -> str:
    return "\n".join([
        "Output format",
        '- Reply with ONLY a JSON object: {"turns": [{"speaker": "doctor" or "patient", "text": "..."}]}',
        '- Each "text" is the exact spoken words; no stage directions, no bracketed notes, '
        "no speaker labels inside the text.",
        "- Turns strictly alternate, starting with the doctor and ending with the doctor "
        f"(so an odd number of turns, between {plan['min_turns']} and {plan['max_turns']}).",
        "- Do not include markdown fences or any text outside the JSON.",
    ])


_SHARED_LONG_REQUIREMENTS = (
    "Other requirements",
    "- What is described above should hold across the whole visit, shown through what is "
    "actually said and done, never by naming it.",
    "- Keep everything else about the visit ordinary; do not add extra emphasis on any other "
    "aspect of the interaction.",
    "- Do not add unsafe medical advice, insults, or discriminatory content.",
    "- Use first names or no names; do not invent surnames, addresses or dates of birth.",
)


def compile_long_prompt(
    *,
    case: Mapping[str, Any],
    people: str,
    plan: Mapping[str, Any],
    profile: Mapping[str, Any],
    focal_construct: str | None = None,
    target_level: int | None = None,
    rubric: Mapping[str, Any] | None = None,
    behavior: str | None = None,
) -> CompiledPrompt:
    components = {
        "case": _render_long_case(case),
        "people": people,
    }
    if behavior is not None:
        components["behavior"] = behavior
    else:
        if focal_construct is None or target_level is None or rubric is None:
            raise ValueError("an ordinal prompt needs focal_construct, target_level and rubric")
        focal_key = str(focal_construct).strip().lower()
        components["behavior"] = render_short_scale_block(rubric, focal_key, target_level)
        components["examples"] = render_short_examples_block(rubric, focal_key)
    components["outline"] = render_long_outline(plan)
    components["shape"] = render_long_shape(plan, profile)
    components["format"] = _format_long_contract(plan)
    sections = ["Write the full transcript of one clinic visit.", ""]
    for text in components.values():
        sections.extend([text, ""])
    sections.extend(_SHARED_LONG_REQUIREMENTS)
    user = "\n".join(sections).rstrip() + "\n"
    assert_no_metadata_leak(user)
    assert_no_metadata_leak(_SYSTEM_PROMPT)
    assert_no_bracket_markers(user, context="long prompt")
    return CompiledPrompt(
        method=LONG_METHOD,
        system=_SYSTEM_PROMPT,
        user=user,
        template_version=LONG_PROMPT_TEMPLATE_VERSION,
        components=components,
    )


def merge_speaker_runs(turns: Sequence[Mapping[str, Any]]) -> list[tuple[str, str]]:
    merged: list[list[str]] = []
    for turn in turns:
        speaker, text = str(turn["speaker"]), str(turn["text"]).strip()
        if not text:
            continue
        if merged and merged[-1][0] == speaker:
            merged[-1][1] += " " + text
        else:
            merged.append([speaker, text])
    return [(speaker, text) for speaker, text in merged]


def dialogue_stats(
    turns: Sequence[Mapping[str, Any]], *, short_max: int = 3, long_min: int = 40
) -> dict[str, float]:
    merged = merge_speaker_runs(turns)
    lengths = [len(text.split()) for _, text in merged]
    doctor = [n for (speaker, _), n in zip(merged, lengths) if speaker == "doctor"]
    patient = [n for (speaker, _), n in zip(merged, lengths) if speaker == "patient"]
    total = sum(lengths)
    if not doctor or not patient or not total:
        raise ValueError("dialogue needs at least one doctor and one patient turn")

    def question_rate(who: str) -> float:
        texts = [text for speaker, text in merged if speaker == who]
        return sum("?" in text for text in texts) / len(texts)

    return {
        "turns": len(merged),
        "words": total,
        "doctor_word_share": sum(doctor) / total,
        "doctor_median_turn_words": statistics.median(doctor),
        "patient_median_turn_words": statistics.median(patient),
        "doctor_mean_turn_words": statistics.mean(doctor),
        "patient_mean_turn_words": statistics.mean(patient),
        "short_turn_fraction": sum(n <= short_max for n in lengths) / len(lengths),
        "long_turn_fraction": sum(n >= long_min for n in lengths) / len(lengths),
        "doctor_question_rate": question_rate("doctor"),
        "patient_question_rate": question_rate("patient"),
        "fillers_per_100_words": 100 * sum(len(_FILLER_PAT.findall(t)) for _, t in merged) / total,
    }


def compare_to_profile(
    stats: Mapping[str, float], profile: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for key, band in profile["metrics"].items():
        value = stats[key]
        position = "low" if value < band["q10"] else ("high" if value > band["q90"] else "in")
        out[key] = {"value": round(float(value), 3), "q10": band["q10"],
                    "median": band["median"], "q90": band["q90"], "band": position}
    return out


def check_long_dialogue(turns: Sequence[Mapping[str, Any]], plan: Mapping[str, Any]) -> list[str]:
    problems: list[str] = []
    if not plan["min_turns"] <= len(turns) <= plan["max_turns"]:
        problems.append(
            f"dialogue must have between {plan['min_turns']} and {plan['max_turns']} turns; "
            f"got {len(turns)}"
        )
    try:
        assert_no_bracket_markers(" ".join(str(t["text"]) for t in turns), context="dialogue")
    except ValueError as exc:
        problems.append(str(exc))
    return problems


SPOKEN_TEMPLATE_VERSION = "long-spoken-rewrite-v4"

_SPOKEN_SYSTEM_PROMPT = (
    "You turn clean written dialogue into a faithful word-for-word transcript of how the same "
    "conversation would actually sound when spoken. You change how things are said, never what "
    "is said. You follow the required output format exactly."
)


def split_into_chunks(
    turns: Sequence[Mapping[str, Any]], *, chunk_turns: int = 30
) -> list[list[dict[str, Any]]]:
    size = max(2, chunk_turns - chunk_turns % 2)
    chunks = [[dict(t) for t in turns[i:i + size]] for i in range(0, len(turns), size)]
    if len(chunks) > 1 and len(chunks[-1]) < size // 3:
        chunks[-2].extend(chunks.pop())
    return chunks


def compile_spoken_rewrite_prompt(
    chunk: Sequence[Mapping[str, Any]], profile: Mapping[str, Any], *, overlap: bool = False
) -> CompiledPrompt:
    m = profile["metrics"]
    d = profile["definitions"]
    words = sum(len(str(t["text"]).split()) for t in chunk)
    n_fillers = max(4, round(words * m["fillers_per_100_words"]["median"] / 100))
    n_interject = max(2, round(len(chunk) / 7))
    n_short_answers = max(2, round(len(chunk) / 10))
    last = chunk[-1]["speaker"]
    source = json.dumps({"turns": [{"speaker": t["speaker"], "text": t["text"]} for t in chunk]},
                        ensure_ascii=False, indent=0)
    user = "\n".join([
        "Below is part of a clinic visit between a doctor and a patient. Rewrite it as a "
        "word-for-word transcript of real speech.",
        "",
        "Keep exactly the same",
        "- every fact, number, name of a medicine, symptom, result and instruction;",
        "- the order of topics and who says what;",
        "- how the doctor treats the patient: just as warm or cold, as attentive or distracted, "
        "as respectful or dismissive, as patient or rushed as in the original;",
        "- how the patient comes across: just as open or guarded, as trusting or doubtful, and "
        "just as ready to ask, push back, state preferences or bring up new topics as in the "
        "original;",
        "- the number of questions each person asks, and the number of apologies, expressions "
        "of sympathy, statements naming or acknowledging a feeling, and statements repeating "
        "back what the other person said. Do not add any of these and do not drop any.",
        "",
        "Change only how it sounds",
        f"- Work in about {n_fillers} filler words across this part, from both speakers. Most "
        "of them belong inside sentences (um, uh, you know, I mean, like, so, well, kind of), "
        "not as separate one-word turns.",
        "- Add false starts and self-corrections (\"it was, it's been about three weeks\"), "
        "repeated words (\"the, the knee\"), and sentences that trail off (\"so I just kind "
        "of...\").",
        f"- {n_interject} times in this part (not fewer), while one person is explaining or "
        "telling something at length, the listener comes in with \"Mm-hmm.\", \"Yeah.\", "
        "\"Okay.\" or \"Right.\" at a natural break between sentences, and the speaker "
        "carries on in a new turn. Spread these out. Never inside a question, never between a "
        "question and its answer, and at most twice in the same explanation.",
        f"- Make about {n_short_answers} of the replies in this part {d['short_turn_max_words']} "
        "words or fewer (\"Yeah.\", \"No.\", \"Not really.\", \"Every day.\", \"Okay.\"), "
        "where the original spells out a full sentence that adds nothing.",
        *(["- Let a turn sometimes get cut off mid-sentence by the other person and be finished "
           "afterwards."] if overlap else
          ["- Do not add interruptions, talking over each other, cut-off sentences or awkward "
           "pauses, and do not smooth away the ones already there: how smoothly the two people "
           "take turns must stay exactly as in the original."]),
        "- Use contractions and everyday wording; replace anything that reads like written "
        "prose. Keep long explanations long.",
        "- No stage directions, no bracketed notes, no laughter or sound descriptions.",
        "",
        "Output format",
        '- Reply with ONLY a JSON object: {"turns": [{"speaker": "doctor" or "patient", "text": "..."}]}',
        f"- Turns strictly alternate, starting with the doctor and ending with the {last}.",
        "- Do not include markdown fences or any text outside the JSON.",
        "",
        "Original",
        source,
    ]) + "\n"
    assert_no_metadata_leak(user.split("\nOriginal\n", 1)[0])
    return CompiledPrompt(
        method="spoken-rewrite",
        system=_SPOKEN_SYSTEM_PROMPT,
        user=user,
        template_version=SPOKEN_TEMPLATE_VERSION,
        components={"source": source},
    )


def check_rewritten_chunk(
    original: Sequence[Mapping[str, Any]], rewritten: Sequence[Mapping[str, Any]]
) -> list[str]:
    problems: list[str] = []
    if not rewritten:
        return ["empty rewrite"]
    if rewritten[0]["speaker"] != original[0]["speaker"]:
        problems.append("rewrite must start with the same speaker as the original")
    if rewritten[-1]["speaker"] != original[-1]["speaker"]:
        problems.append("rewrite must end with the same speaker as the original")
    for before, after in zip(rewritten, rewritten[1:]):
        if before["speaker"] == after["speaker"]:
            problems.append("rewrite breaks strict speaker alternation")
            break
    words_before = sum(len(str(t["text"]).split()) for t in original)
    words_after = sum(len(str(t["text"]).split()) for t in rewritten)
    if not 0.8 * words_before <= words_after <= 1.6 * words_before:
        problems.append(f"rewrite changed length too much ({words_before} -> {words_after} words)")
    try:
        assert_no_bracket_markers(" ".join(str(t["text"]) for t in rewritten), context="rewrite")
    except ValueError as exc:
        problems.append(str(exc))
    return problems
