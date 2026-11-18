"""Global scale and example prompt blocks."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from .prompts import (
    CompiledPrompt,
    assert_no_metadata_leak,
)

SHORT_METHODS = ("M1", "M2", "M3")
SHORT_PROMPT_TEMPLATE_VERSION = "single-label-short-prompts-v2"

SHORT_COUNT_PROMPT_TEMPLATE_VERSION = "all-label-count-prompts-v1"

SHORT_COUNT_METHODS = ("M1", "M2")

MIN_TURNS = 2
MAX_TURNS = 10

_CONTEXT_NEEDED_TURNS = {
    "concerns": 4,
    "attentive": 4,
}

_DEFAULT_TURNS = 2

_STAGES = ("opening", "history", "exploration", "explanation", "closing")

_STAGE_GOALS = {
    "opening": "open the consultation and set the agenda",
    "history": "collect focused history relevant to the chief complaint",
    "exploration": "explore the patient's perspective, worries, and daily impact",
    "explanation": "explain findings and options in plain language",
    "closing": "agree on next steps and close the visit",
}

_SYSTEM_PROMPT = (
    "You write realistic short clinical consultation dialogues between a doctor "
    "and a patient. You follow the required output format exactly, keep every "
    "stated clinical fact consistent, and use natural spoken English."
)


def _canonical_hash(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _plan_seed_int(persona_seed: int, sample_id: str) -> int:
    digest = int(hashlib.sha256(f"short-plan:{sample_id}".encode()).hexdigest()[:12], 16)
    return persona_seed ^ digest


def short_turn_count(focal: str) -> int:
    focal_key = str(focal).strip().lower()
    count = _CONTEXT_NEEDED_TURNS.get(focal_key, _DEFAULT_TURNS)
    return max(MIN_TURNS, min(MAX_TURNS, count))


@dataclass(frozen=True)
class ShortTurnPlan:
    turn_index: int
    speaker: str
    stage: str
    clinical_goal: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "turn_index": self.turn_index,
            "speaker": self.speaker,
            "stage": self.stage,
            "clinical_goal": self.clinical_goal,
        }


@dataclass(frozen=True)
class ShortPlan:
    sample_id: str
    block_id: str
    case_id: str
    focal_construct: str
    target_level: int
    persona_seed: int
    turn_count: int
    turns: tuple[ShortTurnPlan, ...]
    plan_hash: str = field(default="")

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "block_id": self.block_id,
            "case_id": self.case_id,
            "focal_construct": self.focal_construct,
            "target_level": self.target_level,
            "persona_seed": self.persona_seed,
            "turn_count": self.turn_count,
            "turns": [turn.to_dict() for turn in self.turns],
            "plan_hash": self.plan_hash,
        }

    def speakers(self) -> tuple[str, ...]:
        return tuple(turn.speaker for turn in self.turns)


def plan_short_dialogue(
    *,
    sample_id: str,
    block_id: str,
    case_id: str,
    focal_construct: str,
    target_level: int,
    persona_seed: int,
) -> ShortPlan:
    if target_level not in {1, 2, 3}:
        raise ValueError("target_level must be an integer in [1, 3]")
    focal_key = str(focal_construct).strip().lower()
    turn_count = short_turn_count(focal_key)
    speakers = ["doctor" if index % 2 == 0 else "patient" for index in range(turn_count)]
    stages = _allocate_short_stages(turn_count)
    turns = tuple(
        ShortTurnPlan(
            turn_index=index,
            speaker=speakers[index],
            stage=stages[index],
            clinical_goal=_STAGE_GOALS[stages[index]],
        )
        for index in range(turn_count)
    )
    plan = ShortPlan(
        sample_id=str(sample_id),
        block_id=str(block_id),
        case_id=str(case_id),
        focal_construct=focal_key,
        target_level=int(target_level),
        persona_seed=int(persona_seed),
        turn_count=turn_count,
        turns=turns,
    )
    return replace(plan, plan_hash=_canonical_hash(plan.to_dict()))


def _allocate_short_stages(turn_count: int) -> list[str]:
    if turn_count == 2:
        return ["opening", "closing"]
    stages = ["opening"] + ["exploration"] * (turn_count - 2) + ["closing"]
    return stages


def validate_short_plan(plan: ShortPlan) -> None:
    if not (MIN_TURNS <= plan.turn_count <= MAX_TURNS):
        raise ValueError(f"turn_count out of range: {plan.turn_count}")
    if plan.turn_count % 2 != 0:
        raise ValueError(f"short plan turn_count must be even: {plan.turn_count}")
    speakers = [turn.speaker for turn in plan.turns]
    if speakers[0] != "doctor":
        raise ValueError("short plan must start with the doctor")
    for index in range(1, plan.turn_count):
        if speakers[index] == speakers[index - 1]:
            raise ValueError("short plan speakers must strictly alternate")
    if plan.turn_count != len(plan.turns):
        raise ValueError("turn_count does not match the number of turns")


@dataclass(frozen=True)
class ShortCountTurnPlan:
    turn_index: int
    speaker: str
    stage: str
    clinical_goal: str
    behavior_count: int = 0
    setup: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "turn_index": self.turn_index,
            "speaker": self.speaker,
            "stage": self.stage,
            "clinical_goal": self.clinical_goal,
            "behavior_count": self.behavior_count,
            "setup": self.setup,
        }


@dataclass(frozen=True)
class ShortCountPlan:
    sample_id: str
    block_id: str
    case_id: str
    label_family: str
    label: str
    target_count: int
    side: str
    needs_setup: bool
    persona_seed: int
    turn_count: int
    turns: tuple[ShortCountTurnPlan, ...]
    plan_hash: str = field(default="")

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "block_id": self.block_id,
            "case_id": self.case_id,
            "label_family": self.label_family,
            "label": self.label,
            "target_count": self.target_count,
            "side": self.side,
            "needs_setup": self.needs_setup,
            "persona_seed": self.persona_seed,
            "turn_count": self.turn_count,
            "turns": [turn.to_dict() for turn in self.turns],
            "plan_hash": self.plan_hash,
        }

    def speakers(self) -> tuple[str, ...]:
        return tuple(turn.speaker for turn in self.turns)


def short_count_turn_count(
    count: int, *, side: str, needs_setup: bool, presence: bool = False
) -> int:
    if presence:
        return 2
    if side == "doctor":
        if needs_setup:
            return max(MIN_TURNS, min(MAX_TURNS, 2 * count + 1))
        return max(MIN_TURNS, min(MAX_TURNS, 2 * count - 1))
    return max(MIN_TURNS, min(MAX_TURNS, 2 * count))


def plan_short_count_dialogue(
    *,
    sample_id: str,
    block_id: str,
    case_id: str,
    label_family: str,
    label: str,
    target_count: int,
    side: str,
    needs_setup: bool,
    persona_seed: int,
    presence: bool = False,
) -> ShortCountPlan:
    if target_count < 0:
        raise ValueError("target_count must be >= 0")
    if side not in ("doctor", "patient", "topic"):
        raise ValueError(f"unknown side: {side}")
    if presence:
        target_count = 1
    turn_count = short_count_turn_count(
        target_count, side=side, needs_setup=needs_setup, presence=presence
    )
    speakers = ["doctor" if index % 2 == 0 else "patient" for index in range(turn_count)]
    stages = _allocate_short_stages(turn_count)

    behavior_turns: list[int] = []
    setup_turn: int | None = None
    if presence:
        behavior_turns = [1] if turn_count > 1 else [0]
    elif side == "doctor":
        if needs_setup:
            setup_turn = 1
            behavior_turns = [index for index in range(2, turn_count, 2)]
        else:
            behavior_turns = [index for index in range(0, turn_count, 2)]
    else:
        behavior_turns = [index for index in range(1, turn_count, 2)]

    per_turn: dict[int, int] = {index: 0 for index in behavior_turns}
    if behavior_turns:
        for offset in range(target_count):
            per_turn[behavior_turns[offset % len(behavior_turns)]] += 1

    turns = tuple(
        ShortCountTurnPlan(
            turn_index=index,
            speaker=speakers[index],
            stage=stages[index],
            clinical_goal=_STAGE_GOALS[stages[index]],
            behavior_count=per_turn.get(index, 0),
            setup=(index == setup_turn),
        )
        for index in range(turn_count)
    )
    plan = ShortCountPlan(
        sample_id=str(sample_id),
        block_id=str(block_id),
        case_id=str(case_id),
        label_family=str(label_family),
        label=str(label),
        target_count=int(target_count),
        side=str(side),
        needs_setup=bool(needs_setup),
        persona_seed=int(persona_seed),
        turn_count=turn_count,
        turns=turns,
    )
    return replace(plan, plan_hash=_canonical_hash(plan.to_dict()))


def validate_short_count_plan(plan: ShortCountPlan) -> None:
    if not (MIN_TURNS <= plan.turn_count <= MAX_TURNS):
        raise ValueError(f"turn_count out of range: {plan.turn_count}")
    speakers = [turn.speaker for turn in plan.turns]
    if speakers[0] != "doctor":
        raise ValueError("count plan must start with the doctor")
    for index in range(1, plan.turn_count):
        if speakers[index] == speakers[index - 1]:
            raise ValueError("count plan speakers must strictly alternate")
    if plan.turn_count != len(plan.turns):
        raise ValueError("turn_count does not match the number of turns")
    total = sum(turn.behavior_count for turn in plan.turns)
    if total != plan.target_count:
        raise ValueError(
            f"per-turn behavior counts sum to {total}, expected {plan.target_count}"
        )
    if plan.needs_setup:
        setup_turns = [turn.turn_index for turn in plan.turns if turn.setup]
        if not setup_turns:
            raise ValueError("needs_setup plan has no setup turn")
        for setup_index in setup_turns:
            if speakers[setup_index] != "patient":
                raise ValueError("setup turn must be a patient turn")
            if setup_index + 1 >= plan.turn_count:
                raise ValueError("setup turn must be followed by a behavior turn")
            if plan.turns[setup_index + 1].behavior_count < 1:
                raise ValueError("setup turn must precede a behavior turn")


def _require_construct(rubric: Mapping[str, Any], focal: str) -> Mapping[str, Any]:
    if focal not in rubric.get("constructs", {}):
        raise KeyError(f"construct not found in rubric: {focal}")
    return rubric["constructs"][focal]


def render_short_anchor_block(
    rubric: Mapping[str, Any], focal: str, target_level: int
) -> str:
    spec = _require_construct(rubric, focal)
    anchor = spec["anchors"][str(target_level)]
    lines = [
        "Doctor communication style",
        f"Dimension: {spec['name']}",
        f"Write the doctor's side of the dialogue so the interaction matches "
        f"scale point {target_level} of the description below.",
        f"  Description: {anchor}",
    ]
    return "\n".join(lines)


def render_short_scale_block(
    rubric: Mapping[str, Any], focal: str, target_level: int
) -> str:
    spec = _require_construct(rubric, focal)
    lines = [
        "Doctor communication style — full scale for reference",
        f"Dimension: {spec['name']}",
    ]
    for level in (1, 2, 3):
        lines.append(f"- Scale point {level}: {spec['anchors'][str(level)]}")
    lines.append(
        f"Aim for scale point {target_level}: {spec['anchors'][str(target_level)]}"
    )
    return "\n".join(lines)


def render_short_examples_block(
    rubric: Mapping[str, Any], focal: str, max_examples: int = 2
) -> str:
    spec = _require_construct(rubric, focal)
    examples = spec.get("examples", [])
    if not examples:
        return "No example utterances are available for this dimension."
    chosen = examples[:max_examples]
    lines = ["Example utterances consistent with this dimension"]
    for example in chosen:
        lines.append(f"- “{example}”")
    return "\n".join(lines)


def render_short_length_policy(plan: Mapping[str, Any]) -> str:
    lines = [
        "Dialogue length (follow exactly)",
        f"- Write exactly {plan['turn_count']} turns.",
        "- Turns strictly alternate, starting with the doctor.",
    ]
    for turn in plan["turns"]:
        lines.append(
            f"- Turn {turn['turn_index'] + 1} ({turn['speaker']}): "
            f"{turn['stage']} — {turn['clinical_goal']}."
        )
    return "\n".join(lines)


def _format_short_contract(plan: Mapping[str, Any] | None) -> str:
    lines = [
        "Output format",
        '- Reply with ONLY a JSON object: {"turns": [{"speaker": "doctor" or "patient", "text": "..."}]}',
        '- Each "text" is the exact spoken utterance; no stage directions, no speaker labels inside the text.',
    ]
    if plan is not None:
        lines.append(f"- Exactly {plan['turn_count']} turns, strictly alternating starting with the doctor.")
    else:
        lines.append(
            "- Turns strictly alternate starting with the doctor and ending with the doctor; "
            "use an even number of turns between 2 and 10."
        )
    lines.extend(
        [
            "- Do not include markdown fences or any text outside the JSON.",
            "- Each doctor turn is one to three natural spoken sentences.",
        ]
    )
    return "\n".join(lines)


_SHARED_SHORT_REQUIREMENTS = (
    "Other requirements",
    "- Keep the interaction ordinary apart from the doctor communication style "
    "described above; do not add extra emphasis on any other aspect of the interaction.",
    "- Use plain, natural spoken English.",
    "- Do not add unsafe medical advice, insults, or discriminatory content.",
)


def compile_short_prompt(
    method: str,
    *,
    case: Mapping[str, Any],
    persona: Mapping[str, Any],
    focal_construct: str,
    target_level: int,
    rubric: Mapping[str, Any],
    plan: Mapping[str, Any] | None = None,
) -> CompiledPrompt:
    if method not in SHORT_METHODS:
        raise ValueError(f"unknown short method: {method}")
    if method == "M3" and plan is None:
        raise ValueError("method M3 requires a dialogue plan")
    if method in ("M1", "M2") and plan is not None:
        raise ValueError(f"method {method} must not receive a dialogue plan")

    focal_key = str(focal_construct).strip().lower()
    components: dict[str, str] = {
        "case": _render_short_case(case),
        "persona": _render_short_persona(persona),
    }
    if method == "M1":
        behavior = render_short_anchor_block(rubric, focal_key, target_level)
    else:
        behavior = render_short_scale_block(rubric, focal_key, target_level)
    components["behavior"] = behavior

    sections: list[str] = [
        "Write a short clinical consultation dialogue.",
        "",
        components["case"],
        "",
        components["persona"],
        "",
        behavior,
        "",
    ]
    if method in ("M2", "M3"):
        components["examples"] = render_short_examples_block(rubric, focal_key)
        sections.extend([components["examples"], ""])
    if method == "M3":
        components["length_policy"] = render_short_length_policy(plan)
        sections.extend([components["length_policy"], ""])

    components["format"] = _format_short_contract(plan if method == "M3" else None)
    sections.extend(
        [
            components["format"],
            "",
            *_SHARED_SHORT_REQUIREMENTS,
        ]
    )
    user = "\n".join(sections).rstrip() + "\n"
    assert_no_metadata_leak(user)
    assert_no_metadata_leak(_SYSTEM_PROMPT)
    return CompiledPrompt(
        method=method,
        system=_SYSTEM_PROMPT,
        user=user,
        template_version=SHORT_PROMPT_TEMPLATE_VERSION,
        components=components,
    )


def _render_short_case(case: Mapping[str, Any]) -> str:
    facts = [str(fact) for fact in case.get("clinical_facts", [])]
    lines = [
        "Setting",
        f"- Department: {case.get('department', 'unspecified')}",
        f"- Visit stage: {case.get('visit_stage', 'unspecified')}",
        f"- Chief complaint: {case.get('chief_complaint', 'unspecified')}",
        "",
        "Clinical facts (every one of these must stay true and consistent)",
    ]
    if facts:
        lines.extend(f"- {fact}" for fact in facts)
    else:
        lines.append("- No additional clinical facts are provided.")
    return "\n".join(lines)


def _render_short_persona(persona: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            "Patient profile (keep fixed throughout the dialogue)",
            f"- Interaction style: {persona.get('patient_style', 'not specified')}",
            f"- Opening mood: {persona.get('patient_opening_mood', 'not specified')}",
            f"- Doctor style: {persona.get('doctor_style', 'not specified')}",
            f"- Setting: {persona.get('setting_note', 'not specified')}",
        ]
    )


def render_short_count_anchor(
    label_spec: Mapping[str, Any], target_count: int, *, presence: bool = False
) -> str:
    lines = [
        "Target behavior",
        f"Definition: {label_spec['definition']}",
    ]
    if presence:
        lines.append(
            "Write the dialogue so this topic comes up naturally at least once."
        )
    else:
        lines.append(
            f"Write the dialogue so this behavior appears exactly {target_count} "
            "time(s) across the whole dialogue."
        )
    return "\n".join(lines)


def render_short_count_allocation(plan: Mapping[str, Any]) -> str:
    lines = [
        "Dialogue plan (follow exactly)",
        f"- Write exactly {plan['turn_count']} turns.",
        "- Turns strictly alternate, starting with the doctor.",
    ]
    for turn in plan["turns"]:
        prefix = f"Turn {turn['turn_index'] + 1} ({turn['speaker']})"
        goal = f"{turn['stage']} — {turn['clinical_goal']}"
        if turn.get("setup"):
            lines.append(
                f"- {prefix} — {goal}. On this turn the patient first raises the "
                "context the target behavior responds to."
            )
        elif turn.get("behavior_count", 0) > 0:
            lines.append(
                f"- {prefix} — {goal}. Include exactly {turn['behavior_count']} "
                "occurrence(s) of the target behavior on this turn."
            )
        else:
            lines.append(f"- {prefix} — {goal}. No target behavior on this turn.")
    return "\n".join(lines)


def _format_short_count_contract(plan: Mapping[str, Any] | None) -> str:
    lines = [
        "Output format",
        '- Reply with ONLY a JSON object: {"turns": [{"speaker": "doctor" or "patient", "text": "..."}]}',
        '- Each "text" is the exact spoken utterance; no stage directions, no speaker labels inside the text.',
    ]
    if plan is not None:
        lines.append(
            f"- Exactly {plan['turn_count']} turns, strictly alternating starting with the doctor."
        )
    else:
        lines.append(
            "- Turns strictly alternate starting with the doctor; use an even number of turns between 2 and 10."
        )
    lines.extend(
        [
            "- Do not include markdown fences or any text outside the JSON.",
            "- Each turn is one to three natural spoken sentences.",
        ]
    )
    return "\n".join(lines)


_SHARED_COUNT_REQUIREMENTS = (
    "Other requirements",
    "- Keep the interaction ordinary apart from the target behavior described "
    "above; do not add extra emphasis on any other aspect of the interaction.",
    "- Use plain, natural spoken English.",
    "- Do not add unsafe medical advice, insults, or discriminatory content.",
)


def compile_short_count_prompt(
    method: str,
    *,
    case: Mapping[str, Any],
    persona: Mapping[str, Any],
    label_spec: Mapping[str, Any],
    target_count: int,
    plan: Mapping[str, Any] | None = None,
    presence: bool = False,
) -> CompiledPrompt:
    if method not in SHORT_COUNT_METHODS:
        raise ValueError(f"unknown count method: {method}")
    if method == "M2" and plan is None:
        raise ValueError("method M2 requires a count plan")
    if method == "M1" and plan is not None:
        raise ValueError("method M1 must not receive a count plan")

    components: dict[str, str] = {
        "case": _render_short_case(case),
        "persona": _render_short_persona(persona),
        "behavior": render_short_count_anchor(
            label_spec, target_count, presence=presence
        ),
    }
    sections: list[str] = [
        "Write a short clinical consultation dialogue.",
        "",
        components["case"],
        "",
        components["persona"],
        "",
        components["behavior"],
        "",
    ]
    if method == "M2":
        components["allocation"] = render_short_count_allocation(plan)
        sections.extend([components["allocation"], ""])

    components["format"] = _format_short_count_contract(plan if method == "M2" else None)
    sections.extend(
        [
            components["format"],
            "",
            *_SHARED_COUNT_REQUIREMENTS,
        ]
    )
    user = "\n".join(sections).rstrip() + "\n"
    assert_no_metadata_leak(user)
    assert_no_metadata_leak(_SYSTEM_PROMPT)
    return CompiledPrompt(
        method=method,
        system=_SYSTEM_PROMPT,
        user=user,
        template_version=SHORT_COUNT_PROMPT_TEMPLATE_VERSION,
        components=components,
    )
