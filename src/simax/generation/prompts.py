"""Shared prompt components and leak checks."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

PROMPT_TEMPLATE_VERSION = "p1method-prompts-v2"

METADATA_LEAK_PATTERNS: tuple[str, ...] = (
    "you are being tested",
    "being tested",
    "experiment",
    "experimental",
    "judge",
    "evaluat",
    "will be scored",
    "will be rated",
    "will be graded",
    "your output will be",
    "sample_id",
    "sample id",
    "manifest",
    "benchmark",
    "gold label",
    "gold standard",
    "rubric",
    "codebook",
    "condition name",
    "method name",
    "baseline level",
)

P1_METHODS = (
    "target_anchor",
    "full_contrastive_scale",
    "plan_then_generate",
    "plan_generate_verify_revise",
)

_SYSTEM_PROMPT = (
    "You write realistic clinical consultation dialogues between a doctor and a "
    "patient. You follow the required output format exactly, keep every stated "
    "clinical fact consistent, and use natural spoken English."
)


@dataclass(frozen=True)
class CompiledPrompt:
    method: str
    system: str
    user: str
    template_version: str
    components: dict[str, str]

    @property
    def prompt_hash(self) -> str:
        payload = json.dumps(
            {"system": self.system, "user": self.user},
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode()).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "system": self.system,
            "user": self.user,
            "template_version": self.template_version,
            "prompt_hash": self.prompt_hash,
            "components": dict(self.components),
        }


def assert_no_metadata_leak(prompt: str) -> None:
    lowered = prompt.lower()
    for pattern in METADATA_LEAK_PATTERNS:
        if pattern in lowered:
            raise ValueError(
                f"prompt contains forbidden metadata phrase: {pattern!r}"
            )


_BRACKET_MARKER_PAT = re.compile(r"\[(?!LINE_ID)[A-Za-z][A-Za-z0-9 /_-]{1,30}\]")


def assert_no_bracket_markers(text: str, *, context: str = "text") -> None:
    hits = _BRACKET_MARKER_PAT.findall(text)
    if hits:
        raise ValueError(
            f"{context} contains codebook annotation markers that must not "
            f"reach transcripts or TTS input: {sorted(set(hits))[:5]}"
        )


def rubric_version(rubric: Mapping[str, Any]) -> str:
    if "version" in rubric:
        return str(rubric["version"])
    encoded = json.dumps(rubric, sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()[:16]


def _require_rubric_entry(rubric: Mapping[str, Any], construct: str) -> Mapping[str, Any]:
    if construct not in rubric:
        raise KeyError(f"construct not found in rubric: {construct}")
    return rubric[construct]


def render_case_block(case: Mapping[str, Any]) -> str:
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


def render_persona_block(persona: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            "Patient profile (keep fixed throughout the dialogue)",
            f"- Interaction style: {persona.get('patient_style', 'not specified')}",
            f"- Opening mood: {persona.get('patient_opening_mood', 'not specified')}",
            f"- Doctor style: {persona.get('doctor_style', 'not specified')}",
            f"- Setting: {persona.get('setting_note', 'not specified')}",
        ]
    )


def _render_anchor_lines(anchor: Mapping[str, Any]) -> list[str]:
    lines = [f"  Description: {anchor['definition']}"]
    for item in anchor.get("evidence", []):
        lines.append(f"  Observable sign: {item}")
    for item in anchor.get("avoid", []):
        lines.append(f"  Do not include: {item}")
    return lines


def render_anchor_block(
    rubric: Mapping[str, Any], construct: str, target_level: int
) -> str:
    spec = rubric[construct]
    anchor = spec["anchors"][str(target_level)]
    lines = [
        "Doctor communication style",
        f"Dimension: {spec['definition']}",
        f"Write the doctor's side of the dialogue so the interaction matches "
        f"scale point {target_level} of the description below.",
    ]
    lines.extend(_render_anchor_lines(anchor))
    return "\n".join(lines)


def render_scale_block(
    rubric: Mapping[str, Any], construct: str, target_level: int
) -> str:
    spec = rubric[construct]
    lines = [
        "Doctor communication style — full scale for reference",
        f"Dimension: {spec['definition']}",
    ]
    for level in (1, 2, 3):
        lines.append(f"- Scale point {level}:")
        lines.extend(_render_anchor_lines(spec["anchors"][str(level)]))
    lines.append(
        f"Aim for scale point {target_level}: "
        f"{spec['anchors'][str(target_level)]['definition']}"
    )
    return "\n".join(lines)


def render_plan_block(
    plan: Mapping[str, Any],
    act_labels: Mapping[str, str] | None = None,
) -> str:

    def _label(act_name: str) -> str:
        if act_labels and act_name in act_labels:
            return act_labels[act_name]
        return act_name.replace("_", " ")

    lines = ["Dialogue plan (follow exactly)"]
    for turn in plan["turns"]:
        goal = f"{turn['stage']}: {turn['clinical_goal']}"
        prefix = f"Turn {turn['turn_index'] + 1} — {turn['speaker']}"
        if turn["behavior_acts"]:
            counts = ", ".join(
                f"{count} {_label(act_name)}"
                for act_name, count in sorted(turn["behavior_acts"].items())
            )
            lines.append(f"- {prefix} — {goal}")
            lines.append(f"  Required on this turn: {counts}.")
        else:
            lines.append(f"- {prefix} — {goal}")
            lines.append("  Required on this turn: none.")
        focus = [
            control
            for control in turn.get("controls", [])
            if control.get("evidence")
        ]
        if turn["behavior_acts"]:
            for control in focus:
                joined = "; ".join(
                    str(item).rstrip(".") for item in control["evidence"]
                )
                lines.append(f"  Style focus: {joined}.")
    return "\n".join(lines)


def _format_contract(plan: Mapping[str, Any] | None) -> str:
    lines = [
        "Output format",
        '- Reply with ONLY a JSON object: {"turns": [{"speaker": "doctor" or "patient", "text": "..."}]}',
        '- Each "text" is the exact spoken utterance; no stage directions, no speaker labels inside the text.',
    ]
    if plan is not None:
        lines.append(
            f"- Exactly {plan['turn_count']} turns, strictly alternating starting with the "
            "doctor and ending with the doctor."
        )
    else:
        lines.append(
            "- Turns strictly alternate starting with the doctor and ending with the doctor; "
            "use an odd number of turns between 9 and 13."
        )
    lines.extend(
        [
            "- Do not include markdown fences or any text outside the JSON.",
            "- Each doctor turn is one to three natural spoken sentences.",
        ]
    )
    return "\n".join(lines)


_SHARED_REQUIREMENTS = (
    "Other requirements",
    "- Keep the interaction ordinary apart from the doctor communication style "
    "described above; do not add extra emphasis on any other aspect of the interaction.",
    "- Use plain, natural spoken English.",
    "- Do not add unsafe medical advice, insults, or discriminatory content.",
)


def _enrich_plan_controls(
    plan: Mapping[str, Any],
    rubric: Mapping[str, Any],
    construct: str,
    target_level: int,
) -> dict[str, Any]:
    spec = _require_rubric_entry(rubric, construct)
    anchor = spec["anchors"][str(target_level)]
    enriched = json.loads(json.dumps(plan))
    for turn in enriched["turns"]:
        for control in turn.get("controls", []):
            if control.get("construct") == construct:
                control["evidence"] = list(anchor.get("evidence", []))
                control["avoid"] = list(anchor.get("avoid", []))
    return enriched


def compile_prompt(
    method: str,
    *,
    case: Mapping[str, Any],
    persona: Mapping[str, Any],
    construct: str,
    target_level: int,
    rubric: Mapping[str, Any],
    plan: Mapping[str, Any] | None = None,
    act_labels: Mapping[str, str] | None = None,
) -> CompiledPrompt:
    if method not in P1_METHODS:
        raise ValueError(f"unknown method: {method}")
    if method in ("plan_then_generate", "plan_generate_verify_revise") and plan is None:
        raise ValueError(f"method {method} requires a dialogue plan")
    if method in ("target_anchor", "full_contrastive_scale") and plan is not None:
        raise ValueError(f"method {method} must not receive a dialogue plan")

    components: dict[str, str] = {
        "case": render_case_block(case),
        "persona": render_persona_block(persona),
    }
    if method == "target_anchor":
        behavior = render_anchor_block(rubric, construct, target_level)
    else:
        behavior = render_scale_block(rubric, construct, target_level)
    components["behavior"] = behavior

    sections: list[str] = [
        "Write a clinical consultation dialogue.",
        "",
        components["case"],
        "",
        components["persona"],
        "",
        behavior,
        "",
    ]
    if plan is not None:
        enriched_plan = _enrich_plan_controls(plan, rubric, construct, target_level)
        components["plan"] = render_plan_block(enriched_plan, act_labels)
        sections.extend([components["plan"], ""])

    if method == "plan_generate_verify_revise":
        components["verification"] = render_verification_contract(plan)
        sections.extend([components["verification"], ""])

    components["format"] = _format_contract(plan)
    sections.extend(
        [
            components["format"],
            "",
            *_SHARED_REQUIREMENTS,
        ]
    )
    user = "\n".join(sections).rstrip() + "\n"
    assert_no_metadata_leak(user)
    assert_no_metadata_leak(_SYSTEM_PROMPT)
    return CompiledPrompt(
        method=method,
        system=_SYSTEM_PROMPT,
        user=user,
        template_version=PROMPT_TEMPLATE_VERSION,
        components=components,
    )


def render_verification_contract(plan: Mapping[str, Any]) -> str:
    lines = ["Verification contract"]
    lines.append(
        "- Each doctor turn lists required behavior counts. Follow them exactly: "
        "produce neither more nor fewer of these behaviors than the plan requires "
        "on that turn, and do not add them on turns without a requirement."
    )
    lines.append(
        "- After writing, re-read the plan turn by turn and confirm every listed "
        "count is met before replying."
    )
    return "\n".join(lines)


def compile_revision_prompt(
    *,
    case: Mapping[str, Any],
    persona: Mapping[str, Any],
    construct: str,
    target_level: int,
    rubric: Mapping[str, Any],
    plan: Mapping[str, Any],
    dialogue_turns: list[dict[str, Any]],
    offending: Mapping[int, str],
    act_labels: Mapping[str, str] | None = None,
) -> CompiledPrompt:
    if not offending:
        raise ValueError("revision requested but no offending turns given")
    behavior = render_scale_block(rubric, construct, target_level)
    enriched_plan = _enrich_plan_controls(plan, rubric, construct, target_level)
    plan_text = render_plan_block(enriched_plan, act_labels)
    transcript = "\n".join(
        f"[{index + 1}] {turn['speaker']}: {turn['text']}"
        for index, turn in enumerate(dialogue_turns)
    )
    findings = "\n".join(
        f"- Turn {index + 1} ({dialogue_turns[index]['speaker']}): {offending[index]}"
        for index in sorted(offending)
    )
    sections = [
        "A clinical consultation dialogue was written from a plan, but some turns "
        "do not meet their required behavior counts.",
        "",
        render_case_block(case),
        "",
        render_persona_block(persona),
        "",
        behavior,
        "",
        "Dialogue plan:",
        plan_text,
        "",
        "Current dialogue:",
        transcript,
        "",
        "Turns that do not meet their requirements:",
        findings,
        "",
        "Rewrite ONLY the turns listed above.",
        "- Keep every other turn exactly as written, word for word.",
        "- Keep every clinical fact true.",
        "- Make each rewritten turn satisfy its required behavior counts exactly.",
        "- Do not add speaker labels or stage directions inside the text.",
        "",
        "Output format",
        '- Reply with ONLY a JSON object: {"turns": [{"turn_index": <number>, "text": "..."}]}',
        "- Include one entry per rewritten turn, using the same turn numbers as above.",
        "- Do not include markdown fences or any text outside the JSON.",
        "",
        "Example shape (turn numbers are illustrative):",
        '{"turns": [{"turn_index": 2, "text": "Your rewritten utterance here."}, '
        '{"turn_index": 4, "text": "Another rewritten utterance."}]}',
        *_SHARED_REQUIREMENTS,
    ]
    user = "\n".join(sections).rstrip() + "\n"
    assert_no_metadata_leak(user)
    return CompiledPrompt(
        method="plan_generate_verify_revise",
        system=_SYSTEM_PROMPT,
        user=user,
        template_version=PROMPT_TEMPLATE_VERSION,
        components={
            "case": render_case_block(case),
            "persona": render_persona_block(persona),
            "behavior": behavior,
            "plan": plan_text,
            "findings": findings,
        },
    )
