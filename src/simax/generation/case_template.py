"""Fill visit templates into complete cases."""

from __future__ import annotations

import copy
import json
import random
import re
from collections.abc import Callable, Mapping
from typing import Any

from .prompts import METADATA_LEAK_PATTERNS

REGIONS = ("African", "American", "Australian", "British", "South Asian")
GENDERS = ("female", "male")
PATIENT_AGES = {"obstetrics": (19, 42), "orthopedics": (18, 85), "rheumatology": (22, 80)}
CLINICIAN_AGES = (30, 65)
FIRST_NAMES = {
    "African": {"female": ("Amara", "Nia", "Zainab", "Imani", "Folake", "Wanjiru"),
                "male": ("Kwame", "Tunde", "Emeka", "Kofi", "Sipho", "Juma")},
    "American": {"female": ("Jennifer", "Ashley", "Maria", "Megan", "Linda", "Tasha"),
                 "male": ("Michael", "Tyler", "Carlos", "Brandon", "Robert", "Darnell")},
    "Australian": {"female": ("Charlotte", "Chloe", "Isla", "Sophie", "Kylie", "Mia"),
                   "male": ("Jack", "Lachlan", "Noah", "Cooper", "Liam", "Bradley")},
    "British": {"female": ("Olivia", "Emily", "Amelia", "Hannah", "Priya", "Grace"),
                "male": ("Oliver", "George", "Harry", "James", "Callum", "Mohammed")},
    "South Asian": {"female": ("Ananya", "Priya", "Aisha", "Divya", "Fatima", "Kavya"),
                    "male": ("Arjun", "Rahul", "Imran", "Vikram", "Sanjay", "Rohan")},
}
PHASES = ("opening", "history", "medications and daily life", "examination",
          "results and assessment", "plan", "closing")
CASE_FIELDS = ("visit_stage", "chief_complaint", "clinical_facts", "agenda")

MANNER_WORDS = re.compile(
    r"\b(anxious|nervous|worried|reserved|shy|talkative|chatty|friendly|rude|polite|"
    r"guarded|trusting|distrust\w*|skeptical|sceptical|assertive|passive(?!\s+(?:range|motion|movement|stretch|ROM|and|or|\/))|empath\w*|caring|"
    r"dismissive|personality|demeanou?r|communicat\w+ style|easygoing|cheerful|irritable)\b",
    re.IGNORECASE,
)
PHASE_ALIASES = {"interval history": "history"}

FILL_SYSTEM = "You write concise, medically plausible outpatient case summaries."


def _empty(value: Any) -> bool:
    return value is None or value == "" or value == [] or value == {}


def sample_demographics(template: Mapping[str, Any], rng: random.Random) -> tuple[dict, dict]:
    out = copy.deepcopy(dict(template))
    prov: dict[str, str] = {}
    dept = str(out.get("case", {}).get("department") or "").lower()
    for who in ("patient", "clinician"):
        person = out.setdefault(who, {})
        for key in ("age", "gender", "region"):
            path = f"{who}.{key}"
            if not _empty(person.get(key)):
                prov[path] = "preset"
                continue
            if key == "gender":
                person[key] = "female" if (who == "patient" and dept == "obstetrics") else rng.choice(GENDERS)
            elif key == "region":
                person[key] = rng.choice(REGIONS)
            else:
                low, high = PATIENT_AGES.get(dept, (18, 85)) if who == "patient" else CLINICIAN_AGES
                person[key] = rng.randint(low, high)
            prov[path] = "sampled"
    patient = out["patient"]
    if _empty(patient.get("first_name")):
        patient["first_name"] = rng.choice(FIRST_NAMES[patient["region"]][patient["gender"]])
        prov["patient.first_name"] = "sampled"
    else:
        prov["patient.first_name"] = "preset"
    return out, prov


def check_consistency(resolved: Mapping[str, Any]) -> list[str]:
    problems = []
    patient, case = resolved.get("patient", {}), resolved.get("case", {})
    if str(case.get("department", "")).lower() == "obstetrics":
        if patient.get("gender") != "female":
            problems.append("obstetrics patient must be female")
        if not 15 <= int(patient.get("age", 0)) <= 50:
            problems.append("obstetrics patient age out of range")
    for who in ("patient", "clinician"):
        person = resolved.get(who, {})
        if person.get("region") not in REGIONS:
            problems.append(f"{who}.region not one of {REGIONS}")
        if person.get("gender") not in GENDERS:
            problems.append(f"{who}.gender not one of {GENDERS}")
    for key in ("chief_complaint", "visit_stage"):
        if _empty(case.get(key)):
            problems.append(f"case.{key} missing")
    facts = case.get("clinical_facts")
    if not isinstance(facts, list) or len(facts) < 5:
        problems.append("case.clinical_facts needs at least 5 facts")
    agenda = case.get("agenda")
    if not isinstance(agenda, Mapping) or set(agenda) - set(PHASES):
        problems.append(f"case.agenda keys must be among {PHASES}")
    if _empty(resolved.get("patient", {}).get("first_name")):
        problems.append("patient.first_name missing")
    text = json.dumps({k: case.get(k) for k in CASE_FIELDS}, ensure_ascii=False)
    hits = sorted({m.group(0).lower() for m in MANNER_WORDS.finditer(text)})
    if hits:
        problems.append(f"manner/personality wording in case: {hits}")
    leaks = [p for p in METADATA_LEAK_PATTERNS if p in text.lower()]
    if leaks:
        problems.append(f"words the dialogue prompt does not allow: {leaks}")
    return problems


def render_fill_prompt(partial: Mapping[str, Any], requirements: list[str]) -> str:
    case = partial.get("case", {})
    missing = [k for k in CASE_FIELDS if _empty(case.get(k))]
    fixed = {
        "patient": {k: v for k, v in partial.get("patient", {}).items() if not _empty(v)},
        "clinician": {k: v for k, v in partial.get("clinician", {}).items() if not _empty(v)},
        "case": {k: v for k, v in case.items() if not _empty(v)},
    }
    lines = [
        "Complete an outpatient case for a simulated clinic visit.",
        "",
        "Fixed (use exactly as given, do not contradict):",
        json.dumps(fixed, ensure_ascii=False, indent=1),
        "",
        f"Fill in only these fields: {', '.join(missing)}.",
        "- visit_stage: a short phrase, e.g. 'follow-up visit' or 'first visit for this problem'.",
        "- chief_complaint: one sentence, the reason for today's visit.",
        "- clinical_facts: 7-10 short facts (history, exam-relevant findings, test results, "
        "current medicines, relevant daily life). Every fact must be medically plausible for "
        "this patient's age, gender and department.",
        f"- agenda: an object whose keys are among {list(PHASES[1:-1])}; each value is a list "
        "of 1-2 topics for that part of the visit.",
        "",
        "State facts only. Do not describe anyone's personality, mood, emotions, attitude, "
        "or how they talk or behave toward each other.",
        "Do not use the words 'evaluate' or 'evaluation' (say 'work-up', 'check' or name "
        "the test instead).",
    ]
    if not _empty(case.get("condition")):
        lines += ["", "The case is about the condition given as 'condition': build the chief complaint and the "
                  "facts on it. Do not write ICD codes."]
    if requirements:
        lines += ["", "The case must also satisfy:"] + [f"- {r}" for r in requirements]
    lines += ["", 'Reply with ONLY a JSON object with the requested fields, e.g. '
              '{"visit_stage": "...", "chief_complaint": "...", "clinical_facts": ["..."], '
              '"agenda": {"history": ["..."]}}.']
    return "\n".join(lines)


def resolve_template(
    template: Mapping[str, Any],
    *,
    seed: int,
    requirements: list[str] | None = None,
    complete: Callable[[str, str], str] | None = None,
    max_attempts: int = 3,
) -> dict[str, Any]:
    if "case" not in template:
        template = {"case": template}
    template = copy.deepcopy(dict(template))
    agenda = template["case"].get("agenda")
    if isinstance(agenda, Mapping):
        template["case"]["agenda"] = {PHASE_ALIASES.get(k, k): v for k, v in agenda.items()}
    rng = random.Random(seed)
    resolved, prov = sample_demographics(template, rng)
    case = resolved.setdefault("case", {})
    patient = resolved["patient"]
    preset_case = {k: copy.deepcopy(v) for k, v in case.items() if not _empty(v)}
    for key in case:
        if not _empty(case[key]):
            prov[f"case.{key}"] = "preset"
    needs_fill = any(_empty(case.get(k)) for k in CASE_FIELDS)
    if needs_fill:
        if complete is None:
            raise ValueError("template has empty clinical fields but no model to fill them")
        prompt = render_fill_prompt(resolved, list(requirements or []))
        problems: list[str] = []
        for _ in range(max_attempts):
            try:
                reply = json.loads(_json_object(complete(FILL_SYSTEM, prompt)))
            except (ValueError, TypeError):
                problems = ["reply is not a JSON object"]
                continue
            trial = copy.deepcopy(resolved)
            for key in CASE_FIELDS:
                if _empty(trial["case"].get(key)) and key in reply:
                    trial["case"][key] = reply[key]
            trial["case"].update(preset_case)
            problems = check_consistency(trial)
            if not problems:
                for key in CASE_FIELDS:
                    prov.setdefault(f"case.{key}", "ai")
                resolved = trial
                break
        else:
            raise ValueError(f"could not fill template: {problems}")
    else:
        problems = check_consistency(resolved)
        if problems:
            raise ValueError(f"preset case is inconsistent: {problems}")
    resolved["provenance"] = prov
    return resolved


def _json_object(text: str) -> str:
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("no JSON object in reply")
    return text[start:end + 1]


def render_people(resolved: Mapping[str, Any]) -> str:
    p, c = resolved["patient"], resolved["clinician"]
    return "\n".join([
        "People",
        f"- Patient: {p['first_name']}, {p['age']}-year-old {'woman' if p['gender'] == 'female' else 'man'}, "
        f"{p['region']} English speaker.",
        f"- Doctor: {c['age']}-year-old {'woman' if c['gender'] == 'female' else 'man'}, "
        f"{c['region']} English speaker.",
    ])
