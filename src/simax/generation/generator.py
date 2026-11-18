"""Model clients, usage ledger and dialogue generation."""

from __future__ import annotations

import json
import os
import re
import subprocess
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

DEFAULT_TOGETHER_BASE_URL = "https://api.together.xyz/v1"
DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"
DEFAULT_MODEL = "zai-org/GLM-5.3-Flash"
DEFAULT_TEMPERATURE = 0.4
DEFAULT_MAX_TOKENS = 2048
DEFAULT_MAX_RETRIES = 2

REPO_ROOT = Path(__file__).resolve().parents[3]

_VALID_SPEAKERS = ("doctor", "patient")


class MalformedOutputError(ValueError):
    pass


@dataclass(frozen=True)
class ModelReply:
    content: str
    latency_ms: float
    model: str
    finish_reason: str | None
    usage: Mapping[str, Any] | None = None


class GenerationProviderError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        attempts: int = 0,
        provider_retry_delays: list[float] | None = None,
    ) -> None:
        super().__init__(message)
        self.attempts = attempts
        self.provider_retry_delays = list(provider_retry_delays or [])


class ChatClient(Protocol):
    def chat(
        self,
        *,
        system: str,
        user: str,
        model: str,
        temperature: float,
        seed: int,
        max_tokens: int,
        context: Mapping[str, str] | None = None,
    ) -> ModelReply: ...


@dataclass
class GenerationConfig:
    provider: str = "together"
    model: str = DEFAULT_MODEL
    temperature: float = DEFAULT_TEMPERATURE
    generation_seed: int = 11
    max_tokens: int = DEFAULT_MAX_TOKENS
    max_retries: int = DEFAULT_MAX_RETRIES
    max_revisions: int = 2

    @classmethod
    def from_config(
        cls, config: Mapping[str, Any], *, apply_env_overrides: bool = True
    ) -> "GenerationConfig":
        gen = config.get("generation", {}) if isinstance(config, Mapping) else {}
        cfg = cls(
            provider=str(gen.get("provider", "together")),
            model=str(gen.get("model", DEFAULT_MODEL)),
            temperature=float(gen.get("temperature", DEFAULT_TEMPERATURE)),
            generation_seed=int(gen.get("generation_seeds", [11])[0]),
            max_tokens=int(gen.get("max_tokens", DEFAULT_MAX_TOKENS)),
            max_retries=int(gen.get("max_retries", DEFAULT_MAX_RETRIES)),
            max_revisions=int(
                config.get("gates", {}).get("max_revisions", gen.get("max_revisions", 2))
            ),
        )
        if apply_env_overrides:
            cfg = cfg.with_env_overrides()
        return cfg

    def with_env_overrides(self) -> "GenerationConfig":
        env = os.environ
        model = env.get("SIMAX_MODEL", self.model)
        temperature = float(env["SIMAX_TEMPERATURE"]) if "SIMAX_TEMPERATURE" in env else self.temperature
        seed = int(env["SIMAX_GENERATION_SEED"]) if "SIMAX_GENERATION_SEED" in env else self.generation_seed
        max_tokens = int(env["SIMAX_MAX_TOKENS"]) if "SIMAX_MAX_TOKENS" in env else self.max_tokens
        max_retries = int(env["SIMAX_MAX_RETRIES"]) if "SIMAX_MAX_RETRIES" in env else self.max_retries
        return GenerationConfig(
            provider=self.provider,
            model=model,
            temperature=temperature,
            generation_seed=seed,
            max_tokens=max_tokens,
            max_retries=max_retries,
            max_revisions=self.max_revisions,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "temperature": self.temperature,
            "generation_seed": self.generation_seed,
            "max_tokens": self.max_tokens,
            "max_retries": self.max_retries,
            "max_revisions": self.max_revisions,
        }


@dataclass
class GenerationResult:
    turns: list[dict[str, Any]] | None
    status: str
    attempts: int
    latency_ms_total: float
    finish_reasons: list[str | None] = field(default_factory=list)
    malformed_errors: list[str] = field(default_factory=list)
    raw_outputs: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "attempts": self.attempts,
            "latency_ms_total": self.latency_ms_total,
            "finish_reasons": list(self.finish_reasons),
            "malformed_errors": list(self.malformed_errors),
            "raw_outputs": list(self.raw_outputs),
        }


def git_commit_info(repo_root: Path | None = None) -> tuple[str, bool]:
    root = Path(repo_root) if repo_root is not None else REPO_ROOT
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout
        return commit, bool(status.strip())
    except (OSError, subprocess.SubprocessError):
        return "unknown", False


from simax.provider_utils import call_with_wall_clock


def _reasoning_enabled() -> bool:
    return os.environ.get("SIMAX_REASONING", "0") == "1"


class TogetherChatClient:
    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str = DEFAULT_TOGETHER_BASE_URL,
        timeout_s: float = 120.0,
        call_timeout_s: float | None = None,
        usage_ledger: UsageLedger | None = None,
    ) -> None:
        api_key = api_key or os.environ.get("TOGETHER_API_KEY")
        if not api_key:
            raise RuntimeError(
                "TOGETHER_API_KEY is not set; the Together adapter needs it in "
                "the environment (never hard-code keys)"
            )
        from openai import OpenAI

        resolved_call_timeout = float(
            call_timeout_s
            if call_timeout_s is not None
            else os.environ.get("SIMAX_CALL_TIMEOUT_S", "180")
        )
        if timeout_s is None:
            timeout_s = resolved_call_timeout
        self._client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout_s)
        self.base_url = base_url
        self.call_timeout_s = resolved_call_timeout

    def chat(
        self,
        *,
        system: str,
        user: str,
        model: str,
        temperature: float,
        seed: int,
        max_tokens: int,
        context: Mapping[str, str] | None = None,
    ) -> ModelReply:
        del context
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        started = time.perf_counter()
        response = call_with_wall_clock(
            self._client.chat.completions.create,
            timeout_s=self.call_timeout_s,
            model=model,
            messages=messages,
            temperature=temperature,
            seed=seed,
            max_tokens=max_tokens,
            extra_body={"reasoning": {"enabled": _reasoning_enabled()}},
        )
        latency_ms = (time.perf_counter() - started) * 1000.0
        choice = response.choices[0]
        content = choice.message.content or ""
        return ModelReply(
            content=content,
            latency_ms=latency_ms,
            model=getattr(response, "model", model),
            finish_reason=getattr(choice, "finish_reason", None),
        )


def _usage_dict(response: Any) -> dict[str, Any] | None:
    def _get(obj: Any, name: str, default: Any = None) -> Any:
        if obj is None:
            return default
        if isinstance(obj, Mapping):
            return obj.get(name, default)
        return getattr(obj, name, default)

    usage = _get(response, "usage")
    if usage is None:
        return None
    completion_details = _get(usage, "completion_tokens_details") or {}
    prompt_details = _get(usage, "prompt_tokens_details") or {}
    return {
        "prompt_tokens": _get(usage, "prompt_tokens"),
        "completion_tokens": _get(usage, "completion_tokens"),
        "total_tokens": _get(usage, "total_tokens"),
        "reasoning_tokens": _get(completion_details, "reasoning_tokens"),
        "cached_tokens": _get(prompt_details, "cached_tokens"),
    }


class UsageLedger:
    def __init__(self, path: Path | str | None = None) -> None:
        self.rows: list[dict[str, Any]] = []
        self.path = Path(path) if path is not None else None

    def record(
        self,
        *,
        model: str,
        context: Mapping[str, Any] | None,
        usage: Mapping[str, Any] | None,
        latency_ms: float,
        finish_reason: str | None,
        ok: bool = True,
    ) -> None:
        row: dict[str, Any] = {
            "model": model,
            "ok": ok,
            "latency_ms": round(latency_ms, 1),
            "finish_reason": finish_reason,
        }
        if context:
            for key in ("sample_id", "method", "block_id", "construct", "target_level"):
                if key in context:
                    row[key] = context[key]
        if usage:
            row.update({k: v for k, v in usage.items() if v is not None})
        self.rows.append(row)
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, sort_keys=True) + "\n")

    def reload_from_disk(self) -> None:
        if self.path is None or not self.path.exists():
            return
        rows: list[dict[str, Any]] = []
        with open(self.path, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        self.rows = rows

    def totals(self) -> dict[str, Any]:
        numeric = ("prompt_tokens", "completion_tokens", "reasoning_tokens",
                   "total_tokens", "cached_tokens")

        def _zero() -> dict[str, int]:
            return {key: 0 for key in numeric}

        def _add(bucket: dict[str, int], row: Mapping[str, Any]) -> None:
            for key in numeric:
                value = row.get(key)
                if isinstance(value, int):
                    bucket[key] += value

        overall = _zero()
        overall["calls"] = 0
        by_construct: dict[str, dict[str, int]] = {}
        by_level: dict[str, dict[str, int]] = {}
        for row in self.rows:
            _add(overall, row)
            overall["calls"] += 1
            construct = row.get("construct")
            if construct is not None:
                bucket = by_construct.setdefault(str(construct), {**_zero(), "calls": 0})
                _add(bucket, row)
                bucket["calls"] += 1
            level = row.get("target_level")
            if level is not None:
                bucket = by_level.setdefault(str(level), {**_zero(), "calls": 0})
                _add(bucket, row)
                bucket["calls"] += 1
        return {
            "overall": overall,
            "by_construct": dict(sorted(by_construct.items())),
            "by_target_level": dict(sorted(by_level.items())),
        }


class OpenAIChatClient:
    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str = DEFAULT_OPENAI_BASE_URL,
        timeout_s: float = 120.0,
        call_timeout_s: float | None = None,
        usage_ledger: UsageLedger | None = None,
        send_temperature: bool = False,
    ) -> None:
        api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set; the OpenAI adapter needs it in the "
                "environment (never hard-code keys)"
            )
        from openai import OpenAI

        resolved_call_timeout = float(
            call_timeout_s
            if call_timeout_s is not None
            else os.environ.get("SIMAX_CALL_TIMEOUT_S", "180")
        )
        self._client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout_s)
        self.base_url = base_url
        self.call_timeout_s = resolved_call_timeout
        self.usage_ledger = usage_ledger
        self.send_temperature = send_temperature

    def chat(
        self,
        *,
        system: str,
        user: str,
        model: str,
        temperature: float,
        seed: int,
        max_tokens: int,
        context: Mapping[str, str] | None = None,
    ) -> ModelReply:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        request: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "seed": seed,
            "max_completion_tokens": max_tokens,
        }
        if self.send_temperature:
            request["temperature"] = temperature
        effort = os.environ.get("SIMAX_REASONING_EFFORT")
        if effort:
            request["reasoning_effort"] = effort
        started = time.perf_counter()
        try:
            response = call_with_wall_clock(
                self._client.chat.completions.create,
                timeout_s=self.call_timeout_s,
                **request,
            )
        except BaseException:
            latency_ms = (time.perf_counter() - started) * 1000.0
            if self.usage_ledger is not None:
                self.usage_ledger.record(
                    model=model, context=context, usage=None,
                    latency_ms=latency_ms, finish_reason=None, ok=False,
                )
            raise
        latency_ms = (time.perf_counter() - started) * 1000.0
        choice = response.choices[0]
        content = choice.message.content or ""
        usage = _usage_dict(response)
        if self.usage_ledger is not None:
            self.usage_ledger.record(
                model=getattr(response, "model", model), context=context,
                usage=usage, latency_ms=latency_ms,
                finish_reason=getattr(choice, "finish_reason", None),
            )
        return ModelReply(
            content=content,
            latency_ms=latency_ms,
            model=getattr(response, "model", model),
            finish_reason=getattr(choice, "finish_reason", None),
            usage=usage,
        )


def _normalize_text(text: str) -> str:
    text = text.strip()
    lowered = text.lower()
    for prefix in ("doctor:", "patient:", "d:", "p:"):
        if lowered.startswith(prefix):
            text = text[len(prefix):].lstrip()
            break
    return text


def _extract_json_object(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        while lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    try:
        json.loads(stripped)
        return stripped
    except json.JSONDecodeError:
        pass
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start != -1 and end > start:
        candidate = stripped[start : end + 1]
        json.loads(candidate)
        return candidate
    start = stripped.find("[")
    end = stripped.rfind("]")
    if start != -1 and end > start:
        candidate = stripped[start : end + 1]
        json.loads(candidate)
        return candidate
    raise json.JSONDecodeError("no JSON object found", stripped[:200], 0)


def parse_dialogue_payload(
    text: str, *, expected_turn_count: int | None = None
) -> list[dict[str, Any]]:
    try:
        payload = json.loads(_extract_json_object(text))
    except json.JSONDecodeError as exc:
        raise MalformedOutputError(f"reply is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict) or "turns" not in payload:
        raise MalformedOutputError("reply JSON must be an object with a 'turns' list")
    raw_turns = payload["turns"]
    if not isinstance(raw_turns, list) or not raw_turns:
        raise MalformedOutputError("'turns' must be a non-empty list")

    turns: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_turns):
        if not isinstance(raw, dict) or "speaker" not in raw or "text" not in raw:
            raise MalformedOutputError(
                f"turn {index} must be an object with 'speaker' and 'text'"
            )
        speaker = str(raw["speaker"]).strip().lower()
        if speaker not in _VALID_SPEAKERS:
            raise MalformedOutputError(
                f"turn {index} has invalid speaker {raw['speaker']!r}; "
                f"expected one of {_VALID_SPEAKERS}"
            )
        text = _normalize_text(str(raw["text"]))
        if not text:
            raise MalformedOutputError(f"turn {index} has empty text")
        turns.append({"turn_index": index, "speaker": speaker, "text": text})

    if turns[0]["speaker"] != "doctor":
        raise MalformedOutputError("dialogue must start with the doctor")
    if turns[-1]["speaker"] != "doctor":
        raise MalformedOutputError("dialogue must end with the doctor")
    for index in range(1, len(turns)):
        if turns[index]["speaker"] == turns[index - 1]["speaker"]:
            raise MalformedOutputError(
                f"turn {index} breaks strict speaker alternation"
            )
    if expected_turn_count is not None and len(turns) != expected_turn_count:
        raise MalformedOutputError(
            f"dialogue must have exactly {expected_turn_count} turns; got {len(turns)}"
        )
    return turns


def parse_revision_payload(
    text: str, *, allowed_indices: Sequence[int]
) -> dict[int, str]:
    try:
        payload = json.loads(_extract_json_object(text))
    except json.JSONDecodeError as exc:
        raise MalformedOutputError(f"revision reply is not valid JSON: {exc}") from exc
    if isinstance(payload, list):
        raw_turns = payload
    elif isinstance(payload, dict) and "turns" in payload:
        raw_turns = payload["turns"]
    else:
        raise MalformedOutputError(
            "revision JSON must be an object with a 'turns' list or a bare list"
        )
    if not isinstance(raw_turns, list) or not raw_turns:
        raise MalformedOutputError(
            "revision reply must contain a non-empty list of turns"
        )
    allowed = set(int(index) for index in allowed_indices)
    revised: dict[int, str] = {}
    for raw in raw_turns:
        if not isinstance(raw, dict) or "turn_index" not in raw or "text" not in raw:
            raise MalformedOutputError(
                "each revision entry must have 'turn_index' and 'text'"
            )
        try:
            turn_index = int(raw["turn_index"])
        except (TypeError, ValueError) as exc:
            raise MalformedOutputError(
                f"revision turn_index is not an integer: {raw['turn_index']!r}"
            ) from exc
        if turn_index not in allowed:
            raise MalformedOutputError(
                f"revision attempted to change non-offending turn {turn_index}"
            )
        text_value = _normalize_text(str(raw["text"]))
        if not text_value:
            raise MalformedOutputError(f"revision for turn {turn_index} is empty")
        revised[turn_index] = text_value
    if not revised:
        raise MalformedOutputError("revision reply contains no turns")
    missing = allowed - set(revised)
    if missing:
        raise MalformedOutputError(
            f"revision reply is missing offending turns: {sorted(missing)}"
        )
    return revised


def _retry_user_message(reason: str) -> str:
    return (
        "Your previous reply did not meet the required output format: "
        f"{reason}\nReply again with ONLY the corrected JSON object, with no "
        "markdown fences and no text outside the JSON."
    )


_TRANSIENT_PROVIDER_PAT = re.compile(
    r"timed? ?out|connection|rate limit|429|5\d\d|service unavailable|"
    r"no available server|overloaded|temporarily unavailable",
    re.IGNORECASE,
)


def _is_transient_provider_error(exc: BaseException) -> bool:
    if isinstance(exc, (ValueError, TypeError, MalformedOutputError)):
        return False
    name = type(exc).__name__.lower()
    if any(token in name for token in ("timeout", "connection", "ratelimit")):
        return True
    return bool(_TRANSIENT_PROVIDER_PAT.search(str(exc)))


def _retry_after_seconds(exc: BaseException, attempt: int) -> float:
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    raw = None
    try:
        if headers is not None:
            raw = headers.get("retry-after")
    except Exception:
        raw = None
    if raw is not None:
        try:
            return max(0.0, min(30.0, float(raw)))
        except (TypeError, ValueError):
            pass
    return min(30.0, 2.0 * (2**attempt))


def _chat_with_provider_retry(
    client: ChatClient,
    *,
    system: str,
    user: str,
    config: GenerationConfig,
    context: Mapping[str, str] | None,
    max_provider_retries: int,
    retry_log: list[dict[str, Any]],
) -> Any:
    attempt = 0
    while True:
        try:
            return client.chat(
                system=system,
                user=user,
                model=config.model,
                temperature=config.temperature,
                seed=config.generation_seed,
                max_tokens=config.max_tokens,
                context=context,
            )
        except Exception as exc:
            if not _is_transient_provider_error(exc):
                raise
            if attempt >= max_provider_retries:
                raise
            delay = _retry_after_seconds(exc, attempt)
            retry_log.append(delay)
            time.sleep(delay)
            attempt += 1


def generate_dialogue(
    client: ChatClient,
    *,
    compiled_prompt: Any,
    config: GenerationConfig,
    context: Mapping[str, str] | None = None,
    expected_turn_count: int | None = None,
) -> GenerationResult:
    finish_reasons: list[str | None] = []
    malformed_errors: list[str] = []
    raw_outputs: list[str] = []
    provider_retry_delays: list[float] = []
    latency_total = 0.0
    max_attempts = 1 + max(config.max_retries, 0)
    max_provider_retries = int(
        os.environ.get("SIMAX_PROVIDER_RETRIES", "4")
    )

    for attempt in range(max_attempts):
        if attempt == 0:
            user = compiled_prompt.user
        else:
            user = (
                _retry_user_message(malformed_errors[-1])
                + "\n\nOriginal request:\n"
                + compiled_prompt.user
            )
        try:
            reply = _chat_with_provider_retry(
                client,
                system=compiled_prompt.system,
                user=user,
                config=config,
                context=context,
                max_provider_retries=max_provider_retries,
                retry_log=provider_retry_delays,
            )
        except Exception as exc:
            if not _is_transient_provider_error(exc):
                raise
            provider_retry_delays.append(-1.0)
            raise GenerationProviderError(
                f"{type(exc).__name__}: {exc}",
                attempts=attempt + 1,
                provider_retry_delays=provider_retry_delays,
            ) from exc
        latency_total += reply.latency_ms
        finish_reasons.append(reply.finish_reason)
        raw_outputs.append(reply.content)
        try:
            turns = parse_dialogue_payload(
                reply.content, expected_turn_count=expected_turn_count
            )
        except MalformedOutputError as exc:
            malformed_errors.append(str(exc))
            continue
        return GenerationResult(
            turns=turns,
            status="ok",
            attempts=attempt + 1,
            latency_ms_total=latency_total,
            finish_reasons=finish_reasons,
            malformed_errors=malformed_errors,
            raw_outputs=raw_outputs,
        )
    return GenerationResult(
        turns=None,
        status="failed_malformed",
        attempts=max_attempts,
        latency_ms_total=latency_total,
        finish_reasons=finish_reasons,
        malformed_errors=malformed_errors,
        raw_outputs=raw_outputs,
    )


@dataclass
class RevisionResult:
    revised: dict[int, str] | None
    status: str
    attempts: int
    latency_ms_total: float
    malformed_errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "attempts": self.attempts,
            "latency_ms_total": self.latency_ms_total,
            "malformed_errors": list(self.malformed_errors),
            "revised": {str(k): v for k, v in (self.revised or {}).items()},
        }


def revise_turns(
    client: ChatClient,
    *,
    compiled_revision_prompt: Any,
    offending_indices: Sequence[int],
    config: GenerationConfig,
    context: Mapping[str, str] | None = None,
) -> RevisionResult:
    allowed = sorted(int(index) for index in offending_indices)
    malformed_errors: list[str] = []
    latency_total = 0.0
    max_attempts = 1 + max(config.max_retries, 0)
    for attempt in range(max_attempts):
        user = compiled_revision_prompt.user
        if attempt > 0:
            user = (
                _retry_user_message(malformed_errors[-1])
                + "\n\nOriginal request:\n"
                + compiled_revision_prompt.user
            )
        reply = client.chat(
            system=compiled_revision_prompt.system,
            user=user,
            model=config.model,
            temperature=config.temperature,
            seed=config.generation_seed,
            max_tokens=config.max_tokens,
            context=context,
        )
        latency_total += reply.latency_ms
        try:
            revised = parse_revision_payload(
                reply.content, allowed_indices=allowed
            )
        except MalformedOutputError as exc:
            malformed_errors.append(str(exc))
            continue
        return RevisionResult(
            revised=revised,
            status="ok",
            attempts=attempt + 1,
            latency_ms_total=latency_total,
            malformed_errors=malformed_errors,
        )
    return RevisionResult(
        revised=None,
        status="failed_malformed",
        attempts=max_attempts,
        latency_ms_total=latency_total,
        malformed_errors=malformed_errors,
    )
