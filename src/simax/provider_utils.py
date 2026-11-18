"""Wall-clock timeout for provider calls."""

from __future__ import annotations

import threading
from typing import Any, Callable


def call_with_wall_clock(
    fn: Callable[..., Any],
    *args: Any,
    timeout_s: float,
    **kwargs: Any,
) -> Any:
    if timeout_s <= 0:
        raise ValueError("timeout_s must be positive")
    outcome: dict[str, Any] = {}

    def _target() -> None:
        try:
            outcome["value"] = fn(*args, **kwargs)
        except BaseException as exc:
            outcome["error"] = exc

    worker = threading.Thread(target=_target, daemon=True, name="provider-call")
    worker.start()
    worker.join(timeout=timeout_s)
    if worker.is_alive():
        raise TimeoutError(
            f"provider call exceeded {timeout_s:g}s wall clock; the worker "
            "thread was abandoned (daemon) and its connection is reclaimed "
            "when the process exits"
        )
    if "error" in outcome:
        raise outcome["error"]
    return outcome["value"]
