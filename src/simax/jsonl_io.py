"""JSONL reading and writing helpers."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any, Mapping


class JsonlIOError(RuntimeError):
    pass


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def fsync_dir(path: str | os.PathLike[str]) -> None:
    try:
        dir_fd = os.open(str(path), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(dir_fd)
    except OSError:
        pass
    finally:
        os.close(dir_fd)


def encode_jsonl_line(record: Mapping[str, Any]) -> str:
    return json.dumps(record, sort_keys=True, ensure_ascii=False)


def append_jsonl(path: str | os.PathLike[str], record: Mapping[str, Any]) -> None:
    line = encode_jsonl_line(record)
    parent = os.path.dirname(os.fspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(line + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    if parent:
        fsync_dir(parent)


def write_jsonl(
    path: str | os.PathLike[str],
    records: "list[Mapping[str, Any]] | tuple[Mapping[str, Any], ...]",
) -> int:
    parent = os.path.dirname(os.fspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    count = 0
    with open(path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(encode_jsonl_line(record) + "\n")
            count += 1
        handle.flush()
        os.fsync(handle.fileno())
    if parent:
        fsync_dir(parent)
    return count


def read_jsonl_records(path: str | os.PathLike[str]) -> tuple[list[dict[str, Any]], list[str]]:
    file_path = os.fspath(path)
    if not os.path.isfile(file_path):
        raise JsonlIOError(f"JSONL file not found: {file_path}")
    records: list[dict[str, Any]] = []
    warnings: list[str] = []
    with open(file_path, "r", encoding="utf-8") as handle:
        for line_no, raw_line in enumerate(handle, 1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                warnings.append(f"{os.path.basename(file_path)}:{line_no}: skipped unparsable JSON line")
                continue
            if not isinstance(record, dict):
                warnings.append(f"{os.path.basename(file_path)}:{line_no}: skipped non-object record")
                continue
            records.append(record)
    return records, warnings
