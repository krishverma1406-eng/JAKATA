"""Shared intent helpers for planner and agent routing."""

from __future__ import annotations


_EXPLICIT_CODE_WRITER_MARKERS = (
    "create a tool",
    "build a tool",
    "new jarvis tool",
    "make a tool",
    "write tool code",
    "generate tool code",
    "scaffold a tool",
    "repair tool file",
    "fix tool file",
    "code_writer",
    "tool file",
    "tool module",
)


def explicit_code_writer_request(text: str) -> bool:
    lowered = str(text or "").lower().strip()
    return any(marker in lowered for marker in _EXPLICIT_CODE_WRITER_MARKERS)
