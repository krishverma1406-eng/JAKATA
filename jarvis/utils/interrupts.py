"""Keyboard interrupt handling helpers for CLI loops."""

from __future__ import annotations


def register_keyboard_interrupt(interrupt_count: int) -> tuple[int, bool]:
    updated = interrupt_count + 1
    return updated, updated >= 2
