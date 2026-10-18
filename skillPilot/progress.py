"""Small dependency-free terminal formatting helpers."""

from __future__ import annotations

import shutil

CLEAR_LINE = "\033[2K\r"
RESET = "\033[0m"
DIM = "\033[2m"
CYAN = "\033[36m"
_SPINNER_FRAMES = ("|", "/", "-", "\\")


def enter_alternate_screen() -> None:
    """Kept as a compatibility no-op; preserving scrollback is better for agents."""


def exit_alternate_screen() -> None:
    """Kept as a compatibility no-op."""


def clear_terminal_line() -> str:
    return CLEAR_LINE


def format_thought(text: str) -> str:
    return f"{DIM}{text}{RESET}"


def format_thinking() -> str:
    return f"{DIM}Thinking...{RESET}"


def spinner_frame(tick: int) -> str:
    return _SPINNER_FRAMES[tick % len(_SPINNER_FRAMES)]


def render_progress_bar(current: int, total: int, *, label: str = "", width: int | None = None, show_eta: bool = False) -> str:
    terminal_width = width or shutil.get_terminal_size((80, 20)).columns
    ratio = min(1.0, max(0.0, current / max(total, 1)))
    prefix = f"[{label}] " if label else ""
    suffix = f" {current}/{total} ({ratio:.0%})"
    size = max(5, terminal_width - len(prefix) - len(suffix) - 2)
    filled = int(size * ratio)
    return f"{prefix}[{'=' * filled}{'.' * (size - filled)}]{suffix}"


def render_progress_line(current: int, total: int, *, tick: int, label: str = "", width: int | None = None, extra: str = "") -> str:
    line = f"{spinner_frame(tick)} {render_progress_bar(current, total, label=label, width=width)}"
    return CLEAR_LINE + line + (f"  {extra}" if extra else "")


def render_tool_line(tool_name: str, args_summary: str, *, tick: int) -> str:
    return f"{CLEAR_LINE}{spinner_frame(tick)} [{tool_name[:24]}] {args_summary[:120]}"
