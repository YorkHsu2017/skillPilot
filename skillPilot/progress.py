"""Pure-stdlib terminal progress bar and spinner utilities."""

from __future__ import annotations

import shutil
import sys
from typing import Callable

# ── spinner frames ────────────────────────────────────────────────────────────
_SPINNER_FRAMES = ("⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏")

# ── ANSI escapes ──────────────────────────────────────────────────────────────
CLEAR_LINE = "\033[2K\r"
RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
ORANGE = "\033[38;5;214m"
ORANGE_BG = "\033[48;5;214m"
GREEN = "\033[32m"
RED = "\033[31m"
CYAN = "\033[36m"

# ── alternate screen ──────────────────────────────────────────────────────────
ALT_SCREEN_ENTER = "\033[?1049h\033[H\033[2J"
ALT_SCREEN_EXIT = "\033[?1049l"


def _is_tty() -> bool:
    """True when stderr is a real terminal (not piped or redirected)."""
    return sys.stderr.isatty()


def _width() -> int:
    try:
        cols, _ = shutil.get_terminal_size()
        return max(cols, 40)
    except (OSError, ValueError):
        return 80


# ── public helpers ────────────────────────────────────────────────────────────


def enter_alternate_screen() -> None:
    """Switch to the alternate terminal screen buffer (like VIM)."""
    sys.stderr.write(ALT_SCREEN_ENTER)
    sys.stderr.flush()


def exit_alternate_screen() -> None:
    """Restore the original terminal screen buffer."""
    sys.stderr.write(ALT_SCREEN_EXIT)
    sys.stderr.flush()


def format_thought(text: str) -> str:
    """Render a thought bubble in orange."""
    return f"{DIM}{ORANGE}💭 {text}{RESET}"


def format_thinking() -> str:
    """Render a thinking indicator."""
    return f"{DIM}{ORANGE}Thinking...{RESET}"


def spinner_frame(tick: int) -> str:
    """Return a spinner character for the given tick index."""
    frame = _SPINNER_FRAMES[tick % len(_SPINNER_FRAMES)]
    return f"{ORANGE}{frame}{RESET}"


def clear_terminal_line() -> str:
    """ANSI escape to clear the current terminal line."""
    return CLEAR_LINE


def render_progress_bar(
    current: int,
    total: int,
    *,
    label: str = "",
    width: int | None = None,
    show_eta: bool = False,
) -> str:
    """Return a single-line progress bar string (no trailing newline).

    Example: ``[task] ⠋ [████████░░░░░░] 8/12 (67%)``
    """
    w = width or _width()
    ratio = min(1.0, max(0.0, current / max(total, 1)))
    pct = int(ratio * 100)

    prefix = f"[{label}] " if label else ""
    suffix = f" {current}/{total} ({pct}%)"

    bar_available = w - len(prefix) - len(suffix) - 3  # for spinner + space
    bar_available = max(5, bar_available)
    filled = int(ratio * bar_available)
    bar = "█" * filled + "░" * (bar_available - filled)

    return f"{prefix}{bar}{suffix}"


def render_progress_line(
    current: int,
    total: int,
    *,
    tick: int,
    label: str = "",
    width: int | None = None,
    extra: str = "",
) -> str:
    """Render a full progress line with spinner, bar, and optional extra text.

    Returns a string suitable for writing to stderr with ``\\r`` followed by
    the bar, plus ``\\033[K`` already included.
    """
    bar = render_progress_bar(current, total, label=label, width=width)
    spinner = spinner_frame(tick)
    line = f"{spinner} {bar}"
    if extra:
        bar_width = _width()
        available = max(bar_width - len(line) - 2, 8)
        extra_truncated = _truncate_ansi(extra, available)
        line += f"  {extra_truncated}"
    return CLEAR_LINE + line


def render_tool_line(
    tool_name: str,
    args_summary: str,
    *,
    tick: int,
) -> str:
    """Render a concise one-line tool execution status."""
    spinner = spinner_frame(tick)
    name = tool_name[:20]
    summary = _truncate_ansi(args_summary, _width() - len(name) - 15)
    return f"{CLEAR_LINE}{spinner} [{name}] {summary}"


def _truncate_ansi(text: str, max_chars: int) -> str:
    """Truncate text to max_chars, stripping ANSI sequences from the count."""
    if max_chars <= 0:
        return ""
    visible: list[str] = []
    visible_len = 0
    in_escape = False
    for ch in text:
        if in_escape:
            visible.append(ch)
            if ch in "mK":
                in_escape = False
            continue
        if ch == "\033":
            in_escape = True
            visible.append(ch)
            continue
        if visible_len >= max_chars:
            break
        visible.append(ch)
        visible_len += 1
    return "".join(visible)