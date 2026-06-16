from __future__ import annotations

import json
import re
import shlex
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict


ToolHandler = Callable[..., Dict[str, Any]]
OutputSink = Callable[[str], None]  # real-time line-by-line output from tools


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    schema: dict[str, Any]
    handler: ToolHandler


class WorkspaceTools:
    def __init__(self, root: str | Path, *, output_sink: OutputSink | None = None) -> None:
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self._output_sink = output_sink or (lambda _line: None)

    def _resolve(self, path: str | Path) -> Path:
        raw = Path(path).expanduser()
        resolved = raw.resolve() if raw.is_absolute() else (self.root / raw).resolve()
        try:
            resolved.relative_to(self.root)
        except ValueError as exc:
            raise ValueError(f"Path escapes workspace: {path}") from exc
        return resolved

    @staticmethod
    def _ignored(path: Path) -> bool:
        ignored_parts = {".git", ".skillPilot", "__pycache__", ".pytest_cache"}
        return any(part in ignored_parts for part in path.parts)

    @staticmethod
    def _limit(text: str, max_chars: int = 12000) -> str:
        if len(text) <= max_chars:
            return text
        return text[:max_chars] + f"\n... <truncated {len(text) - max_chars} chars>"

    def list_files(self, path: str = ".", pattern: str = "**/*", max_results: int = 200) -> dict[str, Any]:
        base = self._resolve(path)
        if not base.exists():
            raise FileNotFoundError(str(base))
        files = []
        iterator = [base] if base.is_file() else base.glob(pattern)
        for candidate in iterator:
            if len(files) >= max_results:
                break
            if candidate.is_file() and not self._ignored(candidate):
                files.append(str(candidate.relative_to(self.root)))
        return {"ok": True, "count": len(files), "files": sorted(files)}

    def read_file(
        self,
        path: str,
        start_line: int | None = None,
        end_line: int | None = None,
    ) -> dict[str, Any]:
        target = self._resolve(path)
        content_lines = target.read_text(encoding="utf-8").splitlines(keepends=True)
        start = max(1, int(start_line or 1))
        end = int(end_line) if end_line is not None else len(content_lines)
        selected = "".join(content_lines[start - 1 : end])
        return {
            "ok": True,
            "path": str(target.relative_to(self.root)),
            "start_line": start,
            "end_line": end,
            "line_count": len(content_lines),
            "content": self._limit(selected),
        }

    def write_file(self, path: str, content: str) -> dict[str, Any]:
        target = self._resolve(path)
        if self._ignored(target):
            raise ValueError(f"Refusing to write ignored/internal path: {path}")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return {
            "ok": True,
            "path": str(target.relative_to(self.root)),
            "bytes": len(content.encode("utf-8")),
        }

    def grep(
        self,
        pattern: str,
        path: str = ".",
        file_glob: str = "**/*",
        case_sensitive: bool = False,
        max_matches: int = 80,
    ) -> dict[str, Any]:
        flags = 0 if case_sensitive else re.IGNORECASE
        regex = re.compile(pattern, flags)
        base = self._resolve(path)
        candidates = [base] if base.is_file() else base.glob(file_glob)
        matches: list[dict[str, Any]] = []
        for candidate in candidates:
            if len(matches) >= max_matches:
                break
            if not candidate.is_file() or self._ignored(candidate):
                continue
            try:
                lines = candidate.read_text(encoding="utf-8", errors="replace").splitlines()
            except OSError:
                continue
            for line_no, line in enumerate(lines, start=1):
                if regex.search(line):
                    matches.append(
                        {
                            "path": str(candidate.relative_to(self.root)),
                            "line": line_no,
                            "text": line[:500],
                        }
                    )
                    if len(matches) >= max_matches:
                        break
        return {"ok": True, "pattern": pattern, "count": len(matches), "matches": matches}

    def edit_file(
        self,
        path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> dict[str, Any]:
        """Replace old_string with new_string in a file. old_string must occur at least once."""
        target = self._resolve(path)
        if self._ignored(target):
            raise ValueError(f"Refusing to edit ignored/internal path: {path}")
        if not target.exists():
            raise FileNotFoundError(f"File not found: {path}")
        content = target.read_text(encoding="utf-8")
        count = content.count(old_string)
        if count == 0:
            return {"ok": False, "error": "old_string not found in file", "path": str(target.relative_to(self.root))}
        if not replace_all and count > 1:
            return {
                "ok": False,
                "error": f"old_string matches {count} times; set replace_all=true to replace all, or provide more context",
                "path": str(target.relative_to(self.root)),
                "match_count": count,
            }
        new_content = content.replace(old_string, new_string)
        target.write_text(new_content, encoding="utf-8")
        replaced = count if replace_all else 1
        return {
            "ok": True,
            "path": str(target.relative_to(self.root)),
            "replacements": replaced,
            "bytes": len(new_content.encode("utf-8")),
        }

    def bash(self, command: str, timeout: int = 60) -> dict[str, Any]:
        """Run a shell command inside the workspace directory with real-time output."""
        return self._run_with_stream(command, shell=True, timeout=timeout)

    def run_python(self, path: str, args: str = "", timeout: int = 30) -> dict[str, Any]:
        target = self._resolve(path)
        command = [sys.executable, str(target)] + shlex.split(args)
        return self._run_with_stream(command, timeout=timeout)

    def run_pytest(self, args: str = "", timeout: int = 120) -> dict[str, Any]:
        command = [sys.executable, "-m", "pytest"] + shlex.split(args)
        return self._run_with_stream(command, timeout=timeout)

    def _run_with_stream(
        self,
        command: list[str] | str,
        *,
        shell: bool = False,
        timeout: int,
    ) -> dict[str, Any]:
        """Run a command with real-time line-by-line stdout/stderr streaming."""
        popen_kwargs: dict[str, Any] = {
            "cwd": self.root,
            "stdout": subprocess.PIPE,
            "stderr": subprocess.STDOUT,
            "text": True,
        }
        if shell:
            popen_kwargs["args"] = command if isinstance(command, str) else " ".join(command)
            popen_kwargs["shell"] = True
        else:
            popen_kwargs["args"] = command

        collected: list[str] = []
        sink = self._output_sink

        def _drain(stream: Any) -> None:
            for line in iter(stream.readline, ""):
                if line:
                    collected.append(line)
                    sink(line.rstrip("\n"))

        try:
            proc = subprocess.Popen(**popen_kwargs)
        except OSError as exc:
            return {"ok": False, "exit_code": None, "output": "", "error": f"OSError: {exc}"}

        drain_thread = threading.Thread(target=_drain, args=(proc.stdout,), daemon=True)
        drain_thread.start()

        try:
            proc.wait(timeout=timeout)
            drain_thread.join(timeout=2)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
            drain_thread.join(timeout=2)
            output = self._limit("".join(collected))
            return {
                "ok": False,
                "exit_code": None,
                "output": output,
                "error": f"Timed out after {timeout}s",
            }

        output = self._limit("".join(collected))
        return {
            "ok": proc.returncode == 0,
            "exit_code": proc.returncode,
            "output": output,
        }


class ToolRegistry:
    def __init__(self, tools: list[ToolSpec]) -> None:
        self._tools = {tool.name: tool for tool in tools}

    def list(self) -> list[ToolSpec]:
        return list(self._tools.values())

    def run(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        tool = self._tools.get(name)
        if tool is None:
            raise KeyError(f"Unknown tool: {name}")
        return tool.handler(**args)

    def prompt_summary(self) -> str:
        lines = []
        for tool in self.list():
            lines.append(
                json.dumps(
                    {
                        "name": tool.name,
                        "description": tool.description,
                        "schema": tool.schema,
                    },
                    ensure_ascii=False,
                )
            )
        return "\n".join(lines)


def build_default_registry(workspace: str | Path, *, output_sink: OutputSink | None = None) -> ToolRegistry:
    tools = WorkspaceTools(workspace, output_sink=output_sink)
    return ToolRegistry(
        [
            ToolSpec(
                name="list_files",
                description="List files under the workspace. Args: path='.', pattern='**/*', max_results=200.",
                schema={"path": "string", "pattern": "string", "max_results": "integer"},
                handler=tools.list_files,
            ),
            ToolSpec(
                name="read_file",
                description="Read a UTF-8 text file in the workspace, optionally by 1-based line range.",
                schema={"path": "string", "start_line": "integer?", "end_line": "integer?"},
                handler=tools.read_file,
            ),
            ToolSpec(
                name="write_file",
                description="Create or overwrite a UTF-8 text file in the workspace.",
                schema={"path": "string", "content": "string"},
                handler=tools.write_file,
            ),
            ToolSpec(
                name="edit_file",
                description="Replace old_string with new_string in a workspace file. Set replace_all=true to replace every occurrence; otherwise old_string must be unique.",
                schema={"path": "string", "old_string": "string", "new_string": "string", "replace_all": "boolean"},
                handler=tools.edit_file,
            ),
            ToolSpec(
                name="grep",
                description="Regex search file contents in the workspace.",
                schema={
                    "pattern": "string",
                    "path": "string",
                    "file_glob": "string",
                    "case_sensitive": "boolean",
                    "max_matches": "integer",
                },
                handler=tools.grep,
            ),
            ToolSpec(
                name="bash",
                description="Run a shell command inside the workspace. Args: command string, optional timeout seconds (default 60).",
                schema={"command": "string", "timeout": "integer"},
                handler=tools.bash,
            ),
            ToolSpec(
                name="run_python",
                description="Run a Python file in the workspace. Args: path, optional args string, timeout seconds.",
                schema={"path": "string", "args": "string", "timeout": "integer"},
                handler=tools.run_python,
            ),
            ToolSpec(
                name="run_pytest",
                description="Run pytest in the workspace. Args: optional args string and timeout seconds.",
                schema={"args": "string", "timeout": "integer"},
                handler=tools.run_pytest,
            ),
        ]
    )
