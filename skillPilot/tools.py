"""Workspace-scoped tools, permission policy, and lightweight checkpoints."""

from __future__ import annotations

import base64
import json
import os
import re
import shlex
import subprocess
import sys
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

ToolHandler = Callable[..., dict[str, Any]]
OutputSink = Callable[[str], None]
ApprovalCallback = Callable[[str, str], bool]


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    schema: dict[str, Any]
    handler: ToolHandler
    mutating: bool = False

    def api_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "name": self.name,
            "description": self.description,
            "parameters": self.schema,
            "strict": True,
        }


def _object_schema(properties: dict[str, Any]) -> dict[str, Any]:
    # Strict function calling requires every property to be listed in required.
    # Optional values use a nullable type and are removed before handler dispatch.
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


class WorkspaceTools:
    def __init__(
        self,
        root: str | Path,
        *,
        output_sink: OutputSink | None = None,
        permission_mode: str = "auto",
        approval_callback: ApprovalCallback | None = None,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self._output_sink = output_sink or (lambda _line: None)
        self.approval_callback = approval_callback
        self.permission_mode = permission_mode
        self._checkpoint_label = ""
        self._checkpoint_changes: dict[str, dict[str, Any]] | None = None

    @property
    def permission_mode(self) -> str:
        return self._permission_mode

    @permission_mode.setter
    def permission_mode(self, value: str) -> None:
        if value not in {"auto", "ask", "read-only"}:
            raise ValueError("permission mode must be auto, ask, or read-only")
        self._permission_mode = value

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
        ignored = {
            ".git",
            ".skillPilot",
            "__pycache__",
            ".pytest_cache",
            ".mypy_cache",
            ".ruff_cache",
            "node_modules",
            ".venv",
            "venv",
            ".env",
            ".npmrc",
            ".pypirc",
            "id_rsa",
            "id_ed25519",
        }
        return any(part in ignored for part in path.parts)

    @staticmethod
    def _sensitive(path: Path) -> bool:
        name = path.name.lower()
        return (
            name in {".env", ".npmrc", ".pypirc", "id_rsa", "id_ed25519"}
            or name.endswith((".pem", ".key", ".p12", ".pfx"))
            or "credentials" in name
        )

    @staticmethod
    def _limit(text: str, max_chars: int = 16000) -> str:
        if len(text) <= max_chars:
            return text
        return text[:max_chars] + f"\n... <truncated {len(text) - max_chars} chars>"

    def _authorize(self, action: str, detail: str) -> None:
        if self.permission_mode == "read-only":
            raise PermissionError(f"Blocked by read-only mode: {action}")
        if self.permission_mode == "ask":
            allowed = self.approval_callback and self.approval_callback(action, detail)
            if not allowed:
                raise PermissionError(f"User declined: {action}")

    def begin_checkpoint(self, label: str) -> None:
        self._checkpoint_label = label
        self._checkpoint_changes = {}

    def _capture(self, target: Path) -> None:
        if self._checkpoint_changes is None:
            return
        relative = str(target.relative_to(self.root)).replace("\\", "/")
        if relative in self._checkpoint_changes:
            return
        existed = target.exists() and target.is_file()
        content = target.read_bytes() if existed else b""
        self._checkpoint_changes[relative] = {
            "existed": existed,
            "content": base64.b64encode(content).decode("ascii"),
        }

    def finish_checkpoint(self) -> str | None:
        changes = self._checkpoint_changes
        self._checkpoint_changes = None
        if not changes:
            return None
        checkpoint_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-") + uuid.uuid4().hex[:8]
        folder = self.root / ".skillPilot" / "checkpoints"
        folder.mkdir(parents=True, exist_ok=True)
        manifest = {
            "id": checkpoint_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "label": self._checkpoint_label,
            "files": changes,
            "undone": False,
        }
        (folder / f"{checkpoint_id}.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (folder / "latest").write_text(checkpoint_id, encoding="utf-8")
        return checkpoint_id

    def undo_last(self) -> dict[str, Any]:
        folder = self.root / ".skillPilot" / "checkpoints"
        pointer = folder / "latest"
        if not pointer.exists():
            return {"ok": False, "error": "No file checkpoint is available"}
        checkpoint_id = pointer.read_text(encoding="utf-8").strip()
        manifest_path = folder / f"{checkpoint_id}.json"
        if not manifest_path.exists():
            return {"ok": False, "error": "Checkpoint manifest is missing"}
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        restored: list[str] = []
        removed: list[str] = []
        for relative, snapshot in manifest.get("files", {}).items():
            target = self._resolve(relative)
            if snapshot.get("existed"):
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(base64.b64decode(snapshot.get("content", "")))
                restored.append(relative)
            elif target.exists() and target.is_file():
                target.unlink()
                removed.append(relative)
        manifest["undone"] = True
        manifest["undone_at"] = datetime.now(timezone.utc).isoformat()
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

        previous = sorted(
            path.stem
            for path in folder.glob("*.json")
            if path.stem != checkpoint_id
            and not json.loads(path.read_text(encoding="utf-8")).get("undone")
        )
        if previous:
            pointer.write_text(previous[-1], encoding="utf-8")
        else:
            pointer.unlink(missing_ok=True)
        return {
            "ok": True,
            "checkpoint": checkpoint_id,
            "restored": restored,
            "removed": removed,
            "note": "Shell-command side effects are not included in file checkpoints.",
        }

    def list_files(
        self, path: str = ".", pattern: str = "**/*", max_results: int = 300
    ) -> dict[str, Any]:
        base = self._resolve(path)
        if not base.exists():
            raise FileNotFoundError(str(base))
        files: list[str] = []
        iterator = [base] if base.is_file() else base.glob(pattern)
        for candidate in iterator:
            if len(files) >= max_results:
                break
            if candidate.is_file() and not self._ignored(candidate):
                files.append(str(candidate.relative_to(self.root)).replace("\\", "/"))
        return {"ok": True, "count": len(files), "files": sorted(files)}

    def read_file(
        self,
        path: str,
        start_line: int | None = None,
        end_line: int | None = None,
    ) -> dict[str, Any]:
        target = self._resolve(path)
        if self._sensitive(target):
            raise PermissionError(f"Refusing to read likely credential file: {path}")
        if self._ignored(target):
            raise PermissionError(f"Refusing to read internal/ignored path: {path}")
        lines = target.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)
        start = max(1, int(start_line or 1))
        end = int(end_line) if end_line is not None else len(lines)
        return {
            "ok": True,
            "path": str(target.relative_to(self.root)).replace("\\", "/"),
            "start_line": start,
            "end_line": min(end, len(lines)),
            "line_count": len(lines),
            "content": self._limit("".join(lines[start - 1 : end])),
        }

    def write_file(self, path: str, content: str) -> dict[str, Any]:
        target = self._resolve(path)
        if self._ignored(target):
            raise ValueError(f"Refusing to write internal/ignored path: {path}")
        self._authorize("write_file", str(path))
        self._capture(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return {
            "ok": True,
            "path": str(target.relative_to(self.root)).replace("\\", "/"),
            "bytes": len(content.encode("utf-8")),
        }

    def edit_file(
        self,
        path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> dict[str, Any]:
        target = self._resolve(path)
        if self._ignored(target):
            raise ValueError(f"Refusing to edit internal/ignored path: {path}")
        if not target.exists():
            raise FileNotFoundError(f"File not found: {path}")
        content = target.read_text(encoding="utf-8")
        count = content.count(old_string)
        if count == 0:
            return {"ok": False, "error": "old_string not found", "path": path}
        if not replace_all and count > 1:
            return {
                "ok": False,
                "error": f"old_string matches {count} times; add context or set replace_all=true",
                "path": path,
                "match_count": count,
            }
        self._authorize("edit_file", str(path))
        self._capture(target)
        updated = content.replace(old_string, new_string) if replace_all else content.replace(old_string, new_string, 1)
        target.write_text(updated, encoding="utf-8")
        return {
            "ok": True,
            "path": str(target.relative_to(self.root)).replace("\\", "/"),
            "replacements": count if replace_all else 1,
            "bytes": len(updated.encode("utf-8")),
        }

    def delete_file(self, path: str) -> dict[str, Any]:
        target = self._resolve(path)
        if self._ignored(target) or not target.is_file():
            raise ValueError(f"Refusing to delete missing/internal path: {path}")
        self._authorize("delete_file", str(path))
        self._capture(target)
        target.unlink()
        return {"ok": True, "path": str(path)}

    def grep(
        self,
        pattern: str,
        path: str = ".",
        file_glob: str = "**/*",
        case_sensitive: bool = False,
        max_matches: int = 100,
    ) -> dict[str, Any]:
        regex = re.compile(pattern, 0 if case_sensitive else re.IGNORECASE)
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
            for line_no, line in enumerate(lines, 1):
                if regex.search(line):
                    matches.append(
                        {
                            "path": str(candidate.relative_to(self.root)).replace("\\", "/"),
                            "line": line_no,
                            "text": line[:500],
                        }
                    )
                    if len(matches) >= max_matches:
                        break
        return {"ok": True, "pattern": pattern, "count": len(matches), "matches": matches}

    def bash(self, command: str, timeout: int = 120) -> dict[str, Any]:
        self._authorize("bash", command)
        return self._run_with_stream(command, shell=True, timeout=min(max(int(timeout), 1), 600))

    def run_python(self, path: str, args: str = "", timeout: int = 60) -> dict[str, Any]:
        target = self._resolve(path)
        self._authorize("run_python", f"{path} {args}".strip())
        command = [sys.executable, str(target)] + shlex.split(args)
        return self._run_with_stream(command, timeout=min(max(int(timeout), 1), 600))

    def run_pytest(self, args: str = "", timeout: int = 180) -> dict[str, Any]:
        self._authorize("run_pytest", args or "pytest")
        command = [sys.executable, "-m", "pytest"] + shlex.split(args)
        return self._run_with_stream(command, timeout=min(max(int(timeout), 1), 600))

    def git_status(self) -> dict[str, Any]:
        return self._run_with_stream(["git", "status", "--short"], timeout=30)

    def git_diff(self, staged: bool = False, path: str | None = None) -> dict[str, Any]:
        command = ["git", "diff"]
        if staged:
            command.append("--staged")
        if path:
            target = self._resolve(path)
            command.extend(["--", str(target.relative_to(self.root))])
        return self._run_with_stream(command, timeout=30)

    def _run_with_stream(
        self,
        command: list[str] | str,
        *,
        shell: bool = False,
        timeout: int,
    ) -> dict[str, Any]:
        environment = os.environ.copy()
        environment.setdefault("PYTHONIOENCODING", "utf-8")
        kwargs: dict[str, Any] = {
            "args": command,
            "cwd": self.root,
            "stdout": subprocess.PIPE,
            "stderr": subprocess.STDOUT,
            "text": True,
            "encoding": "utf-8",
            "errors": "replace",
            "shell": shell,
            "env": environment,
        }
        collected: list[str] = []

        try:
            process = subprocess.Popen(**kwargs)
        except OSError as exc:
            return {"ok": False, "exit_code": None, "output": "", "error": f"OSError: {exc}"}

        def drain() -> None:
            assert process.stdout is not None
            for line in iter(process.stdout.readline, ""):
                collected.append(line)
                self._output_sink(line.rstrip("\n"))

        thread = threading.Thread(target=drain, daemon=True)
        thread.start()
        try:
            process.wait(timeout=timeout)
            thread.join(timeout=2)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
            thread.join(timeout=2)
            return {
                "ok": False,
                "exit_code": None,
                "output": self._limit("".join(collected)),
                "error": f"Timed out after {timeout}s",
            }
        return {
            "ok": process.returncode == 0,
            "exit_code": process.returncode,
            "output": self._limit("".join(collected)),
        }


class ToolRegistry:
    def __init__(self, tools: list[ToolSpec], workspace_tools: WorkspaceTools) -> None:
        self._tools = {tool.name: tool for tool in tools}
        self.workspace_tools = workspace_tools

    def list(self) -> list[ToolSpec]:
        return list(self._tools.values())

    def run(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        tool = self._tools.get(name)
        if tool is None:
            raise KeyError(f"Unknown tool: {name}")
        clean_args = {key: value for key, value in args.items() if value is not None}
        return tool.handler(**clean_args)

    def api_schemas(self) -> list[dict[str, Any]]:
        return [tool.api_schema() for tool in self.list()]

    def begin_checkpoint(self, label: str) -> None:
        self.workspace_tools.begin_checkpoint(label)

    def finish_checkpoint(self) -> str | None:
        return self.workspace_tools.finish_checkpoint()

    def undo_last(self) -> dict[str, Any]:
        return self.workspace_tools.undo_last()

    @property
    def permission_mode(self) -> str:
        return self.workspace_tools.permission_mode

    @permission_mode.setter
    def permission_mode(self, value: str) -> None:
        self.workspace_tools.permission_mode = value

    def prompt_summary(self) -> str:
        return json.dumps(self.api_schemas(), ensure_ascii=False, indent=2)


def build_default_registry(
    workspace: str | Path,
    *,
    output_sink: OutputSink | None = None,
    permission_mode: str = "auto",
    approval_callback: ApprovalCallback | None = None,
) -> ToolRegistry:
    workspace_tools = WorkspaceTools(
        workspace,
        output_sink=output_sink,
        permission_mode=permission_mode,
        approval_callback=approval_callback,
    )
    optional_string = {"type": ["string", "null"]}
    optional_integer = {"type": ["integer", "null"]}
    optional_boolean = {"type": ["boolean", "null"]}
    specs = [
        ToolSpec(
            "list_files",
            "List files under the workspace. Use this before guessing repository structure.",
            _object_schema({"path": optional_string, "pattern": optional_string, "max_results": optional_integer}),
            workspace_tools.list_files,
        ),
        ToolSpec(
            "read_file",
            "Read a UTF-8 text file, optionally by an inclusive 1-based line range.",
            _object_schema({"path": {"type": "string"}, "start_line": optional_integer, "end_line": optional_integer}),
            workspace_tools.read_file,
        ),
        ToolSpec(
            "grep",
            "Search workspace file contents with a regular expression.",
            _object_schema({
                "pattern": {"type": "string"},
                "path": optional_string,
                "file_glob": optional_string,
                "case_sensitive": optional_boolean,
                "max_matches": optional_integer,
            }),
            workspace_tools.grep,
        ),
        ToolSpec(
            "write_file",
            "Create or overwrite a UTF-8 text file. Prefer edit_file for a focused change.",
            _object_schema({"path": {"type": "string"}, "content": {"type": "string"}}),
            workspace_tools.write_file,
            True,
        ),
        ToolSpec(
            "edit_file",
            "Replace an exact string in a file. Include enough surrounding context to make it unique.",
            _object_schema({
                "path": {"type": "string"},
                "old_string": {"type": "string"},
                "new_string": {"type": "string"},
                "replace_all": optional_boolean,
            }),
            workspace_tools.edit_file,
            True,
        ),
        ToolSpec(
            "delete_file",
            "Delete one workspace file. Use only when removal is required by the task.",
            _object_schema({"path": {"type": "string"}}),
            workspace_tools.delete_file,
            True,
        ),
        ToolSpec(
            "bash",
            "Run a shell command in the workspace. Use for project-native build, test, lint, package, and git commands.",
            _object_schema({"command": {"type": "string"}, "timeout": optional_integer}),
            workspace_tools.bash,
            True,
        ),
        ToolSpec(
            "run_python",
            "Run a Python file with the active interpreter.",
            _object_schema({"path": {"type": "string"}, "args": optional_string, "timeout": optional_integer}),
            workspace_tools.run_python,
            True,
        ),
        ToolSpec(
            "run_pytest",
            "Run pytest in the workspace.",
            _object_schema({"args": optional_string, "timeout": optional_integer}),
            workspace_tools.run_pytest,
            True,
        ),
        ToolSpec("git_status", "Show concise Git working-tree status.", _object_schema({}), workspace_tools.git_status),
        ToolSpec(
            "git_diff",
            "Show the current Git diff, optionally staged or limited to one path.",
            _object_schema({"staged": optional_boolean, "path": optional_string}),
            workspace_tools.git_diff,
        ),
    ]
    return ToolRegistry(specs, workspace_tools)
