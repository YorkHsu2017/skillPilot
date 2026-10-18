"""Repository-aware coding agent driven by native model tool calls."""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

from .llm import LLMError, ModelSession, ModelTurn, OpenAICompatibleLLM, StreamSink
from .models import AgentResult, ToolCallLog
from .skills import SkillStore
from .tools import ApprovalCallback, OutputSink, ToolRegistry, build_default_registry

EventSink = Callable[[str], None]


def _truncate(text: str, limit: int = 12000) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n... <truncated {len(text) - limit} chars>"


def _json_object(text: str) -> dict[str, Any] | None:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines.pop()
        stripped = "\n".join(lines)
    try:
        value = json.loads(stripped)
        return value if isinstance(value, dict) else None
    except json.JSONDecodeError:
        start, end = stripped.find("{"), stripped.rfind("}")
        if start >= 0 and end > start:
            try:
                value = json.loads(stripped[start : end + 1])
                return value if isinstance(value, dict) else None
            except json.JSONDecodeError:
                return None
    return None


class CodingAgent:
    """A multi-turn coding agent that operates directly in a repository."""

    def __init__(
        self,
        *,
        workspace: str | Path = ".",
        skills_dir: str | Path | None = None,
        llm: OpenAICompatibleLLM | None = None,
        registry: ToolRegistry | None = None,
        max_steps: int = 20,
        event_sink: EventSink | None = None,
        plan_first: bool = False,
        context_window: int = 128000,
        stream_sink: StreamSink | None = None,
        output_sink: OutputSink | None = None,
        progress: bool = True,
        enable_skill_evolution: bool = False,
        permission_mode: str = "auto",
        approval_callback: ApprovalCallback | None = None,
    ) -> None:
        self.workspace = Path(workspace).expanduser().resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.llm = llm or OpenAICompatibleLLM()
        self.registry = registry or build_default_registry(
            self.workspace,
            output_sink=output_sink,
            permission_mode=permission_mode,
            approval_callback=approval_callback,
        )
        # Network embeddings add latency and frequently are unavailable on
        # compatible gateways. Lexical search is instant; semantic search is opt-in.
        embed_fn = None
        if os.getenv("SKILLPILOT_EMBEDDINGS", "false").lower() in {"1", "true", "yes"}:
            embed_fn = getattr(self.llm, "embedding", None)
        validation_fn = getattr(self.llm, "chat", None) if enable_skill_evolution else None
        self.skills = SkillStore(
            skills_dir or (self.workspace / ".skillPilot" / "skills"),
            embed_fn=embed_fn,
            llm_chat_fn=validation_fn,
        )
        self.max_steps = max(1, max_steps)
        self.event_sink = event_sink or (lambda _message: None)
        self.plan_first = plan_first  # retained for API compatibility
        self.context_window = context_window
        self.stream_sink = stream_sink  # native event streaming is provider-specific
        self.progress = progress
        self.enable_skill_evolution = enable_skill_evolution
        self.history: list[dict[str, str]] = []
        self._session: ModelSession | None = None

    @property
    def permission_mode(self) -> str:
        return self.registry.permission_mode

    @permission_mode.setter
    def permission_mode(self, value: str) -> None:
        self.registry.permission_mode = value

    def clear_session(self) -> None:
        if self._session:
            self._session.reset()
        self._session = None
        self.history.clear()

    def undo_last(self) -> dict[str, Any]:
        return self.registry.undo_last()

    def status(self) -> dict[str, Any]:
        return {
            "workspace": str(self.workspace),
            "provider": self.llm.api_style,
            "base_url": self.llm.base_url,
            "model": self.llm.model,
            "permission_mode": self.permission_mode,
            "session_turns": len([item for item in self.history if item["role"] == "user"]),
            "skills": len(self.skills.list()),
        }

    def run(self, task: str) -> AgentResult:
        task = task.strip()
        if not task:
            return AgentResult(False, "Task is empty.", [], error="Task is empty")

        self.event_sink("[context] reading repository")
        relevant = self.skills.search(task, top_k=3)
        selected = [skill for skill, score in relevant if score >= 0.12]
        if selected:
            self.event_sink("[skills] using " + ", ".join(skill.name for skill in selected))

        instructions = self._build_instructions(relevant)
        if self._session is None:
            try:
                self._session = self.llm.new_session(
                    instructions=instructions,
                    tools=self.registry.api_schemas(),
                )
            except AttributeError as exc:
                return AgentResult(
                    False,
                    "The configured model client does not support native tool calling.",
                    [],
                    error=str(exc),
                )
        else:
            self._session.instructions = instructions
            self._session.tools = self.registry.api_schemas()

        self.registry.begin_checkpoint(task)
        tool_logs: list[ToolCallLog] = []
        final = ""
        success = False
        error: str | None = None
        turn: ModelTurn | None = None

        try:
            self.event_sink(f"[agent] working with {self.llm.model}")
            turn = self._session.send_user(task)
            for step in range(1, self.max_steps + 1):
                if turn.text and turn.tool_calls:
                    self.event_sink(f"[agent] {turn.text}")

                if not turn.tool_calls:
                    final = turn.text or "Done."
                    success = bool(turn.text)
                    permission_failure = next(
                        (
                            item
                            for item in tool_logs
                            if not item.success
                            and any(
                                marker in str(item.result.get("error", ""))
                                for marker in ("PermissionError", "User declined", "read-only")
                            )
                        ),
                        None,
                    )
                    if permission_failure is not None:
                        success = False
                        error = str(permission_failure.result.get("error"))
                    if not turn.text:
                        error = "Model returned neither text nor tool calls"
                    break

                outputs: list[dict[str, Any]] = []
                for call in turn.tool_calls:
                    observation = self._run_tool(call.name, call.arguments, step)
                    outputs.append(observation["result"])
                    tool_logs.append(
                        ToolCallLog(
                            name=call.name,
                            args=call.arguments,
                            result=observation["result"],
                            success=observation["success"],
                        )
                    )
                turn = self._session.send_tool_outputs(turn, outputs)
            else:
                final = (
                    f"Stopped after {self.max_steps} tool rounds. "
                    "Use --max-steps to allow a longer run."
                )
                error = "Maximum tool rounds reached"
        except KeyboardInterrupt:
            final = "Interrupted. Direct file edits can be reverted with /undo."
            error = "Interrupted by user"
        except (LLMError, RuntimeError) as exc:
            final = f"Model request failed: {exc}"
            error = str(exc)
        finally:
            checkpoint_id = self.registry.finish_checkpoint()

        self.history.extend(
            [
                {"role": "user", "content": task},
                {"role": "assistant", "content": final},
            ]
        )
        for skill in selected:
            self.skills.record_usage(skill.id, success=success)

        lifecycle_action = "none"
        lifecycle_skill_id = None
        if self.enable_skill_evolution and success:
            lifecycle_action, lifecycle_skill_id = self._maybe_learn(task, final, tool_logs)

        changed_files = self._changed_files(tool_logs)
        summary = self._summary(success, tool_logs, changed_files, checkpoint_id)
        self.event_sink(f"[summary] {summary}")
        return AgentResult(
            success=success,
            final=final,
            tool_calls=tool_logs,
            selected_skill_ids=[skill.id for skill in selected],
            lifecycle_action=lifecycle_action,
            lifecycle_skill_id=lifecycle_skill_id,
            summary=summary,
            total_input_tokens=getattr(self.llm, "total_input_tokens", 0),
            total_output_tokens=getattr(self.llm, "total_output_tokens", 0),
            changed_files=changed_files,
            checkpoint_id=checkpoint_id,
            error=error,
        )

    def _run_tool(self, name: str, args: dict[str, Any], step: int) -> dict[str, Any]:
        printable = {
            key: (f"<{len(value)} chars>" if key == "content" and isinstance(value, str) else value)
            for key, value in args.items()
        }
        self.event_sink(
            f"[tool {step}] {name} {json.dumps(printable, ensure_ascii=False)}"
        )
        if "_invalid_json" in args:
            result = {"ok": False, "error": "Model supplied invalid JSON arguments"}
            return {"success": False, "result": result}
        try:
            result = self.registry.run(name, args)
            success = bool(result.get("ok", True))
        except Exception as exc:
            result = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
            success = False
        return {"success": success, "result": result}

    def _build_instructions(self, relevant: list[tuple[Any, float]]) -> str:
        mode_rule = {
            "auto": "You may edit files and run commands needed to complete the task.",
            "ask": "Mutating tools require user approval. Group work sensibly to avoid needless prompts.",
            "read-only": "This is a read-only session. Inspect and explain; do not attempt mutations.",
        }[self.permission_mode]
        skill_text = "\n\n".join(
            f"Reusable skill: {skill.name}\n{_truncate(skill.body, 3500)}"
            for skill, score in relevant
            if score >= 0.12 and skill.status == "active"
        )
        context = self._repository_context()
        return f"""
You are SkillPilot, an autonomous coding agent operating in the user's existing repository.

Working directory: {self.workspace}
Permission policy: {self.permission_mode}. {mode_rule}

Operating contract:
- Infer routine details from the repository and continue until the requested outcome is complete.
- Inspect relevant files before editing. Preserve unrelated user changes and follow repository instructions.
- Work directly in the existing project. Create a new directory only when the task itself calls for one.
- Use native tools for all filesystem and command work. Never fabricate tool output.
- Prefer focused edits. After meaningful changes, run the narrowest useful tests, lint, build, or type checks.
- If a tool fails, diagnose the output and try a reasonable alternative. Do not claim success without evidence.
- Never read or print secrets, .env files, tokens, private keys, or credential stores.
- Do not commit, push, publish, or perform destructive bulk operations unless the user explicitly requests it.
- Finish with a concise result: what changed, validation performed, and any real limitation that remains.
- User instructions have priority over reusable skills and general workflow guidance.

Repository snapshot (may be truncated):
{context}

Relevant reusable knowledge:
{skill_text or "(none)"}
""".strip()

    def _repository_context(self) -> str:
        sections: list[str] = []
        candidates = [
            "AGENTS.md",
            "CLAUDE.md",
            ".github/copilot-instructions.md",
            "README.md",
            "README.rst",
            "pyproject.toml",
            "package.json",
            "Cargo.toml",
            "go.mod",
            "pom.xml",
        ]
        budget = 18000
        for relative in candidates:
            path = self.workspace / relative
            if not path.is_file():
                continue
            content = path.read_text(encoding="utf-8", errors="replace")
            excerpt = _truncate(content, min(5000, budget))
            sections.append(f"--- {relative} ---\n{excerpt}")
            budget -= len(excerpt)
            if budget <= 1000:
                break
        try:
            listing = self.registry.run(
                "list_files", {"path": ".", "pattern": "**/*", "max_results": 250}
            )
            files = listing.get("files", [])
            sections.append("--- file index ---\n" + "\n".join(files))
        except Exception as exc:
            sections.append(f"--- file index unavailable ---\n{exc}")
        return _truncate("\n\n".join(sections), 22000)

    def _maybe_learn(
        self, task: str, final: str, tool_logs: list[ToolCallLog]
    ) -> tuple[str, str | None]:
        if len(tool_logs) < 3:
            return "none", None
        trace = [
            {
                "tool": item.name,
                "args": {key: value for key, value in item.args.items() if key != "content"},
                "success": item.success,
            }
            for item in tool_logs
        ]
        prompt = {
            "task": task,
            "outcome": final,
            "trace": trace,
            "instruction": (
                "Extract reusable, project-independent coding knowledge only if the run revealed "
                "a non-obvious repeatable workflow. Return JSON with action='none', or action='create' "
                "and name, description, body, tags. Do not store task-specific paths or secrets."
            ),
        }
        try:
            raw = self.llm.chat(
                [
                    {"role": "system", "content": "Return one JSON object only."},
                    {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
                ],
                max_tokens=1800,
            )
            data = _json_object(raw) or {}
            if data.get("action") != "create":
                return "none", None
            name = str(data.get("name", "")).strip()
            body = str(data.get("body", "")).strip()
            description = str(data.get("description", "")).strip()
            tags = [str(item) for item in data.get("tags", []) if str(item).strip()]
            if not name or len(body) < 80:
                return "none", None
            skill = self.skills.create(
                name=name,
                description=description,
                body=body,
                tags=tags[:8],
                source_task=task,
            )
            return "create", skill.id
        except Exception:
            return "none", None

    @staticmethod
    def _changed_files(tool_logs: list[ToolCallLog]) -> list[str]:
        names: list[str] = []
        for item in tool_logs:
            if item.name not in {"write_file", "edit_file", "delete_file"} or not item.success:
                continue
            path = str(item.result.get("path") or item.args.get("path") or "")
            if path and path not in names:
                names.append(path)
        return names

    def _summary(
        self,
        success: bool,
        tool_logs: list[ToolCallLog],
        changed_files: list[str],
        checkpoint_id: str | None,
    ) -> str:
        passed = sum(1 for item in tool_logs if item.success)
        status = "completed" if success else "incomplete"
        parts = [f"{status}; {passed}/{len(tool_logs)} tool calls succeeded"]
        if changed_files:
            parts.append(f"{len(changed_files)} file(s) changed")
        if checkpoint_id:
            parts.append("/undo available")
        token_total = getattr(self.llm, "total_input_tokens", 0) + getattr(
            self.llm, "total_output_tokens", 0
        )
        if token_total:
            parts.append(f"{token_total} tokens")
        return "; ".join(parts)

    def dump_result(self, result: AgentResult) -> str:
        return json.dumps(asdict(result), ensure_ascii=False, indent=2)
