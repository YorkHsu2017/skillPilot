from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

from .llm import ChatLLM, OpenAICompatibleLLM, StreamSink
from .models import AgentResult, ToolCallLog
from .progress import clear_terminal_line, format_thought, render_progress_line, render_tool_line
from .skills import SkillStore
from .tools import OutputSink, ToolRegistry, build_default_registry


EventSink = Callable[[str], None]


def _truncate(text: str, limit: int = 12000) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n... <truncated {len(text) - limit} chars>"


def _json_block(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    try:
        parsed = json.loads(stripped)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    start = stripped.find("{")
    if start == -1:
        raise ValueError("No JSON object found")
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(stripped)):
        char = stripped[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                parsed = json.loads(stripped[start : index + 1])
                if not isinstance(parsed, dict):
                    raise ValueError("JSON response must be an object")
                return parsed
    raise ValueError("No complete JSON object found")


class CodingAgent:
    def __init__(
        self,
        *,
        workspace: str | Path = ".",
        skills_dir: str | Path | None = None,
        llm: ChatLLM | None = None,
        registry: ToolRegistry | None = None,
        max_steps: int = 12,
        event_sink: EventSink | None = None,
        plan_first: bool = False,
        context_window: int = 128000,
        stream_sink: StreamSink | None = None,
        output_sink: OutputSink | None = None,
        progress: bool = True,
        enable_skill_evolution: bool = True,
    ) -> None:
        self.workspace = Path(workspace).expanduser().resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.llm = llm or OpenAICompatibleLLM()
        self._output_sink = output_sink or (lambda _line: None)
        self.registry = registry or build_default_registry(self.workspace, output_sink=self._output_sink)
        # Create embed_fn if LLM supports embedding
        embed_fn = None
        if hasattr(self.llm, 'embedding') and callable(getattr(self.llm, 'embedding')):
            embed_fn = self.llm.embedding
        # Create llm_chat_fn for quality validation
        llm_chat_fn = None
        if hasattr(self.llm, 'chat') and callable(getattr(self.llm, 'chat')):
            llm_chat_fn = self.llm.chat
        self.skills = SkillStore(
            skills_dir or (self.workspace / ".skillPilot" / "skills"),
            embed_fn=embed_fn,
            llm_chat_fn=llm_chat_fn,
        )
        self.max_steps = max_steps
        self.event_sink = event_sink or (lambda _message: None)
        self.plan_first = plan_first
        self.context_window = context_window
        self.stream_sink = stream_sink
        self._progress = progress
        self._step_counter = 0
        self.history: list[dict[str, str]] = []
        self.plan: list[str] = []
        self.enable_skill_evolution = enable_skill_evolution
        self._last_search_results: list[tuple] = []

    def run(self, task: str) -> AgentResult:
        self.event_sink("[skills] searching relevant skills")
        relevant = self.skills.search(task, top_k=5)
        self._last_search_results = relevant  # Store for skill lifecycle decision
        if relevant:
            skill_names = [f"{s.name} ({score:.2f})" for s, score in relevant[:3]]
            self.event_sink(f"[skills] found {len(relevant)} candidate(s): {', '.join(skill_names)}")
        else:
            self.event_sink("[skills] no existing skills found")

        skill_decision = self._decide_skill_usage(task, relevant)
        selected_skill_ids = [
            str(skill_id)
            for skill_id in skill_decision.get("use_skill_ids", [])
            if self.skills.get(str(skill_id)) is not None
        ]

        # Log skill reuse decision explicitly
        if selected_skill_ids:
            names = []
            for sid in selected_skill_ids:
                s = self.skills.get(sid)
                names.append(s.name if s else sid)
            rationale = skill_decision.get("rationale", "")
            self.event_sink(
                f"[skills] ✓ REUSE: {', '.join(names)}"
                + (f" — {rationale}" if rationale else "")
            )
        else:
            self.event_sink("[skills] — no skill to reuse, proceeding from scratch")

        plan = self._create_plan(task) if self.plan_first else []
        self.plan = plan
        if plan:
            self.event_sink(f"[plan] {len(plan)} steps: {' | '.join(plan)}")

        observations: list[dict[str, Any]] = []
        tool_calls: list[ToolCallLog] = []
        final = ""
        success = False
        self.history.append({"role": "user", "content": task})

        for _step in range(self.max_steps):
            self._step_counter = _step
            if self._progress:
                sys.stderr.write(render_progress_line(_step + 1, self.max_steps, tick=_step, label="agent"))
                sys.stderr.flush()
            else:
                self.event_sink(f"[progress] step {_step + 1}/{self.max_steps}")
            observations = self._maybe_compress_observations(observations)
            action = self._next_action(task, relevant, skill_decision, observations)
            thought = action.get("thought", "")
            if thought:
                if self._progress:
                    sys.stderr.write(clear_terminal_line())
                    sys.stderr.write(format_thought(thought) + "\n")
                    sys.stderr.write(render_progress_line(_step + 1, self.max_steps, tick=_step, label="agent"))
                    sys.stderr.flush()
                else:
                    self.event_sink(format_thought(thought))
            if action.get("action") == "final":
                validation_error = self._pending_python_validation(tool_calls)
                if validation_error:
                    observations.append(
                        {
                            "tool": "agent_guard",
                            "args": {},
                            "success": False,
                            "result": {"ok": False, "error": validation_error},
                        }
                    )
                    continue
                final = str(action.get("final", "")).strip() or "Done."
                success = True
                break

            if action.get("action") != "tool":
                final = str(action.get("final") or action.get("message") or action)
                success = True
                break

            tool_name = str(action.get("tool", ""))
            args = action.get("args", {})
            if not isinstance(args, dict):
                args = {}
            observation = self._run_tool(tool_name, args)
            observations.append(observation)
            tool_calls.append(
                ToolCallLog(
                    name=tool_name,
                    args=args,
                    result=observation["result"],
                    success=bool(observation["success"]),
                )
            )
        else:
            final = "Stopped before completion because the maximum tool step count was reached."

        self.history.append({"role": "assistant", "content": final})
        for skill_id in selected_skill_ids:
            self.skills.record_usage(skill_id, success=success)

        # Run deprecation audit before lifecycle decisions
        deprecated = self.skills.check_and_deprecate()
        for dep in deprecated:
            self.event_sink(
                f"[skills] deprecated {dep['skill_id']} ({dep['name']}): {dep['reason']}"
            )

        lifecycle_action, lifecycle_skill_id = self._run_skill_lifecycle(
            task=task,
            final=final,
            success=success,
            tool_calls=tool_calls,
            relevant_skill_ids=[skill.id for skill, _score in relevant],
        )
        
        # Log lifecycle decision explicitly
        if lifecycle_action == "create":
            skill = self.skills.get(lifecycle_skill_id) if lifecycle_skill_id else None
            name = skill.name if skill else "unknown"
            self.event_sink(f"[skills] → CREATE: {name}")
        elif lifecycle_action == "update":
            skill = self.skills.get(lifecycle_skill_id) if lifecycle_skill_id else None
            name = skill.name if skill else lifecycle_skill_id
            self.event_sink(f"[skills] → UPDATE: {name}")
        else:
            self.event_sink("[skills] → no skill lifecycle action")
        
        summary = self._summarize_task(
            task=task, success=success, final=final, tool_calls=tool_calls,
        )
        if hasattr(self.llm, "total_input_tokens"):
            summary += (
                f"\nTokens: {self.llm.total_input_tokens} in / {self.llm.total_output_tokens} out"
            )
        if self._progress:
            sys.stderr.write(clear_terminal_line())
            sys.stderr.flush()
        self.event_sink(f"[summary]\n{summary}")
        return AgentResult(
            success=success,
            final=final,
            tool_calls=tool_calls,
            selected_skill_ids=selected_skill_ids,
            lifecycle_action=lifecycle_action,
            lifecycle_skill_id=lifecycle_skill_id,
            plan=plan if plan else None,
            summary=summary,
            total_input_tokens=self.llm.total_input_tokens if hasattr(self.llm, "total_input_tokens") else 0,
            total_output_tokens=self.llm.total_output_tokens if hasattr(self.llm, "total_output_tokens") else 0,
        )

    def _decide_skill_usage(
        self,
        task: str,
        relevant: list[tuple[Any, float]],
    ) -> dict[str, Any]:
        prompt = {
            "task": task,
            "relevant_skills": self.skills.to_prompt_items(relevant),
            "instruction": (
                "Decide whether any retrieved skill should be referenced for this new task. "
                "Return JSON only: {\"use_skill_ids\": [\"...\"], \"rationale\": \"...\"}. "
                "Use an empty list if no skill is useful."
            ),
        }
        try:
            raw = self.llm.chat(
                [
                    {
                        "role": "system",
                        "content": "You evaluate reusable skills for a CLI coding agent. Return strict JSON only.",
                    },
                    {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
                ],
                temperature=0,
                max_tokens=1000,
            )
        except RuntimeError:
            return {"use_skill_ids": [], "rationale": "LLM call failed"}
        try:
            parsed = _json_block(raw)
        except ValueError:
            parsed = {"use_skill_ids": [], "rationale": raw[:500]}
        if not isinstance(parsed.get("use_skill_ids", []), list):
            parsed["use_skill_ids"] = []
        return parsed

    def _create_plan(self, task: str) -> list[str]:
        """Ask the LLM to decompose the task into ordered steps (plan-first mode)."""
        prompt = {
            "task": task,
            "instruction": (
                "Decompose this coding task into a concrete, ordered list of actionable steps. "
                "Each step should be one sentence describing a single tool-level action or check. "
                "Keep steps practical and scoped to what a terminal coding agent can do. "
                "Return JSON only: {\"steps\": [\"step 1\", \"step 2\", ...]}"
            ),
        }
        try:
            raw = self.llm.chat(
                [
                    {
                        "role": "system",
                        "content": "You decompose coding tasks into ordered actionable steps. Return strict JSON only.",
                    },
                    {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
                ],
                temperature=0,
                max_tokens=2000,
            )
        except RuntimeError:
            return []
        try:
            parsed = _json_block(raw)
        except ValueError:
            return []
        steps = parsed.get("steps", [])
        if isinstance(steps, list):
            return [str(s).strip() for s in steps if s and str(s).strip()]
        return []

    def _next_action(
        self,
        task: str,
        relevant: list[tuple[Any, float]],
        skill_decision: dict[str, Any],
        observations: list[dict[str, Any]],
    ) -> dict[str, Any]:
        prompt = f"""
You are a terminal coding agent, similar in behavior to Copilot CLI.

Current workspace: {self.workspace}
Current task: {task}

Plan (ordered steps to follow):
{json.dumps(self.plan if self.plan else ["(no plan — work step by step)"], ensure_ascii=False)}

Rules:
- Work by calling tools. If the task requires code, create or edit executable files with write_file or edit_file.
- Use edit_file for small targeted changes instead of rewriting entire files — it's faster and safer.
- Inspect files with list_files/read_file/grep when useful.
- Use bash to run shell commands (pip install, git, npm, etc.) inside the workspace when needed.
- After writing Python files, validate with run_python or run_pytest before final.
- Stay inside the workspace; never ask the user to do work you can do with tools.
- For every task, create a NEW subfolder under ./workspace (e.g. ./workspace/my_task/) and place ALL generated files inside it. Never write code files directly in the workspace root or the project root.
- Return exactly one JSON object and no markdown.

Response schema for a tool call:
{{"action": "tool", "tool": "write_file", "args": {{"path": "workspace/<task_subfolder>/file.py", "content": "..."}}, "thought": "brief reason"}}

Response schema when done:
{{"action": "final", "final": "Concise completion message with files/commands/results."}}

Available tools:
{self.registry.prompt_summary()}

Relevant skills retrieved before this task:
{json.dumps(self.skills.to_prompt_items(relevant), ensure_ascii=False, indent=2)}

LLM decision about skill reuse:
{json.dumps(skill_decision, ensure_ascii=False, indent=2)}

Recent conversation:
{json.dumps(self.history[-8:], ensure_ascii=False, indent=2)}

Tool observations so far:
{_truncate(json.dumps(observations, ensure_ascii=False, indent=2), 16000)}
""".strip()
        try:
            if self.stream_sink and hasattr(self.llm, "stream_chat"):
                raw = self.llm.stream_chat(
                    [
                        {
                            "role": "system",
                            "content": (
                                "You are an autonomous CLI coding agent. You must answer with strict JSON only."
                            ),
                        },
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0,
                    max_tokens=6000,
                    stream_sink=self.stream_sink,
                )
            else:
                raw = self.llm.chat(
                    [
                        {
                            "role": "system",
                            "content": (
                                "You are an autonomous CLI coding agent. You must answer with strict JSON only."
                            ),
                        },
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0,
                    max_tokens=6000,
                )
        except RuntimeError:
            return {"action": "final", "final": "LLM call failed; cannot continue."}
        try:
            return _json_block(raw)
        except ValueError:
            return {"action": "final", "final": raw.strip()}

    def _run_tool(self, tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
        printable_args = {
            key: (f"<{len(value)} chars>" if key == "content" and isinstance(value, str) else value)
            for key, value in args.items()
        }
        if self._progress:
            sys.stderr.write(
                render_tool_line(
                    tool_name,
                    json.dumps(printable_args, ensure_ascii=False),
                    tick=self._step_counter,
                )
            )
            sys.stderr.flush()
        else:
            self.event_sink(f"[tool] {tool_name} {json.dumps(printable_args, ensure_ascii=False)}")
        try:
            result = self.registry.run(tool_name, args)
            success = bool(result.get("ok", True))
        except Exception as exc:  # Tool errors are observations for the next LLM step.
            result = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
            success = False
        return {"tool": tool_name, "args": printable_args, "success": success, "result": result}

    def _maybe_compress_observations(
        self, observations: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Compress older observations when estimated prompt tokens exceed 80% of context window."""
        if len(observations) <= 3:
            return observations
        prompt_est = self._estimate_prompt_tokens(observations)
        threshold = int(self.context_window * 0.8)
        if prompt_est <= threshold:
            return observations
        # Keep the 3 most recent observations intact; compress the rest into one summary.
        old = observations[:-3]
        recent = observations[-3:]
        summary = self._summarize_observations(old)
        self.event_sink(f"[compress] {len(old)} observations → {summary[:200]}")
        return [{"tool": "compressed", "args": {}, "success": True, "result": {"summary": summary}}] + recent

    def _estimate_prompt_tokens(self, observations: list[dict[str, Any]]) -> int:
        """Estimate the token count of the static prompt + observations."""
        est = OpenAICompatibleLLM
        tokens = 0
        # Static prompt overhead (rules, tools, etc.) — roughly 3000 tokens.
        tokens += 3000
        tokens += est.estimate_tokens(json.dumps(self.plan, ensure_ascii=False))
        tokens += est.estimate_tokens(json.dumps(self.history[-8:], ensure_ascii=False))
        tokens += est.estimate_tokens(json.dumps(observations, ensure_ascii=False, default=str))
        return tokens

    def _summarize_observations(self, observations: list[dict[str, Any]]) -> str:
        """Ask the LLM to summarize a batch of older observations into a single paragraph."""
        payload = json.dumps(observations, ensure_ascii=False, default=str)
        prompt = {
            "instruction": (
                "Summarize these tool observations into a single concise paragraph (max 150 words). "
                "Focus on what was done, what succeeded, and what failed. "
                "Return JSON only: {\"summary\": \"...\"}"
            ),
            "observations": payload[:8000],
        }
        try:
            raw = self.llm.chat(
                [
                    {"role": "system", "content": "You summarize tool observations. Return strict JSON only."},
                    {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
                ],
                temperature=0,
                max_tokens=400,
            )
        except RuntimeError:
            return f"Earlier tool calls: {_truncate(payload, 500)}"
        try:
            parsed = _json_block(raw)
        except ValueError:
            return f"Earlier tool calls: {_truncate(payload, 500)}"
        return str(parsed.get("summary", "")) or f"Earlier tool calls: {_truncate(payload, 500)}"

    @staticmethod
    def _pending_python_validation(tool_calls: list[ToolCallLog]) -> str | None:
        last_python_write = -1
        last_successful_validation = -1
        for index, call in enumerate(tool_calls):
            if call.name == "write_file" and call.success:
                path = str(call.args.get("path", "")).lower()
                if path.endswith(".py"):
                    last_python_write = index
            if call.name in {"run_python", "run_pytest"} and call.success:
                last_successful_validation = index

        if last_python_write >= 0 and last_successful_validation < last_python_write:
            return (
                "A Python file was written or changed, but there has not been a successful "
                "run_python or run_pytest after the latest Python write. Validate the generated "
                "code before returning final."
            )
        return None

    @staticmethod
    def _compute_task_metrics(tool_calls: list[ToolCallLog]) -> dict[str, Any]:
        """Compute quantitative metrics for skill evolution decision."""
        total_calls = len(tool_calls)
        
        # Count file writes and lines written
        files_written = 0
        lines_written = 0
        for call in tool_calls:
            if call.name == "write_file" and call.success:
                files_written += 1
                content = call.args.get("content", "")
                if isinstance(content, str):
                    lines_written += content.count("\n") + (1 if content and not content.endswith("\n") else 0)
        
        return {
            "total_tool_calls": total_calls,
            "files_written": files_written,
            "lines_written": lines_written,
        }
    
    @staticmethod
    def _classify_task(task: str) -> list[str]:
        """Classify task into categories based on keywords."""
        task_lower = task.lower()
        categories = []
        
        category_keywords = {
            "networking": ["http", "api", "request", "fetch", "client", "server", "endpoint"],
            "file_processing": ["read", "write", "file", "parse", "csv", "json", "xml", "convert"],
            "cli_tool": ["cli", "command", "argparse", "click", "terminal", "script"],
            "testing": ["test", "pytest", "unittest", "mock", "assert"],
            "data_processing": ["filter", "transform", "aggregate", "analyze", "dataframe"],
            "configuration": ["config", "settings", "env", "yaml", "toml", "ini"],
            "error_handling": ["retry", "backoff", "timeout", "exception", "error", "handle"],
            "cryptography": ["encrypt", "decrypt", "cipher", "hash", "crypto"],
            "web_scraping": ["scrape", "crawl", "html", "selenium", "beautifulsoup"],
            "database": ["database", "sql", "query", "orm", "sqlite", "postgres"],
        }
        
        for category, keywords in category_keywords.items():
            if any(kw in task_lower for kw in keywords):
                categories.append(category)
        
        return categories if categories else ["general"]
    
    def _find_similar_skill(self, relevant_skills: list[dict[str, Any]]) -> tuple[str | None, float]:
        """Find the most similar existing skill. Returns (skill_id, similarity_score) or (None, 0.0)."""
        if not relevant_skills:
            return None, 0.0
        
        # Find skill with highest similarity score
        best_skill_id = None
        best_score = 0.0
        
        for skill_dict in relevant_skills:
            # Similarity score is stored in the skill dict by search
            score = skill_dict.get("similarity_score", 0.0)
            if score > best_score:
                best_score = score
                best_skill_id = skill_dict.get("id")
        
        return best_skill_id, best_score
    
    def _run_skill_lifecycle(
        self,
        *,
        task: str,
        final: str,
        success: bool,
        tool_calls: list[ToolCallLog],
        relevant_skill_ids: list[str],
    ) -> tuple[str, str | None]:
        # Skip if skill evolution is disabled
        if not self.enable_skill_evolution:
            return "none", None
        
        # Compute metrics
        metrics = self._compute_task_metrics(tool_calls)
        categories = self._classify_task(task)
        
        # Build tool calls payload
        calls_payload = [
            {
                "name": call.name,
                "success": call.success,
                "args": self._summarize_value(call.args),
                "result": self._summarize_value(call.result),
            }
            for call in tool_calls
        ]
        
        # Build relevant skills list with similarity scores
        relevant_skills = []
        for skill_id in relevant_skill_ids:
            skill = self.skills.get(skill_id)
            if skill is not None:
                skill_dict = skill.to_dict()
                # Find this skill in search results to get similarity score
                for search_skill, score in self._last_search_results:
                    if search_skill.id == skill_id:
                        skill_dict["similarity_score"] = score
                        break
                relevant_skills.append(skill_dict)
        
        # Find most similar existing skill
        similar_skill_id, similar_score = self._find_similar_skill(relevant_skills)
        
        # Build enhanced prompt
        prompt = {
            "task": task,
            "success": success,
            "final": final,
            "metrics": metrics,
            "categories": categories,
            "tool_calls": calls_payload,
            "most_similar_existing_skill": {
                "id": similar_skill_id,
                "similarity": round(similar_score, 3),
            } if similar_skill_id else None,
            "relevant_existing_skills": relevant_skills,
            "instruction": (
                "Evaluate the skill lifecycle after this task. "
                "Consider the quantitative metrics and task categories. "
                "Create a new skill if:\n"
                "- The task was successful\n"
                "- The task involved significant complexity (multiple tool calls, files written)\n"
                "- The task represents a reusable pattern (not a one-off specific task)\n"
                "- No highly similar skill already exists (similarity < 0.75)\n\n"
                "Update an existing skill if:\n"
                "- A highly similar skill exists (similarity >= 0.75)\n"
                "- The current task adds meaningful improvements to that skill\n\n"
                "Return strict JSON with one of these shapes: "
                "{\"action\":\"none\",\"reason\":\"...\"}; "
                "{\"action\":\"create\",\"name\":\"...\",\"description\":\"...\",\"body\":\"...\",\"tags\":[\"...\"]}; "
                "{\"action\":\"update\",\"skill_id\":\"...\",\"description\":\"...\",\"body\":\"...\",\"tags\":[\"...\"],\"reason\":\"...\"}."
            ),
        }
        try:
            raw = self.llm.chat(
                [
                    {
                        "role": "system",
                        "content": (
                            "You maintain a local reusable skill library for a coding agent. "
                            "You analyze task metrics and make informed decisions about skill creation and updates. "
                            "Return strict JSON only."
                        ),
                    },
                    {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
                ],
                temperature=0,
                max_tokens=3000,
            )
        except RuntimeError:
            return "none", None
        try:
            decision = _json_block(raw)
        except ValueError:
            return "none", None

        action = str(decision.get("action", "none")).lower()
        if action == "create":
            skill = self.skills.create(
                name=str(decision.get("name", "reusable_coding_pattern")),
                description=str(decision.get("description", "")),
                body=str(decision.get("body", "")),
                tags=[str(tag) for tag in decision.get("tags", []) if tag],
                source_task=task,
            )
            self.event_sink(f"[skills] created {skill.id}")
            return "create", skill.id
        if action == "update":
            skill_id = str(decision.get("skill_id", ""))
            if not self.skills.get(skill_id):
                return "none", None
            skill = self.skills.update(
                skill_id,
                description=str(decision.get("description", "")) or None,
                body=str(decision.get("body", "")) or None,
                tags=[str(tag) for tag in decision.get("tags", []) if tag],
                reason=str(decision.get("reason", "")),
            )
            self.event_sink(f"[skills] updated {skill.id}")
            return "update", skill.id
        return "none", None

    @staticmethod
    def _summarize_value(value: Any) -> Any:
        serialized = json.dumps(value, ensure_ascii=False, default=str)
        if len(serialized) <= 2000:
            return value
        return _truncate(serialized, 2000)

    def _summarize_task(
        self,
        *,
        task: str,
        success: bool,
        final: str,
        tool_calls: list[ToolCallLog],
    ) -> str:
        """Generate a concise summary: status, core changes, files modified."""
        files_touched = sorted({
            str(call.args.get("path", call.args.get("file", "")))
            for call in tool_calls
            if call.name in ("write_file", "edit_file") and call.args.get("path")
        })
        calls_summary = [
            f"{call.name}({', '.join(f'{k}={v!r}' for k, v in call.args.items())}) → {'✓' if call.success else '✗'}"
            for call in tool_calls[-10:]
        ]
        prompt = {
            "task": task,
            "success": success,
            "final": final,
            "files_touched": files_touched,
            "tool_calls": calls_summary,
            "instruction": (
                "Write a concise 3-point summary. Each point MUST be 1-2 sentences. No fluff. "
                "Return JSON only: "
                '{"status": "<success/failure and 1-line reason>", '
                '"core_changes": "<1-2 sentences describing what was done>", '
                '"files_modified": "<1-2 sentences listing key files and their roles>"}'
            ),
        }
        try:
            raw = self.llm.chat(
                [
                    {
                        "role": "system",
                        "content": "You summarize coding agent task results. Return strict JSON only. Be concise.",
                    },
                    {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
                ],
                temperature=0,
                max_tokens=600,
            )
        except RuntimeError:
            return self._fallback_summary(success, final, files_touched)
        try:
            parsed = _json_block(raw)
        except ValueError:
            return self._fallback_summary(success, final, files_touched)
        return (
            f"Status: {parsed.get('status', 'unknown')}\n"
            f"Changes: {parsed.get('core_changes', 'none')}\n"
            f"Files: {parsed.get('files_modified', 'none')}"
        )

    @staticmethod
    def _fallback_summary(success: bool, final: str, files_touched: list[str]) -> str:
        status = "Success." if success else "Failed."
        files = ", ".join(files_touched[:5]) if files_touched else "none"
        return (
            f"Status: {status}\n"
            f"Changes: {final[:200]}\n"
            f"Files: {files}"
        )

    def dump_result(self, result: AgentResult) -> str:
        return json.dumps(asdict(result), ensure_ascii=False, indent=2)
