"""SkillPilot command-line interface."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from . import __version__
from .agent import CodingAgent
from .llm import OpenAICompatibleLLM


def _load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'\"")
        if key and key not in os.environ:
            os.environ[key] = value


class Console:
    def __init__(self, *, quiet: bool = False) -> None:
        self.quiet = quiet
        self.color = sys.stderr.isatty() and "NO_COLOR" not in os.environ
        self.approve_all = False

    def _paint(self, text: str, code: str) -> str:
        return f"\033[{code}m{text}\033[0m" if self.color else text

    def event(self, message: str) -> None:
        if self.quiet:
            return
        if message.startswith("[tool"):
            message = self._paint(message, "36")
        elif message.startswith("[summary]"):
            message = self._paint(message, "2")
        elif message.startswith("[agent]"):
            message = self._paint(message, "33")
        print(message, file=sys.stderr)

    def output(self, line: str) -> None:
        if not self.quiet:
            print(self._paint(f"  │ {line}", "2"), file=sys.stderr)

    def approve(self, action: str, detail: str) -> bool:
        if self.approve_all:
            return True
        if not sys.stdin.isatty():
            return False
        preview = detail if len(detail) <= 140 else detail[:137] + "..."
        answer = input(f"Allow {action}: {preview}? [y/N/a(ll)] ").strip().lower()
        if answer in {"a", "all"}:
            self.approve_all = True
            return True
        return answer in {"y", "yes"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="skillpilot",
        description="A repository-aware coding agent with native tool calling.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("prompt", nargs="*", help="Task to run; omit for interactive mode")
    parser.add_argument("-C", "--workspace", default=os.getcwd(), help="Repository/workspace directory")
    parser.add_argument("-p", "--once", default=None, help="Backward-compatible single task option")
    parser.add_argument("--provider", choices=["auto", "responses", "chat"], default=None)
    parser.add_argument("--model", default=None, help="Model ID")
    parser.add_argument("--base-url", default=None, help="API base URL ending in /v1 when applicable")
    parser.add_argument("--reasoning", choices=["low", "medium", "high", "xhigh", "max"], default=None)
    parser.add_argument("--mode", choices=["auto", "ask", "read-only"], default="auto", help="Tool permission policy")
    parser.add_argument("--max-steps", type=int, default=20, help="Maximum model/tool rounds per task")
    parser.add_argument("--skills-dir", default=None, help="Skill storage directory")
    parser.add_argument("--learn", action="store_true", help="Learn reusable skills after substantial successful runs")
    parser.add_argument("--no-skill-evolution", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--env-file", default=".env", help="Optional dotenv file")
    parser.add_argument("--json", action="store_true", help="Emit a machine-readable result for a single task")
    parser.add_argument("--quiet", action="store_true", help="Hide progress and command output")
    parser.add_argument("--doctor", action="store_true", help="Check provider configuration and connectivity")
    parser.add_argument("--stream", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--no-progress", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def _make_client(args: argparse.Namespace) -> OpenAICompatibleLLM:
    return OpenAICompatibleLLM(
        provider=args.provider,
        model=args.model,
        base_url=args.base_url,
        reasoning_effort=args.reasoning,
    )


def _print_mapping(data: dict[str, Any]) -> None:
    width = max(len(str(key)) for key in data) if data else 0
    for key, value in data.items():
        print(f"{key:<{width}}  {value}")


def _help() -> None:
    print(
        """Commands
  /help                 Show this help
  /status               Show workspace, provider, model, and session state
  /doctor               Test API configuration and connectivity
  /mode <policy>        Switch between auto, ask, and read-only
  /diff                 Show the current Git diff
  /undo                 Revert direct file edits from the last task
  /clear                Start a fresh model conversation
  /skills               List reusable local skills
  /skill <id>           Show one skill
  /audit                Validate the skill library
  /exit                 Quit

Any other input is a coding task. End a line with \\ to continue it."""
    )


def _read_task() -> str:
    lines: list[str] = []
    prompt = "you › "
    while True:
        line = input(prompt)
        if line.endswith("\\"):
            lines.append(line[:-1])
            prompt = "  … "
            continue
        lines.append(line)
        return "\n".join(lines).strip()


def _run_once(agent: CodingAgent, task: str, as_json: bool) -> int:
    result = agent.run(task)
    print(agent.dump_result(result) if as_json else result.final)
    return 0 if result.success else 1


def _handle_command(agent: CodingAgent, command: str, client: OpenAICompatibleLLM) -> bool:
    name, _, argument = command.partition(" ")
    if name in {"/exit", "/quit"}:
        raise EOFError
    if name == "/help":
        _help()
    elif name == "/status":
        _print_mapping(agent.status())
    elif name == "/doctor":
        _print_mapping(client.diagnose())
    elif name == "/clear":
        agent.clear_session()
        print("Conversation cleared. Workspace files are unchanged.")
    elif name == "/undo":
        result = agent.undo_last()
        print(json.dumps(result, ensure_ascii=False, indent=2))
    elif name == "/diff":
        result = agent.registry.run("git_diff", {"staged": False, "path": None})
        print(result.get("output") or result.get("error") or "No diff.")
    elif name == "/mode":
        if argument not in {"auto", "ask", "read-only"}:
            print("Usage: /mode auto|ask|read-only")
        else:
            agent.permission_mode = argument
            print(f"Permission mode: {argument}")
    elif name == "/skills":
        skills = agent.skills.list(include_inactive=True)
        if not skills:
            print("No reusable skills yet.")
        for skill in skills:
            rate = skill.success_count / skill.usage_count if skill.usage_count else 0
            print(
                f"{skill.id}  {skill.name}  quality={skill.quality_score:.2f} "
                f"success={rate:.0%}  {skill.status}"
            )
    elif name == "/skill":
        skill = agent.skills.get(argument.strip())
        print(json.dumps(skill.to_dict(), ensure_ascii=False, indent=2) if skill else "Skill not found.")
    elif name == "/audit":
        deprecated = agent.skills.check_and_deprecate()
        print(f"Audit complete. Deprecated {len(deprecated)} skill(s).")
    else:
        print(f"Unknown command: {name}. Type /help.")
    return True


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    env_path = Path(args.env_file).expanduser()
    if not env_path.is_absolute():
        env_path = Path(args.workspace).expanduser() / env_path
    _load_env_file(env_path)

    console = Console(quiet=args.quiet)
    client = _make_client(args)
    if args.doctor:
        diagnosis = client.diagnose()
        _print_mapping(diagnosis)
        return 0 if diagnosis.get("ok") else 1

    agent = CodingAgent(
        workspace=args.workspace,
        skills_dir=args.skills_dir,
        llm=client,
        max_steps=args.max_steps,
        event_sink=console.event,
        output_sink=console.output,
        progress=not args.no_progress,
        enable_skill_evolution=args.learn and not args.no_skill_evolution,
        permission_mode=args.mode,
        approval_callback=console.approve,
    )
    single_task = args.once or " ".join(args.prompt).strip()
    if single_task:
        return _run_once(agent, single_task, args.json)

    status = agent.status()
    print(f"SkillPilot {__version__}  ·  {status['model']} via {status['provider']}")
    print(f"workspace  {status['workspace']}")
    print("Type /help for commands. Ctrl+C interrupts a run; Ctrl+D exits.")
    while True:
        try:
            task = _read_task()
            if not task:
                continue
            if task.startswith("/"):
                _handle_command(agent, task, client)
                continue
            result = agent.run(task)
            print(f"\n{result.final}\n")
        except KeyboardInterrupt:
            print("\nInterrupted. Type /undo to revert direct file edits.")
        except EOFError:
            print()
            return 0


if __name__ == "__main__":
    raise SystemExit(main())
