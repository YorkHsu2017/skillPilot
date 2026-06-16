from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from .agent import CodingAgent
from .progress import enter_alternate_screen, exit_alternate_screen


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


def _print_event(message: str) -> None:
    print(message, file=sys.stderr)


def _print_stream(token: str) -> None:
    sys.stdout.write(token)
    sys.stdout.flush()


def _quality_bar(score: float) -> str:
    """Return a visual quality bar: [████░░] 0.67."""
    filled = int(score * 10)
    bar = "█" * filled + "░" * (10 - filled)
    return f"[{bar}] {score:.2f}"


def _print_help() -> None:
    print(
        "\n".join(
            [
                "Commands:",
                "  /help              show this help",
                "  /skills            list learned skills (with quality score)",
                "  /skill <skill_id>  show one skill as JSON",
                "  /audit             validate all skills and auto-deprecate weak ones",
                "  /exit              quit",
                "",
                "Any other input is treated as a coding task.",
            ]
        )
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Skill Agent CLI")
    parser.add_argument("--workspace", default=os.getcwd(), help="Workspace where files are read/written")
    parser.add_argument("--skills-dir", default=None, help="Skill storage directory")
    parser.add_argument("--env-file", default=".env", help="Optional env file to load before creating the LLM")
    parser.add_argument("--once", default=None, help="Run one task non-interactively")
    parser.add_argument("--max-steps", type=int, default=12, help="Maximum tool calls per task")
    parser.add_argument("--json", action="store_true", help="Print full JSON result for --once")
    parser.add_argument("--quiet", action="store_true", help="Suppress tool progress events")
    parser.add_argument("--stream", action="store_true", help="Stream LLM responses token-by-token")
    parser.add_argument("--no-progress", action="store_true", help="Disable animated progress bar")
    parser.add_argument("--no-skill-evolution", action="store_true", help="Disable skill creation and updates")
    return parser


def _output_line(line: str) -> None:
    sys.stderr.write(f"  {line}\n")
    sys.stderr.flush()


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _load_env_file(Path(args.env_file).expanduser())

    use_progress = not args.no_progress and not args.quiet and sys.stderr.isatty()
    use_stream = args.stream and sys.stdout.isatty()

    if use_progress:
        enter_alternate_screen()

    try:
        agent = CodingAgent(
            workspace=args.workspace,
            skills_dir=args.skills_dir,
            max_steps=args.max_steps,
            event_sink=(lambda _message: None) if args.quiet else _print_event,
            stream_sink=(lambda token: _print_stream(token)) if use_stream else None,
            output_sink=_output_line,
            progress=use_progress,
            enable_skill_evolution=not args.no_skill_evolution,
        )

        if args.once:
            result = agent.run(args.once)
            print(agent.dump_result(result) if args.json else result.final)
            return 0 if result.success else 1

        print(f"Skill Agent CLI workspace={Path(args.workspace).resolve()}")
        print("Type /help for commands, /exit to quit.")
        while True:
            try:
                user_input = input("skill-agent> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                return 0
            if not user_input:
                continue
            if user_input in {"/exit", "/quit"}:
                return 0
            if user_input == "/help":
                _print_help()
                continue
            if user_input == "/skills":
                skills = agent.skills.list(include_inactive=True)
                if not skills:
                    print("No skills learned yet.")
                    continue
                for skill in skills:
                    status_marker = "⚠ DEPRECATED" if skill.status == "deprecated" else "✓ active"
                    quality_bar = _quality_bar(skill.quality_score)
                    print(
                        f"{skill.id} | v{skill.version} | {status_marker} | "
                        f"Q={quality_bar} | used={skill.usage_count} "
                        f"| ok={skill.success_count} | {skill.name}: {skill.description}"
                    )
                    if skill.status == "deprecated":
                        print(f"  └─ deprecated: {skill.deprecated_reason}")
                continue
            if user_input.startswith("/skill "):
                skill_id = user_input.split(maxsplit=1)[1]
                skill = agent.skills.get(skill_id)
                if skill is None:
                    print(f"Skill not found: {skill_id}")
                else:
                    print(json.dumps(skill.to_dict(), ensure_ascii=False, indent=2))
                continue

            if user_input == "/audit":
                print("Running skill audit...")
                # Re-validate all skills
                all_skills = agent.skills.list(include_inactive=False)
                if not all_skills:
                    print("No active skills to audit.")
                    continue
                validated = 0
                for skill in all_skills:
                    result = agent.skills.validate_skill(skill.id)
                    if result:
                        validated += 1
                        print(
                            f"  ✓ {skill.id}: quality={result['quality_score']:.2f} "
                            f"(static={result['static_score']:.2f}, "
                            f"syntax={result['syntax_score']:.2f}, "
                            f"llm={result['llm_score']:.2f})"
                        )
                # Run deprecation check
                deprecated = agent.skills.check_and_deprecate()
                if deprecated:
                    for dep in deprecated:
                        print(f"  ⚠ DEPRECATED {dep['skill_id']}: {dep['reason']}")
                else:
                    print(f"  All {validated} skills passed quality checks.")
                continue

            result = agent.run(user_input)
            print(result.final)
    finally:
        if use_progress:
            exit_alternate_screen()


if __name__ == "__main__":
    raise SystemExit(main())
