# Skill Agent

A compact coding agent inspired by Copilot CLI and the EvoTool skill lifecycle idea. It runs in a terminal, accepts multi-turn user requests, writes executable files in a workspace, calls tools, and maintains a local skill library.

## Quick start

```bash
cp .env.example .env
# Put your real OPENAI_API_KEY in .env, or export it in the shell.
python -m skillPilot.cli --once "implement a Python script that prints a Christmas tree"
python -m skillPilot.cli
```

Useful REPL commands:

```text
/help
/skills
/skill <skill_id>
/exit
```

## Behavior

For every new task the agent:

1. Searches `.skillPilot/skills` for related skills.
2. Asks the LLM whether those skills should be referenced.
3. Runs a JSON tool-calling loop with tools such as `grep`, `read_file`, `write_file`, `run_python`, and `run_pytest`.
4. Asks the LLM whether the completed trajectory should create a new skill, update an existing one, or do nothing.

The API client is OpenAI-compatible and reads `OPENAI_BASE_URL` or `OPENAI_API_BASE`, `OPENAI_API_KEY`, and `OPENAI_MODEL`.
