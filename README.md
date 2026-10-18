# SkillPilot

SkillPilot is a small, repository-aware coding agent for the terminal. It inspects an existing project, edits files in place, runs the project's own checks, and keeps a multi-turn conversation. The runtime uses only the Python standard library.

The 0.2 release replaces the old prompt-based JSON action protocol with native function calling. Official OpenAI endpoints use the Responses API; DeepSeek, Ollama, Open WebUI, and other OpenAI-compatible servers use Chat Completions.

## Install

```bash
python -m pip install -e .
```

The previous `skill-agent` executable remains as a compatibility alias.

Run one task:

```bash
skillpilot "find and fix the failing tests"
skillpilot -C ../another-repo "add pagination to the users endpoint"
```

Or open an interactive session:

```bash
skillpilot
```

## Configure a model

SkillPilot reads `.env` from the selected workspace, but regular environment variables take precedence.

### OpenAI Responses API

```dotenv
SKILLPILOT_PROVIDER=responses
SKILLPILOT_MODEL=gpt-5.6-terra
SKILLPILOT_BASE_URL=https://api.openai.com/v1
OPENAI_API_KEY=your-key
```

### OpenAI-compatible gateway or DeepSeek

```dotenv
SKILLPILOT_PROVIDER=chat
SKILLPILOT_MODEL=your-model-id
SKILLPILOT_BASE_URL=https://your-gateway.example/v1
OPENAI_API_KEY=your-key
```

### Local Ollama

```bash
ollama serve
skillpilot --provider chat --base-url http://localhost:11434/v1 \
  --model qwen3-coder "explain this repository"
```

Localhost endpoints do not require a dummy API key. Before starting a long run, check the effective configuration:

```bash
skillpilot --doctor
```

`--doctor` reports the provider, endpoint, model, whether a key is present, and whether the endpoint is reachable. It never prints the key.

## Useful workflows

```bash
# Let the agent edit and execute as needed (default)
skillpilot --mode auto "implement the feature and run tests"

# Ask before writes, deletes, tests, scripts, or shell commands
skillpilot --mode ask "upgrade the dependency"

# Explore or review without changing the workspace
skillpilot --mode read-only "review this branch for correctness issues"

# Machine-readable result and deterministic exit code
skillpilot --json "fix the parser bug"
```

Interactive commands:

```text
/status               effective workspace, provider, model, and mode
/doctor               test provider connectivity
/mode ask             change permission policy during the session
/diff                 show the current Git diff
/undo                 revert direct file edits from the previous task
/clear                reset model conversation state
/skills               list reusable local skills
/help                 show all commands
/exit                 quit
```

End a line with `\` to enter a multi-line task.

## What makes it repository-aware

At the start of every task SkillPilot refreshes a compact repository snapshot. It includes the file index and relevant content from `AGENTS.md`, `CLAUDE.md`, `.github/copilot-instructions.md`, README files, and common project manifests. The model can then inspect more files with tools instead of receiving an expensive dump of the entire tree.

The agent operates directly in the selected workspace. It does not invent a nested `workspace/` directory, does not silently commit or push, and does not read common credential files such as `.env`, private keys, or package-manager credential files.

## Recovery and safety

Direct `write_file`, `edit_file`, and `delete_file` operations are checkpointed under `.skillPilot/checkpoints`. `/undo` restores the previous contents or removes newly created files. Changes made indirectly by a shell command (formatters, generators, package managers, and similar tools) are intentionally not presented as fully reversible; use Git for those changes.

Permission modes are an execution boundary:

- `auto` lets the agent make scoped changes needed by the task.
- `ask` prompts before writes, deletes, tests, scripts, and shell commands. Enter `a` once to approve the rest of the current process.
- `read-only` blocks all direct mutation tools and shell commands.

## Provider behavior

Provider selection defaults to `auto`:

- `https://api.openai.com/...` selects the Responses API.
- Every other endpoint selects OpenAI-compatible Chat Completions.
- `--provider responses|chat` overrides detection.

Responses sessions use `previous_response_id` by default. Set `SKILLPILOT_STORE=false` to replay response items client-side instead. Compatible chat sessions replay assistant tool calls and tool results in message history.

Supported configuration variables:

| Variable | Purpose |
| --- | --- |
| `SKILLPILOT_PROVIDER` | `auto`, `responses`, or `chat` |
| `SKILLPILOT_MODEL` | active model ID |
| `SKILLPILOT_BASE_URL` | API base URL |
| `SKILLPILOT_API_KEY` | optional key alias; `OPENAI_API_KEY` also works |
| `SKILLPILOT_SEND_KEY_TO_LOCAL` | send a configured key to localhost; default `false` |
| `SKILLPILOT_REASONING` | `low`, `medium`, `high`, `xhigh`, or `max` |
| `SKILLPILOT_STORE` | Responses server-side state, default `true` |
| `SKILLPILOT_EMBEDDINGS` | opt into semantic skill search; lexical search is default |

The older `OPENAI_MODEL`, `OPENAI_BASE_URL`, and `OPENAI_API_BASE` names remain supported.

## Local skills

Skill search is fast and local by default. Add `--learn` to let substantial successful runs extract reusable project-independent knowledge into `.skillPilot/skills`. Learning is opt-in because it adds a model call and not every coding task deserves permanent memory.

## Development

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
python -m ruff check skillPilot tests
```

The test suite covers workspace containment, file editing, command execution, checkpoints, secret-file blocking, skill quality, both API protocols, native tool result continuation, and the agent loop.
