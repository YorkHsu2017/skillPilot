from __future__ import annotations

from skillPilot.tools import build_default_registry


def test_file_grep_python_and_pytest_tools(tmp_path):
    registry = build_default_registry(tmp_path)
    registry.run(
        "write_file",
        {
            "path": "hello.py",
            "content": "def greet(name):\n    return f'hello {name}'\n\nif __name__ == '__main__':\n    print(greet('agent'))\n",
        },
    )
    registry.run(
        "write_file",
        {
            "path": "test_hello.py",
            "content": "from hello import greet\n\ndef test_greet():\n    assert greet('agent') == 'hello agent'\n",
        },
    )

    read = registry.run("read_file", {"path": "hello.py"})
    assert "def greet" in read["content"]

    grep = registry.run("grep", {"pattern": "greet", "path": "."})
    assert grep["count"] >= 2

    run = registry.run("run_python", {"path": "hello.py"})
    assert run["ok"] is True
    assert "hello agent" in run["output"]

    pytest = registry.run("run_pytest", {"args": "-q"})
    assert pytest["ok"] is True


def test_tools_prevent_workspace_escape(tmp_path):
    registry = build_default_registry(tmp_path)
    result = None
    try:
        registry.run("write_file", {"path": "../outside.txt", "content": "nope"})
    except ValueError as exc:
        result = str(exc)
    assert result is not None
    assert "escapes workspace" in result


def test_edit_file_single_replace(tmp_path):
    registry = build_default_registry(tmp_path)
    registry.run(
        "write_file",
        {"path": "greet.py", "content": "def greet(name):\n    return f'hi {name}'\n"},
    )
    result = registry.run(
        "edit_file",
        {"path": "greet.py", "old_string": "hi", "new_string": "hello"},
    )
    assert result["ok"] is True
    assert result["replacements"] == 1
    content = registry.run("read_file", {"path": "greet.py"})["content"]
    assert "hello" in content
    assert "hi" not in content


def test_edit_file_replace_all(tmp_path):
    registry = build_default_registry(tmp_path)
    registry.run(
        "write_file",
        {"path": "data.txt", "content": "foo bar foo baz foo\n"},
    )
    result = registry.run(
        "edit_file",
        {"path": "data.txt", "old_string": "foo", "new_string": "qux", "replace_all": True},
    )
    assert result["ok"] is True
    assert result["replacements"] == 3
    content = registry.run("read_file", {"path": "data.txt"})["content"]
    assert "foo" not in content
    assert content.count("qux") == 3


def test_edit_file_ambiguous_without_replace_all(tmp_path):
    registry = build_default_registry(tmp_path)
    registry.run(
        "write_file",
        {"path": "dup.txt", "content": "x x x\n"},
    )
    result = registry.run(
        "edit_file",
        {"path": "dup.txt", "old_string": "x", "new_string": "y"},
    )
    assert result["ok"] is False
    assert "matches" in result.get("error", "").lower() or result.get("match_count", 0) > 1


def test_edit_file_not_found(tmp_path):
    registry = build_default_registry(tmp_path)
    registry.run(
        "write_file",
        {"path": "sample.txt", "content": "hello world\n"},
    )
    result = registry.run(
        "edit_file",
        {"path": "sample.txt", "old_string": "nope", "new_string": "yep"},
    )
    assert result["ok"] is False
    assert "not found" in result.get("error", "")


def test_bash_simple_echo(tmp_path):
    registry = build_default_registry(tmp_path)
    result = registry.run("bash", {"command": "echo hello bash"})
    assert result["ok"] is True
    assert "hello bash" in result["output"]


def test_bash_failing_command(tmp_path):
    registry = build_default_registry(tmp_path)
    result = registry.run("bash", {"command": "exit 1"})
    assert result["ok"] is False
    assert result["exit_code"] == 1


def test_sensitive_files_cannot_be_read_or_listed(tmp_path):
    registry = build_default_registry(tmp_path)
    (tmp_path / ".env").write_text("TOKEN=secret", encoding="utf-8")
    listing = registry.run("list_files", {"path": "."})
    assert ".env" not in listing["files"]
    try:
        registry.run("read_file", {"path": ".env"})
    except PermissionError as exc:
        assert "credential" in str(exc)
    else:
        raise AssertionError("Expected sensitive-file read to be blocked")

    git_config = tmp_path / ".git" / "config"
    git_config.parent.mkdir()
    git_config.write_text("remote secret", encoding="utf-8")
    try:
        registry.run("read_file", {"path": ".git/config"})
    except PermissionError as exc:
        assert "internal" in str(exc)
    else:
        raise AssertionError("Expected internal-file read to be blocked")


def test_checkpoint_restores_edited_file(tmp_path):
    registry = build_default_registry(tmp_path)
    target = tmp_path / "value.txt"
    target.write_text("before", encoding="utf-8")
    registry.begin_checkpoint("edit value")
    registry.run(
        "edit_file",
        {"path": "value.txt", "old_string": "before", "new_string": "after"},
    )
    checkpoint = registry.finish_checkpoint()
    assert checkpoint
    assert target.read_text(encoding="utf-8") == "after"
    assert registry.undo_last()["ok"] is True
    assert target.read_text(encoding="utf-8") == "before"
