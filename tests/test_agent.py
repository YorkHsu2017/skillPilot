from skillPilot.agent import CodingAgent
from skillPilot.llm import LLMError, ModelTurn, ToolCall


class FakeSession:
    def __init__(self, fail=False):
        self.instructions = ""
        self.tools = []
        self.fail = fail

    def send_user(self, task):
        if self.fail:
            raise LLMError("offline")
        return ModelTurn(
            tool_calls=[
                ToolCall(
                    id="call_1",
                    name="write_file",
                    arguments={"path": "hello.py", "content": "print('hello')\n"},
                )
            ]
        )

    def send_tool_outputs(self, turn, outputs):
        if outputs[0]["ok"]:
            return ModelTurn(text="Created hello.py and verified the write.")
        return ModelTurn(text="The requested write was blocked by the permission policy.")

    def reset(self):
        pass


class FakeLLM:
    api_style = "responses"
    base_url = "https://example.test/v1"
    model = "fake-coder"
    total_input_tokens = 10
    total_output_tokens = 5

    def __init__(self, fail=False):
        self.fail = fail

    def new_session(self, *, instructions, tools):
        session = FakeSession(self.fail)
        session.instructions = instructions
        session.tools = tools
        return session


def test_agent_edits_existing_workspace_and_can_undo(tmp_path):
    (tmp_path / "README.md").write_text("project docs", encoding="utf-8")
    agent = CodingAgent(workspace=tmp_path, llm=FakeLLM(), progress=False)
    result = agent.run("Create hello.py")

    assert result.success is True
    assert (tmp_path / "hello.py").read_text(encoding="utf-8") == "print('hello')\n"
    assert result.changed_files == ["hello.py"]
    assert result.checkpoint_id
    assert "README.md" in agent._session.instructions
    assert "Create a new directory only when the task itself calls for one" in agent._session.instructions

    undone = agent.undo_last()
    assert undone["ok"] is True
    assert not (tmp_path / "hello.py").exists()


def test_agent_reports_provider_failure_without_claiming_success(tmp_path):
    agent = CodingAgent(workspace=tmp_path, llm=FakeLLM(fail=True), progress=False)
    result = agent.run("Do something")
    assert result.success is False
    assert "offline" in result.final
    assert result.error == "offline"


def test_read_only_mode_blocks_mutation(tmp_path):
    agent = CodingAgent(
        workspace=tmp_path,
        llm=FakeLLM(),
        progress=False,
        permission_mode="read-only",
    )
    result = agent.run("Create hello.py")
    assert result.success is False
    assert result.tool_calls[0].success is False
    assert "read-only" in result.tool_calls[0].result["error"]
    assert not (tmp_path / "hello.py").exists()
