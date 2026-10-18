import json

from skillPilot.llm import ModelTurn, OpenAICompatibleLLM

TOOLS = [
    {
        "type": "function",
        "name": "read_file",
        "description": "Read a file",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
            "additionalProperties": False,
        },
        "strict": True,
    }
]


def test_provider_auto_selects_responses_only_for_official_openai():
    official = OpenAICompatibleLLM(api_key="x", base_url="https://api.openai.com/v1")
    local = OpenAICompatibleLLM(base_url="http://localhost:11434/v1")
    assert official.api_style == "responses"
    assert local.api_style == "chat"
    assert local.configured is True


def test_base_url_accepts_api_root_or_full_endpoint():
    official = OpenAICompatibleLLM(api_key="x", base_url="https://api.openai.com")
    compatible = OpenAICompatibleLLM(
        api_key="x", base_url="https://gateway.test/v1/chat/completions"
    )
    assert official.base_url == "https://api.openai.com/v1"
    assert compatible.base_url == "https://gateway.test/v1"


def test_openai_key_is_not_forwarded_to_localhost_by_accident(monkeypatch):
    monkeypatch.delenv("SKILLPILOT_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "remote-secret")
    client = OpenAICompatibleLLM(base_url="http://localhost:11434/v1")
    assert "Authorization" not in client._headers()

    monkeypatch.setenv("SKILLPILOT_API_KEY", "intentional-local-key")
    explicit = OpenAICompatibleLLM(base_url="http://localhost:11434/v1")
    assert explicit._headers()["Authorization"] == "Bearer intentional-local-key"


def test_responses_session_uses_native_tool_outputs(monkeypatch):
    client = OpenAICompatibleLLM(
        provider="responses",
        api_key="x",
        base_url="https://example.test/v1",
        model="test-model",
    )
    requests = []
    replies = iter(
        [
            {
                "id": "resp_1",
                "status": "completed",
                "output": [
                    {
                        "type": "function_call",
                        "call_id": "call_1",
                        "name": "read_file",
                        "arguments": '{"path":"README.md"}',
                    }
                ],
            },
            {
                "id": "resp_2",
                "status": "completed",
                "output": [
                    {
                        "type": "message",
                        "content": [{"type": "output_text", "text": "Finished."}],
                    }
                ],
            },
        ]
    )

    def fake_request(method, path, payload=None):
        requests.append((method, path, payload))
        return next(replies)

    monkeypatch.setattr(client, "_request_json", fake_request)
    session = client.new_session(instructions="Be useful", tools=TOOLS)
    first = session.send_user("Inspect the readme")
    assert first.tool_calls[0].arguments == {"path": "README.md"}

    second = session.send_tool_outputs(first, [{"ok": True, "content": "hello"}])
    assert second.text == "Finished."
    assert requests[0][1] == "/responses"
    assert requests[0][2]["tools"] == TOOLS
    assert requests[1][2]["previous_response_id"] == "resp_1"
    output = requests[1][2]["input"][0]
    assert output["type"] == "function_call_output"
    assert output["call_id"] == "call_1"
    assert json.loads(output["output"])["ok"] is True


def test_chat_session_replays_assistant_tool_call(monkeypatch):
    client = OpenAICompatibleLLM(
        provider="chat", api_key="x", base_url="https://gateway.test/v1", model="model"
    )
    requests = []
    replies = iter(
        [
            {
                "choices": [
                    {
                        "finish_reason": "tool_calls",
                        "message": {
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_a",
                                    "type": "function",
                                    "function": {
                                        "name": "read_file",
                                        "arguments": '{"path":"a.py"}',
                                    },
                                }
                            ],
                        },
                    }
                ]
            },
            {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": "All good."},
                    }
                ]
            },
        ]
    )

    def fake_request(method, path, payload=None):
        requests.append((method, path, payload))
        return next(replies)

    monkeypatch.setattr(client, "_request_json", fake_request)
    session = client.new_session(instructions="System", tools=TOOLS)
    first = session.send_user("Read a.py")
    second = session.send_tool_outputs(first, [{"ok": True}])

    assert second.text == "All good."
    messages = requests[1][2]["messages"]
    assert messages[0] == {"role": "system", "content": "System"}
    assert messages[-2]["role"] == "assistant"
    assert messages[-1]["role"] == "tool"
    # Responses-only strict metadata is not sent to compatible chat gateways.
    assert "strict" not in requests[0][2]["tools"][0]["function"]


def test_chat_session_keeps_completed_assistant_reply_for_next_user(monkeypatch):
    client = OpenAICompatibleLLM(
        provider="chat", api_key="x", base_url="https://gateway.test/v1"
    )
    requests = []

    def fake_request(method, path, payload=None):
        requests.append(payload)
        text = "First answer" if len(requests) == 1 else "Second answer"
        return {"choices": [{"finish_reason": "stop", "message": {"content": text}}]}

    monkeypatch.setattr(client, "_request_json", fake_request)
    session = client.new_session(instructions="System", tools=[])
    session.send_user("First question")
    session.send_user("Follow-up")
    assert requests[1]["messages"][-2:] == [
        {"role": "assistant", "content": "First answer"},
        {"role": "user", "content": "Follow-up"},
    ]


def test_stateless_responses_replays_items_instead_of_previous_id(monkeypatch):
    client = OpenAICompatibleLLM(
        provider="responses",
        api_key="x",
        base_url="https://example.test/v1",
        store=False,
    )
    requests = []
    replies = iter(
        [
            {
                "id": "resp_1",
                "output": [
                    {
                        "type": "function_call",
                        "call_id": "call_1",
                        "name": "read_file",
                        "arguments": '{"path":"README.md"}',
                    }
                ],
            },
            {
                "id": "resp_2",
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "Done"}],
                    }
                ],
            },
        ]
    )

    def fake_request(method, path, payload=None):
        requests.append(payload)
        return next(replies)

    monkeypatch.setattr(client, "_request_json", fake_request)
    session = client.new_session(instructions="System", tools=TOOLS)
    first = session.send_user("Read it")
    session.send_tool_outputs(first, [{"ok": True}])
    assert "previous_response_id" not in requests[1]
    assert requests[1]["input"][0] == {"role": "user", "content": "Read it"}
    assert requests[1]["input"][1]["type"] == "function_call"
    assert requests[1]["input"][2]["type"] == "function_call_output"


def test_parse_responses_collects_all_text_parts():
    client = OpenAICompatibleLLM(provider="responses", api_key="x")
    turn = client._parse_responses_turn(
        {
            "output": [
                {
                    "type": "message",
                    "content": [
                        {"type": "output_text", "text": "one "},
                        {"type": "output_text", "text": "two"},
                    ],
                }
            ]
        }
    )
    assert isinstance(turn, ModelTurn)
    assert turn.text == "one two"
