import json
from unittest.mock import patch, MagicMock

from src.agents.base import BaseAgent


def _mock_text_response(text):
    """Create a mock Messages API response with a text block."""
    block = MagicMock()
    block.type = "text"
    block.text = text

    response = MagicMock()
    response.content = [block]
    return response


@patch("src.agents.base.anthropic.Anthropic")
def test_call_llm_returns_text(mock_anthropic_cls):
    mock_client = MagicMock()
    mock_anthropic_cls.return_value = mock_client
    mock_client.messages.create.return_value = _mock_text_response("Hello, world!")

    agent = BaseAgent()
    result = agent._call_llm("You are helpful.", "Say hello")

    assert result == "Hello, world!"
    mock_client.messages.create.assert_called_once()
    call_kwargs = mock_client.messages.create.call_args.kwargs
    assert call_kwargs["model"] == "claude-opus-4-6"
    assert call_kwargs["system"] == "You are helpful."
    assert call_kwargs["messages"] == [{"role": "user", "content": "Say hello"}]


@patch("src.agents.base.anthropic.Anthropic")
def test_call_llm_json_returns_dict(mock_anthropic_cls):
    mock_client = MagicMock()
    mock_anthropic_cls.return_value = mock_client

    expected = {"name": "test", "value": 42}
    mock_client.messages.create.return_value = _mock_text_response(json.dumps(expected))

    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "value": {"type": "integer"},
        },
        "required": ["name", "value"],
        "additionalProperties": False,
    }

    agent = BaseAgent()
    result = agent._call_llm_json("You are helpful.", "Give me data", schema)

    assert result == expected
    call_kwargs = mock_client.messages.create.call_args.kwargs
    assert call_kwargs["output_config"]["format"]["type"] == "json_schema"
    assert call_kwargs["output_config"]["format"]["schema"] == schema


@patch("src.agents.base.anthropic.Anthropic")
def test_custom_model(mock_anthropic_cls):
    mock_client = MagicMock()
    mock_anthropic_cls.return_value = mock_client
    mock_client.messages.create.return_value = _mock_text_response("ok")

    agent = BaseAgent(model="claude-haiku-4-5")
    agent._call_llm("sys", "msg")

    call_kwargs = mock_client.messages.create.call_args.kwargs
    assert call_kwargs["model"] == "claude-haiku-4-5"


@patch("src.agents.base.anthropic.Anthropic")
def test_max_retries_passed_to_client(mock_anthropic_cls):
    BaseAgent(max_retries=5)
    mock_anthropic_cls.assert_called_once_with(max_retries=5)
