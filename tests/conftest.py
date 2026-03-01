import os
from unittest.mock import Mock, patch

import pytest
from litellm import Choices, Message, ModelResponse


def build_llm_response(
    content: str = "ok",
    *,
    role: str = "assistant",
    tool_calls: list | None = None,
    response_id: str = "mock-response-id",
    model: str = "openai/mock-model",
    created: int = 0,
) -> ModelResponse:
    """Create a typed LiteLLM response object for tests."""
    return ModelResponse(
        id=response_id,
        created=created,
        model=model,
        object="chat.completion",
        choices=[
            Choices(
                message=Message(
                    role=role,
                    content=content,
                    tool_calls=tool_calls,
                )
            )
        ],
    )


@pytest.fixture(autouse=True)
def mock_environment():
    """Ensure tests don't depend on real environment variables"""
    with patch.dict(
        os.environ,
        {"PROVIDER_API_KEY": "test_key", "OPENAI_API_KEY": "test_openai_key"},
        clear=True,
    ):
        yield


@pytest.fixture
def mock_agent():
    """Create a mock agent for testing"""
    agent = Mock()
    agent.__class__.__name__ = "TestAgent"
    agent.unique_id = 123
    agent.__str__ = Mock(return_value="TestAgent(123)")
    agent.model = Mock()
    agent.model.steps = 1
    agent.model.events = []
    agent.step_prompt = "Test step prompt"
    agent.llm = Mock()
    return agent


@pytest.fixture
def mock_llm():
    """Create a mock LLM for testing"""
    with patch("mesa_llm.module_llm.ModuleLLM") as mock_llm_class:
        mock_llm_instance = Mock()
        mock_llm_class.return_value = mock_llm_instance
        yield mock_llm_instance


@pytest.fixture
def llm_response_factory():
    """Fixture wrapper around typed LiteLLM response builder."""
    return build_llm_response
