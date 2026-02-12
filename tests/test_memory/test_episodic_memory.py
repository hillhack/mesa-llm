import json
from collections import deque
from unittest.mock import AsyncMock, MagicMock

import pytest

from mesa_llm.memory.episodic_memory import EpisodicMemory
from mesa_llm.memory.memory import MemoryEntry


@pytest.fixture
def mock_agent():
    agent = MagicMock(name="MockLLMAgent")

    # Create a MagicMock for the LLM's response
    mock_response = MagicMock()

    # This line *defines* the full nested path on the mock
    mock_response.choices[0].message.content = json.dumps({"grade": 3})

    # Set this as the return value
    agent.llm.generate.return_value = mock_response

    agent.model.steps = 100
    agent.model.events = []
    return agent


class TestEpisodicMemory:
    """Core functionality test"""

    def test_memory_init(self, mock_agent):
        """Test EpisodicMemory class initialization with defaults and custom values"""
        memory = EpisodicMemory(
            agent=mock_agent,
            max_capacity=10,
            considered_entries=5,
            llm_model="provider/test_model",
        )

        assert memory.agent == mock_agent
        assert memory.max_capacity == 10
        assert memory.considered_entries == 5
        assert isinstance(memory.memory_entries, deque)
        assert memory.memory_entries.maxlen == 10
        assert memory.system_prompt is not None
        """FYI: The above line may not always work; use the one below if needed."""
        # assert isinstance(memory.system_prompt,str), memory.system_prompt.strip() != ""

    def test_add_memory_entry(self, mock_agent):
        """Test adding memories to Episodic memory"""
        memory = EpisodicMemory(agent=mock_agent, llm_model="provider/test_model")

        # Test basic addition with observation
        memory.add_to_memory("observation", {"step": 1, "content": "Test content"})

        # Test with planning
        memory.add_to_memory("planning", {"plan": "Test plan", "importance": "high"})

        # Test with action
        memory.add_to_memory("action", {"action": "Test action"})

        # Should be empty step_content initially
        assert memory.step_content != {}

    def test_grade_event_importance(self, mock_agent):
        """Test grading event importance"""
        memory = EpisodicMemory(agent=mock_agent, llm_model="provider/test_model")

        # 1. Set up a specific grade for this test
        mock_response = MagicMock()
        mock_response.choices[0].message.content = json.dumps({"grade": 5})
        mock_agent.llm.generate.return_value = mock_response

        # 2. Call the method
        grade = memory.grade_event_importance("observation", {"data": "critical info"})

        # 3. Assert the grade is correct
        assert grade == 5

        # 4. Assert the LLM was called correctly
        mock_agent.llm.generate.assert_called_once()

        # Check that the system prompt was set on the llm object
        assert memory.llm.system_prompt == memory.system_prompt

    def test_retrieve_top_k_entries(self, mock_agent):
        """Test the sorting logic for retrieving entries (importance - recency_penalty)."""
        memory = EpisodicMemory(agent=mock_agent, llm_model="provider/test_model")
        # Set current step
        mock_agent.model.steps = 100

        # Manually add entries to bypass grading and control scores
        # score = importance - (current_step - entry_step)

        # score = 5 - (100 - 98) = 3
        entry_a = MemoryEntry(
            content={"importance": 5, "id": "A"}, step=98, agent=mock_agent
        )
        # score = 1 - (100 - 99) = 0
        entry_b = MemoryEntry(
            content={"importance": 1, "id": "B"}, step=99, agent=mock_agent
        )
        # score = 4 - (100 - 90) = -6
        entry_c = MemoryEntry(
            content={"importance": 4, "id": "C"}, step=90, agent=mock_agent
        )
        # score = 4 - (100 - 95) = -1
        entry_d = MemoryEntry(
            content={"importance": 4, "id": "D"}, step=95, agent=mock_agent
        )

        memory.memory_entries.extend([entry_a, entry_b, entry_c, entry_d])

        # Retrieve top 3 (k=3)
        top_entries = memory.retrieve_top_k_entries(3)

        # Expected order: A (3), B (0), D (-1)
        assert len(top_entries) == 3
        assert top_entries[0].content["id"] == "A"
        assert top_entries[1].content["id"] == "B"
        assert top_entries[2].content["id"] == "D"

        # Entry C (score -6) should be omitted
        assert "C" not in [e.content["id"] for e in top_entries]

    def test_process_step_pre_step(self, mock_agent):
        """
        The process_step function in the episodic_memory when called with 'pre_step=True' takes whatever is already inside the step_content,
        then calls the add_to_memory function and then clears the step_content.

        This test function performs the following 2 tests,
            - Checks whether the add_to_memory function is called correctly when 'pre_step=True.'
            - Also performs a final check to ensure the step_content is cleared.
        """
        memory = EpisodicMemory(agent=mock_agent, llm_model="provider/test_model")

        # Pre-populate step_content
        memory.step_content = {"observation": {"data": "test"}}

        # Spy on add_to_memory and call the process step with param as True
        memory.add_to_memory = MagicMock()
        memory.process_step(pre_step=True)

        # Checks if add_to_memory was called once
        memory.add_to_memory.assert_called_once_with(
            type="observation",
            content={"observation": {"data": "test"}},
        )

        # checks whether the step_content is cleared at the end
        assert memory.step_content == {}

    @pytest.mark.asyncio
    async def test_aprocess_step_pre_step(self, mock_agent):
        """
        Asynchronous version of the 'test_process_step_pre_step'
        Implements the same checks as the sync counterpart function but in an async manner.
            - checks whether aadd_to_memory function was called correctly
            - checks whether the step_content was cleared correctly at the end
        """
        memory = EpisodicMemory(agent=mock_agent, llm_model="provider/test_model")

        memory.step_content = {"observation": {"data": "test"}}
        memory.aadd_to_memory = AsyncMock()

        await memory.aprocess_step(pre_step=True)

        #
        memory.aadd_to_memory.assert_awaited_once_with(
            type="observation",
            content={"observation": {"data": "test"}},
        )

        assert memory.step_content == {}

    @pytest.mark.asyncio
    async def test_async_add_memory_entry(self, mock_agent):
        """
        The aadd_to_memory function assigns an 'importance' value to the content and then calls the add_to_memory function

        The test function does the following
            - mocks the llm to produece a pre-determined grading.
            - then calls the aad_to_memory function
            - checks to ensure that the step_content is not empty as the aadd_to_memory function will have added entries into it.
        """
        memory = EpisodicMemory(agent=mock_agent, llm_model="provider/test_model")

        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(message=MagicMock(content=json.dumps({"grade": 3})))
        ]

        # Assigns the mock response
        mock_agent.llm.agenerate = AsyncMock(return_value=mock_response)

        # adds content into the memory using the async counter part of add_to_memory function
        await memory.aadd_to_memory("observation", {"content": "Test content"})
        await memory.aadd_to_memory("planning", {"plan": "Test plan"})
        await memory.aadd_to_memory("action", {"action": "Test action"})

        # checks to ensure that step content is not empty
        assert memory.step_content != {}

    def test_build_grade_prompt_no_previous_entries(self, mock_agent):
        """
        The _build_grade_prompt function inserts 'No previous memory entries this message if there are no entries passed to it.

        This test function checks to see if this fall-back indeed works correctly
            - No memory entries are added before the _build_grade_prompt function call
            - So when the memory is empty we expect to see 'No previous memory entries' in the returned prompt.
        """
        memory = EpisodicMemory(agent=mock_agent, llm_model="provider/test_model")

        prompt = memory._build_grade_prompt("observation", {"data": "test"})

        # checks if the fallback condition actaually works
        assert "No previous memory entries" in prompt
        assert "observation" in prompt

    def test_get_communication_history(self, mock_agent):
        """
        Return a formatted string of all messages stored in memory.

        This function:
        - Looks through all memory entries
        - Selects only entries that contain a "message" field
        - Formats each message as: "step <step_number>: <message>"
        - Combines them into one single string

        Returns:
            str: A string containing all communication messages
                from memory, separated by new lines.
        """
        memory = EpisodicMemory(agent=mock_agent, llm_model="provider/test_model")

        entry_with_message = MemoryEntry(
            content={"importance": 3, "message": "Hello"},
            step=1,
            agent=mock_agent,
        )

        entry_without_message = MemoryEntry(
            content={"importance": 2, "data": "No message here"},
            step=2,
            agent=mock_agent,
        )

        memory.memory_entries.append(entry_with_message)
        memory.memory_entries.append(entry_without_message)

        history = memory.get_communication_history()

        # assertion checks must return true
        assert "Hello" in history
        assert "step 1" in history
        assert (
            "No message here" not in history
        )  # step 2  does not have message field thus it must not be present in the returned string
