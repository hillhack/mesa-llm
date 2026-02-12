from unittest.mock import AsyncMock, patch

import pytest

from mesa_llm.memory.lt_memory import LongTermMemory
from mesa_llm.memory.memory import MemoryEntry


class TestLTMemory:
    """Test the Memory class core functionality"""

    def test_memory_initialization(self, mock_agent):
        """Test Memory class initialization with defaults and custom values"""
        memory = LongTermMemory(
            agent=mock_agent,
            llm_model="provider/test_model",
        )

        assert memory.agent == mock_agent
        assert memory.long_term_memory == ""
        assert memory.llm.system_prompt is not None

    def test_update_long_term_memory(self, mock_agent, mock_llm):
        """Test updating long-term memory functionality"""
        # Mock the LLM's generate method
        mock_llm.generate.return_value = "Updated long-term memory"

        memory = LongTermMemory(agent=mock_agent, llm_model="provider/test_model")
        # Replace the real LLM with our mock
        memory.llm = mock_llm
        memory.long_term_memory = "Previous memory"

        # Add some content to buffer
        memory.buffer = MemoryEntry(
            agent=mock_agent,
            content={"message": "Test message"},
            step=1,
        )

        memory._update_long_term_memory()

        # Verify LLM can call with correct prompt structure
        call_args = mock_llm.generate.call_args[0][0]
        assert "new memory entry" in call_args
        assert "Long term memory" in call_args

        assert memory.long_term_memory == "Updated long-term memory"

    # process step test
    def test_process_step(self, mock_agent):
        """Test process_step functionality"""
        memory = LongTermMemory(agent=mock_agent, llm_model="provider/test_model")

        # Add some content
        memory.add_to_memory("observation", {"content": "Test observation"})
        memory.add_to_memory("plan", {"content": "Test plan"})

        # Process the step
        with (
            patch("rich.console.Console"),
            patch.object(memory.llm, "generate", return_value="mocked summary"),
        ):
            memory.process_step(pre_step=True)
            assert isinstance(memory.buffer, MemoryEntry)
            # assert memory.buffer is not None

            # Process post-step
            memory.process_step(pre_step=False)
            assert memory.long_term_memory == "mocked summary"

    # format memories test
    def test_format_long_term(self, mock_agent):
        """Test formatting long-term memory"""
        memory = LongTermMemory(agent=mock_agent, llm_model="provider/test_model")
        memory.long_term_memory = "Long-term summary"

        assert memory.format_long_term() == "Long-term summary"

    @pytest.mark.asyncio
    async def test_aupdate_long_term_memory(self, mock_agent, mock_llm):
        """
        The _aupdate_long_term_memory is the async version of the update_long_term_memory, it checks whether the agenerate() method is called
        and also stores the return value to self.long_term_memory

        The test function ensures that these 2 conditions are checked and verified.
        """
        mock_llm.agenerate = AsyncMock(return_value="async summary")

        memory = LongTermMemory(agent=mock_agent, llm_model="provider/test_model")
        memory.llm = mock_llm
        memory.buffer = "buffer"

        await memory._aupdate_long_term_memory()

        # checks whether agenerate() function was called once
        mock_llm.agenerate.assert_called_once()

        # checks to ensure that the the long term memory is updated correctly with the value we gave through agenerate()
        assert memory.long_term_memory == "async summary"

    @pytest.mark.asyncio
    async def test_aprocess_step(self, mock_agent):
        """
        Test asynchronous aprocess_step functionality

        This test is performed in 2 parts ,
            - If pre_step = True then a new memory entry is created and this must be verified.
            - If pre_step = False then a according to the aprocess_step function the previous content is restored and this is set to as the new memory entry
              the check verifies this behavior.
        """
        memory = LongTermMemory(agent=mock_agent, llm_model="provider/test_model")

        # populate with content
        memory.add_to_memory("observation", {"content": "Test observation"})
        memory.add_to_memory("plan", {"content": "Test plan"})

        # Mock async LLM call
        memory.llm.agenerate = AsyncMock(return_value="mocked async summary")

        with patch("rich.console.Console"):
            # Pre-step assertion is used to check if a new memory entry was created correctly as intended
            await memory.aprocess_step(pre_step=True)
            assert isinstance(memory.buffer, MemoryEntry)
            assert memory.buffer.step is None

            # Post-step assertion is used to check if the previous content was restored
            await memory.aprocess_step(pre_step=False)
            assert memory.long_term_memory == "mocked async summary"
            assert memory.step_content == {}
            assert memory.buffer.step is not None
