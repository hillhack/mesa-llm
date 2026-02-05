# tests/test_llm_agent.py


import pytest
from mesa.model import Model
from mesa.space import ContinuousSpace, MultiGrid

from mesa_llm import Plan
from mesa_llm.llm_agent import LLMAgent
from mesa_llm.memory.st_memory import ShortTermMemory
from mesa_llm.reasoning.react import ReActReasoning

# MODULE-LEVEL CONSTANTS & HELPERS

DEFAULT_AGENT_CONFIG = {
    "reasoning": ReActReasoning,
    "system_prompt": "Test",
    "internal_state": ["test"],
}


class MockCell:
    """Mock cell with coordinate attribute"""

    def __init__(self, coordinate):
        self.coordinate = coordinate


# FIXTURES


@pytest.fixture(autouse=True)
def setup_api_key(monkeypatch):
    """Auto-set dummy API key for all tests"""
    monkeypatch.setenv("GEMINI_API_KEY", "dummy")


@pytest.fixture
def basic_model():
    """Create basic model without grid"""
    return Model(seed=42)


@pytest.fixture
def grid_model():
    """Create model with MultiGrid"""

    class GridModel(Model):
        def __init__(self):
            super().__init__(seed=42)
            self.grid = MultiGrid(10, 10, torus=False)

    return GridModel()


@pytest.fixture
def basic_agent(basic_model):
    """Create single agent with memory"""
    agents = LLMAgent.create_agents(basic_model, n=1, vision=0, **DEFAULT_AGENT_CONFIG)
    agent = agents[0]
    agent.memory = ShortTermMemory(agent=agent, n=5, display=True)
    return agent


@pytest.fixture
def disable_memory(monkeypatch):
    """Helper to disable memory add_to_memory method"""

    def _disable(agent):
        monkeypatch.setattr(agent.memory, "add_to_memory", lambda *args, **kwargs: None)

    return _disable


# TESTS


def test_apply_plan_adds_to_memory(grid_model, monkeypatch):
    """Test that applying a plan adds to memory step_content"""
    system_prompt = "You are an agent in a simulation."
    agents = LLMAgent.create_agents(
        grid_model,
        n=1,
        reasoning=ReActReasoning,
        system_prompt=system_prompt,
        vision=-1,
        internal_state=["test_state"],
    )
    agent = agents[0]
    grid_model.grid.place_agent(agent, (1, 1))
    agent.memory = ShortTermMemory(agent=agent, n=5, display=True)

    # Fake response from tool manager
    fake_response = [{"tool": "foo", "argument": "bar"}]

    # Monkeypatch to avoid real tool calls
    monkeypatch.setattr(
        agent.tool_manager, "call_tools", lambda agent, llm_response: fake_response
    )

    # Create and apply plan
    plan = Plan(step=0, llm_plan="do something")
    resp = agent.apply_plan(plan)

    # Check response matches
    assert resp == fake_response

    # Check memory step_content was updated
    assert {
        "tool": "foo",
        "argument": "bar",
    } in agent.memory.step_content.values() or agent.memory.step_content == {
        "tool": "foo",
        "argument": "bar",
    }


def test_generate_obs_with_one_neighbor(grid_model, disable_memory):
    """Test observation generation with one neighbor"""
    agents = LLMAgent.create_agents(
        grid_model,
        n=2,
        reasoning=ReActReasoning,
        system_prompt="Test",
        vision=-1,
        internal_state=["test"],
    )
    agent1, agent2 = agents
    agent1.unique_id = 1
    agent2.unique_id = 2
    agent1.memory = ShortTermMemory(agent=agent1, n=5, display=True)
    agent2.memory = ShortTermMemory(agent=agent2, n=5, display=True)

    grid_model.grid.place_agent(agent1, (1, 1))
    grid_model.grid.place_agent(agent2, (1, 2))

    disable_memory(agent1)
    obs = agent1.generate_obs()

    # Check agent's own state
    assert obs.self_state["agent_unique_id"] == 1

    # Should have exactly one neighboring agent
    assert len(obs.local_state) == 1

    # Extract and verify the neighbor
    key = next(iter(obs.local_state.keys()))
    assert key == "LLMAgent 2"

    entry = obs.local_state[key]
    assert entry["position"] == (1, 2)
    assert entry["internal_state"] == ["test"]


def test_send_message_updates_both_agents_memory(grid_model, monkeypatch):
    """Test that sending messages updates memory for both sender and recipient"""
    agents = LLMAgent.create_agents(
        grid_model,
        n=2,
        reasoning=lambda agent: None,
        system_prompt="Test",
        vision=-1,
        internal_state=["test_state"],
    )
    sender, recipient = agents
    sender.unique_id = 1
    recipient.unique_id = 2

    sender.memory = ShortTermMemory(agent=sender, n=5, display=True)
    recipient.memory = ShortTermMemory(agent=recipient, n=5, display=True)

    grid_model.grid.place_agent(sender, (0, 0))
    grid_model.grid.place_agent(recipient, (1, 1))

    # Track how many times add_to_memory is called
    call_counter = {"count": 0}

    original_add_to_memory = sender.memory.add_to_memory

    def counting_add_to_memory(*args, **kwargs):
        call_counter["count"] += 1
        return original_add_to_memory(*args, **kwargs)

    # Patch both agents' add_to_memory
    monkeypatch.setattr(sender.memory, "add_to_memory", counting_add_to_memory)
    monkeypatch.setattr(recipient.memory, "add_to_memory", counting_add_to_memory)

    # Send message
    sender.send_message("Hello", [recipient])

    # Should be called twice: once for sender, once for recipient
    assert call_counter["count"] == 2


def test_safer_cell_access_agent_with_cell_no_pos(basic_agent, disable_memory):
    """Test agent location uses cell.coordinate when pos=None"""
    basic_agent.pos = None
    basic_agent.cell = MockCell(coordinate=(3, 4))

    disable_memory(basic_agent)
    obs = basic_agent.generate_obs()

    assert obs.self_state["location"] == (3, 4)


def test_safer_cell_access_agent_without_cell_or_pos(basic_agent, disable_memory):
    """Test agent location returns None when no pos or cell"""
    basic_agent.pos = None
    if hasattr(basic_agent, "cell"):
        delattr(basic_agent, "cell")

    disable_memory(basic_agent)
    obs = basic_agent.generate_obs()

    assert obs.self_state["location"] is None


def test_safer_cell_access_neighbor_with_cell_no_pos(grid_model, disable_memory):
    """Test neighbor position uses cell.coordinate when pos=None"""
    agents = LLMAgent.create_agents(grid_model, n=2, vision=-1, **DEFAULT_AGENT_CONFIG)
    agent, neighbor = agents
    agent.unique_id = 1
    neighbor.unique_id = 2

    agent.memory = ShortTermMemory(agent=agent, n=5, display=True)
    neighbor.memory = ShortTermMemory(agent=neighbor, n=5, display=True)

    grid_model.grid.place_agent(agent, (1, 1))

    # Neighbor has cell but no pos
    neighbor.pos = None
    neighbor.cell = MockCell(coordinate=(2, 2))

    disable_memory(agent)
    obs = agent.generate_obs()

    assert obs.local_state["LLMAgent 2"]["position"] == (2, 2)


def test_safer_cell_access_neighbor_without_cell_or_pos(grid_model, disable_memory):
    """Test neighbor position returns None when no pos or cell"""
    agents = LLMAgent.create_agents(grid_model, n=2, vision=-1, **DEFAULT_AGENT_CONFIG)
    agent, neighbor = agents
    agent.unique_id = 1
    neighbor.unique_id = 2

    agent.memory = ShortTermMemory(agent=agent, n=5, display=True)
    neighbor.memory = ShortTermMemory(agent=neighbor, n=5, display=True)

    grid_model.grid.place_agent(agent, (1, 1))

    # Neighbor has no pos or cell
    neighbor.pos = None
    if hasattr(neighbor, "cell"):
        delattr(neighbor, "cell")

    disable_memory(agent)
    obs = agent.generate_obs()

    assert obs.local_state["LLMAgent 2"]["position"] is None


def test_generate_obs_with_continuous_space(basic_model, disable_memory):
    """Test neighbor detection with ContinuousSpace"""
    # Add ContinuousSpace to model
    basic_model.space = ContinuousSpace(x_max=10.0, y_max=10.0, torus=False)

    agents = LLMAgent.create_agents(
        basic_model, n=3, vision=2.0, **DEFAULT_AGENT_CONFIG
    )
    agent, nearby_neighbor, far_neighbor = agents
    agent.unique_id = 1
    nearby_neighbor.unique_id = 2
    far_neighbor.unique_id = 3

    for a in agents:
        a.memory = ShortTermMemory(agent=a, n=5, display=True)

    # Place agents
    basic_model.space.place_agent(agent, (5.0, 5.0))
    basic_model.space.place_agent(nearby_neighbor, (6.0, 5.0))  # Distance ~1.0
    basic_model.space.place_agent(far_neighbor, (9.0, 9.0))  # Distance ~5.66

    disable_memory(agent)
    obs = agent.generate_obs()

    # Should see nearby neighbor but not far one
    assert len(obs.local_state) == 1
    assert "LLMAgent 2" in obs.local_state
    assert "LLMAgent 3" not in obs.local_state


def test_generate_obs_vision_all_agents(grid_model, disable_memory):
    """Test vision=-1 to see all agents in simulation"""
    agents = LLMAgent.create_agents(grid_model, n=4, vision=-1, **DEFAULT_AGENT_CONFIG)

    for idx, a in enumerate(agents):
        a.unique_id = idx + 1
        a.memory = ShortTermMemory(agent=a, n=5, display=True)
        grid_model.grid.place_agent(a, (idx, idx))

    agent = agents[0]
    disable_memory(agent)
    obs = agent.generate_obs()

    # Should see ALL other 3 agents (not self)
    assert len(obs.local_state) == 3
    assert "LLMAgent 2" in obs.local_state
    assert "LLMAgent 3" in obs.local_state
    assert "LLMAgent 4" in obs.local_state


def test_generate_obs_no_grid_with_vision(basic_model, disable_memory):
    """Test fallback when model has no grid/space but vision > 0"""
    agents = LLMAgent.create_agents(basic_model, n=2, vision=5, **DEFAULT_AGENT_CONFIG)
    agent = agents[0]
    agent.unique_id = 1
    agent.memory = ShortTermMemory(agent=agent, n=5, display=True)

    disable_memory(agent)
    obs = agent.generate_obs()

    # Should return empty neighbors (fallback path)
    assert len(obs.local_state) == 0
