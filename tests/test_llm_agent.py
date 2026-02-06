# tests/test_llm_agent.py

import re

import pytest
from mesa.discrete_space import OrthogonalMooreGrid
from mesa.space import ContinuousSpace

from mesa_llm import Plan
from mesa_llm.llm_agent import LLMAgent
from mesa_llm.memory.st_memory import ShortTermMemory
from mesa_llm.reasoning.react import ReActReasoning
from tests.conftest import DEFAULT_AGENT_CONFIG

# Fixtures (now in conftest.py):
# - mock_environment
# - basic_model
# - grid_model
# - basic_agent
# - disable_memory
# - DEFAULT_AGENT_CONFIG
# - MockCell


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
    agents = LLMAgent.create_agents(grid_model, n=2, vision=-1, **DEFAULT_AGENT_CONFIG)
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

    # Send message and verify format
    result = sender.send_message("Hello", [recipient])
    pattern = r"LLMAgent 1 → \[<mesa_llm\.llm_agent\.LLMAgent object at 0x[0-9A-Fa-f]+>\] : Hello"
    assert re.match(pattern, result)

    # Should be called twice: once for sender, once for recipient
    assert call_counter["count"] == 2


# Parametrized tests for safer cell access scenarios
@pytest.mark.parametrize(
    "test_case,agent_setup,expected_location",
    [
        (
            "agent_with_cell_no_pos",
            lambda agent: setattr(
                agent, "cell", type("MockCell", (), {"coordinate": (3, 4)})()
            )
            or setattr(agent, "pos", None),
            (3, 4),
        ),
        (
            "agent_without_cell_or_pos",
            lambda agent: setattr(agent, "pos", None)
            or (delattr(agent, "cell") if hasattr(agent, "cell") else None),
            None,
        ),
    ],
)
def test_safer_cell_access_self(
    basic_agent, disable_memory, test_case, agent_setup, expected_location
):
    """Test agent self location with various cell/pos configurations"""
    agent_setup(basic_agent)
    disable_memory(basic_agent)
    obs = basic_agent.generate_obs()
    assert obs.self_state["location"] == expected_location


@pytest.mark.parametrize(
    "test_case,neighbor_setup,expected_position",
    [
        (
            "neighbor_with_cell_no_pos",
            lambda neighbor: setattr(
                neighbor, "cell", type("MockCell", (), {"coordinate": (2, 2)})()
            )
            or setattr(neighbor, "pos", None),
            (2, 2),
        ),
        (
            "neighbor_without_cell_or_pos",
            lambda neighbor: setattr(neighbor, "pos", None)
            or (delattr(neighbor, "cell") if hasattr(neighbor, "cell") else None),
            None,
        ),
    ],
)
def test_safer_cell_access_neighbor(
    grid_model, disable_memory, test_case, neighbor_setup, expected_position
):
    """Test neighbor location with various cell/pos configurations"""
    agents = LLMAgent.create_agents(grid_model, n=2, vision=-1, **DEFAULT_AGENT_CONFIG)
    agent, neighbor = agents
    agent.unique_id = 1
    neighbor.unique_id = 2

    agent.memory = ShortTermMemory(agent=agent, n=5, display=True)
    neighbor.memory = ShortTermMemory(agent=neighbor, n=5, display=True)

    grid_model.grid.place_agent(agent, (1, 1))

    # Setup neighbor's cell/pos state
    neighbor_setup(neighbor)

    disable_memory(agent)
    obs = agent.generate_obs()

    assert obs.local_state["LLMAgent 2"]["position"] == expected_position


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


def test_generate_obs_with_orthogonal_grid(basic_model, disable_memory):
    """Test neighbor detection with OrthogonalMooreGrid"""
    # Add OrthogonalMooreGrid to model
    basic_model.grid = OrthogonalMooreGrid(
        [3, 3], torus=False, random=basic_model.random
    )

    agents = LLMAgent.create_agents(basic_model, n=2, vision=1, **DEFAULT_AGENT_CONFIG)

    agent = agents[0]
    neighbor = agents[1]

    agent.unique_id = 1
    neighbor.unique_id = 2

    agent.memory = ShortTermMemory(agent=agent, n=5, display=True)
    neighbor.memory = ShortTermMemory(agent=neighbor, n=5, display=True)

    # Place agents in adjacent cells (discrete space uses cell.add_agent())
    cells = list(basic_model.grid.all_cells)
    cells[4].add_agent(agent)  # Center
    cells[5].add_agent(neighbor)  # Adjacent
    agent.cell = cells[4]
    neighbor.cell = cells[5]

    disable_memory(agent)
    obs = agent.generate_obs()

    # Should detect neighbor via cell.connections
    assert len(obs.local_state) == 1
    assert "LLMAgent 2" in obs.local_state
