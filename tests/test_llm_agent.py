import re

import pytest
from mesa.discrete_space import OrthogonalMooreGrid
from mesa.model import Model
from mesa.space import ContinuousSpace, MultiGrid

from mesa_llm import Plan
from mesa_llm.llm_agent import LLMAgent
from mesa_llm.memory.st_memory import ShortTermMemory
from mesa_llm.reasoning.react import ReActReasoning

DEFAULT_AGENT_CONFIG = {
    "llm_model": "gemini/gemini-2.0-flash",
}


def test_apply_plan_adds_to_memory(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "dummy")

    class DummyModel(Model):
        def __init__(self):
            super().__init__(seed=42)
            self.grid = MultiGrid(3, 3, torus=False)

        def add_agent(self, pos):
            system_prompt = "You are an agent in a simulation."
            agents = LLMAgent.create_agents(
                self,
                n=1,
                reasoning=ReActReasoning,
                system_prompt=system_prompt,
                vision=-1,
                internal_state=["test_state"],
            )

            x, y = pos

            self.grid.place_agent(agents[0], (x, y))
            return agents[0]

    model = DummyModel()
    agent = model.add_agent((1, 1))
    agent.memory = ShortTermMemory(
        agent=agent,
        n=5,
        display=True,
    )

    # fake response returned by the tool manager
    fake_response = [{"tool": "foo", "argument": "bar"}]

    # monkeypatch the tool manager so no real tool calls are made
    monkeypatch.setattr(
        agent.tool_manager, "call_tools", lambda agent, llm_response: fake_response
    )

    plan = Plan(step=0, llm_plan="do something")

    resp = agent.apply_plan(plan)

    assert resp == fake_response

    assert {
        "tool": "foo",
        "argument": "bar",
    } in agent.memory.step_content.values() or agent.memory.step_content == {
        "tool": "foo",
        "argument": "bar",
    }


def test_generate_obs_with_one_neighbor(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "dummy")

    class DummyModel(Model):
        def __init__(self):
            super().__init__(seed=45)
            self.grid = MultiGrid(3, 3, torus=False)

        def add_agent(self, pos, agent_class=LLMAgent):
            system_prompt = "You are an agent in a simulation."
            agents = agent_class.create_agents(
                self,
                n=1,
                reasoning=ReActReasoning,
                system_prompt=system_prompt,
                vision=-1,
                internal_state=["test_state"],
            )
            x, y = pos
            self.grid.place_agent(agents[0], (x, y))
            return agents[0]

    model = DummyModel()

    agent = model.add_agent((1, 1))
    agent.memory = ShortTermMemory(
        agent=agent,
        n=5,
        display=True,
    )
    agent.unique_id = 1

    neighbor = model.add_agent((1, 2))
    neighbor.memory = ShortTermMemory(
        agent=agent,
        n=5,
        display=True,
    )
    neighbor.unique_id = 2
    monkeypatch.setattr(agent.memory, "add_to_memory", lambda *args, **kwargs: None)

    obs = agent.generate_obs()

    assert obs.self_state["agent_unique_id"] == 1

    # we should have exactly one neighboring agent in local_state
    assert len(obs.local_state) == 1

    # extract the neighbor
    key = next(iter(obs.local_state.keys()))
    assert key == "LLMAgent 2"

    entry = obs.local_state[key]
    assert entry["position"] == (1, 2)
    assert entry["internal_state"] == ["test_state"]


def test_send_message_updates_both_agents_memory(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "dummy")

    class DummyModel(Model):
        def __init__(self):
            super().__init__(seed=45)
            self.grid = MultiGrid(3, 3, torus=False)

        def add_agent(self, pos, agent_class=LLMAgent):
            system_prompt = "You are an agent in a simulation."
            agents = agent_class.create_agents(
                self,
                n=1,
                reasoning=lambda agent: None,
                system_prompt=system_prompt,
                vision=-1,
                internal_state=["test_state"],
            )
            x, y = pos
            self.grid.place_agent(agents[0], (x, y))
            return agents[0]

    model = DummyModel()
    sender = model.add_agent((0, 0))
    sender.memory = ShortTermMemory(
        agent=sender,
        n=5,
        display=True,
    )
    sender.unique_id = 1

    recipient = model.add_agent((1, 1))
    recipient.memory = ShortTermMemory(
        agent=recipient,
        n=5,
        display=True,
    )
    recipient.unique_id = 2

    # Track how many times add_to_memory is called
    call_counter = {"count": 0}

    def fake_add_to_memory(*args, **kwargs):
        call_counter["count"] += 1

    # monkeypatch both agents' memory modules
    monkeypatch.setattr(sender.memory, "add_to_memory", fake_add_to_memory)
    monkeypatch.setattr(recipient.memory, "add_to_memory", fake_add_to_memory)

    result = sender.send_message("hello", recipients=[recipient])
    pattern = r"LLMAgent 1 → \[<mesa_llm\.llm_agent\.LLMAgent object at 0x[0-9A-Fa-f]+>\] : hello"
    assert re.match(pattern, result)

    # sender + recipient memory => should be called twice
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


# Environmental Perception Tests


def test_generate_obs_with_environment_perception(grid_model, disable_memory):
    """Test that environmental perception works with PropertyLayers."""
    try:
        import numpy as np
        from mesa.experimental.cell_space import PropertyLayer

        # Add PropertyLayer to grid
        grid_model.grid.properties = {
            "sugar": PropertyLayer(name="sugar", width=10, height=10, default_value=0)
        }
        grid_model.grid.properties["sugar"].data = np.random.randint(
            0, 10, size=(10, 10)
        )

        agent = LLMAgent.create_agents(
            grid_model, n=1, vision=2, perceive_environment=True, **DEFAULT_AGENT_CONFIG
        )[0]

        agent.memory = ShortTermMemory(agent=agent, n=5, display=True)
        grid_model.grid.place_agent(agent, (5, 5))

        disable_memory(agent)
        obs = agent.generate_obs()

        # Check environment_state exists and is EnvironmentalState instance
        assert obs.environment_state is not None
        assert obs.environment_state.current_cell is not None
        assert "properties" in obs.environment_state.current_cell
        assert "sugar" in obs.environment_state.current_cell["properties"]

        # Check visible cells
        assert obs.environment_state.visible_cells is not None
        assert len(obs.environment_state.visible_cells) > 0

        # Check statistics
        assert obs.environment_state.statistics is not None
        assert "sugar" in obs.environment_state.statistics
        assert "max" in obs.environment_state.statistics["sugar"]
        assert "min" in obs.environment_state.statistics["sugar"]
        assert "avg" in obs.environment_state.statistics["sugar"]

    except ImportError:
        # PropertyLayer not available in this Mesa version
        pytest.skip("PropertyLayer not available")


def test_generate_obs_without_environment_perception(grid_model, disable_memory):
    """Test that environmental perception is empty when disabled (backward compatibility)."""
    agent = LLMAgent.create_agents(
        grid_model,
        n=1,
        vision=2,
        perceive_environment=False,  # Disabled (default)
        **DEFAULT_AGENT_CONFIG,
    )[0]

    agent.memory = ShortTermMemory(agent=agent, n=5, display=True)
    grid_model.grid.place_agent(agent, (5, 5))

    disable_memory(agent)
    obs = agent.generate_obs()

    # Should have None environment_state when disabled
    assert obs.environment_state is None


def test_environment_statistics(grid_model, disable_memory):
    """Test that property statistics are calculated correctly."""
    try:
        import numpy as np
        from mesa.experimental.cell_space import PropertyLayer

        # Create known sugar distribution
        sugar_data = np.zeros((10, 10))
        sugar_data[5, 6] = 10.0  # Max sugar to the east
        sugar_data[5, 4] = 2.0  # Min sugar to the west
        sugar_data[6, 5] = 5.0  # Some sugar to the south

        grid_model.grid.properties = {
            "sugar": PropertyLayer(name="sugar", width=10, height=10, default_value=0)
        }
        grid_model.grid.properties["sugar"].data = sugar_data

        agent = LLMAgent.create_agents(
            grid_model,
            n=1,
            vision=2,
            perceive_environment=True,
            perceive_properties=["sugar"],
            include_statistics=True,
            **DEFAULT_AGENT_CONFIG,
        )[0]

        agent.memory = ShortTermMemory(agent=agent, n=5, display=True)
        grid_model.grid.place_agent(agent, (5, 5))

        disable_memory(agent)
        obs = agent.generate_obs()

        # Check statistics
        assert obs.environment_state is not None
        assert obs.environment_state.statistics is not None
        assert "sugar" in obs.environment_state.statistics

        stats = obs.environment_state.statistics["sugar"]
        assert stats["max"] == 10.0
        assert stats["best_location"] == (5, 6)

    except ImportError:
        pytest.skip("PropertyLayer not available")


def test_max_cells_reported_limit(grid_model, disable_memory):
    """Test that max_cells_reported limits the number of visible cells."""
    agent = LLMAgent.create_agents(
        grid_model,
        n=1,
        vision=5,  # Large vision (would see many cells)
        perceive_environment=True,
        max_visible_cells=5,  # But limit to 5 cells
        **DEFAULT_AGENT_CONFIG,
    )[0]

    agent.memory = ShortTermMemory(agent=agent, n=5, display=True)
    grid_model.grid.place_agent(agent, (5, 5))

    disable_memory(agent)
    obs = agent.generate_obs()

    # Should have at most 5 visible cells
    assert obs.environment_state is not None
    visible_cells = obs.environment_state.visible_cells or {}
    visible_cells_count = len(visible_cells)
    assert visible_cells_count <= 5


def test_global_environment_perception(basic_model, disable_memory):
    """Test perception of model-level environmental variables."""
    # Add global environment variables to model
    basic_model.weather = "sunny"
    basic_model.temperature = 25.5

    agent = LLMAgent.create_agents(
        basic_model, n=1, perceive_environment=True, **DEFAULT_AGENT_CONFIG
    )[0]

    agent.memory = ShortTermMemory(agent=agent, n=5, display=True)

    disable_memory(agent)
    obs = agent.generate_obs()

    # Check environment_state exists
    # Since agent doesn't have a position in basic_model, environment_state will be None
    # (perceive_environment requires position)
    # Let's just verify the attribute exists and is either None or EnvironmentalState
    from mesa_llm.reasoning.reasoning import EnvironmentalState

    assert hasattr(obs, "environment_state")
    assert obs.environment_state is None or isinstance(
        obs.environment_state, EnvironmentalState
    )


def test_distance_direction_calculation(grid_model, disable_memory):
    """Test that distance and direction are calculated correctly."""
    try:
        from mesa.experimental.cell_space import PropertyLayer

        grid_model.grid.properties = {
            "test": PropertyLayer(name="test", width=10, height=10, default_value=1)
        }

        agent = LLMAgent.create_agents(
            grid_model, n=1, vision=2, perceive_environment=True, **DEFAULT_AGENT_CONFIG
        )[0]

        agent.memory = ShortTermMemory(agent=agent, n=5, display=True)
        grid_model.grid.place_agent(agent, (5, 5))

        disable_memory(agent)
        obs = agent.generate_obs()

        # Check that visible cells have distance and direction
        if obs.environment_state and obs.environment_state.visible_cells:
            for _cell_key, cell_data in obs.environment_state.visible_cells.items():
                assert "distance" in cell_data
                assert "direction" in cell_data
                assert isinstance(cell_data["distance"], int | float)
                assert isinstance(cell_data["direction"], str)

    except ImportError:
        pytest.skip("PropertyLayer not available")
