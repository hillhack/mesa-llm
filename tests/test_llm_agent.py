# tests/test_llm_agent.py

import re

from mesa.model import Model
from mesa.space import MultiGrid

from mesa_llm import Plan
from mesa_llm.llm_agent import LLMAgent
from mesa_llm.memory.st_memory import ShortTermMemory
from mesa_llm.reasoning.react import ReActReasoning


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


def test_safer_cell_access_agent_with_cell_no_pos(monkeypatch):
    """Test safer cell access: agent with cell but pos=None uses cell.coordinate"""
    monkeypatch.setenv("GEMINI_API_KEY", "dummy")

    class DummyModel(Model):
        def __init__(self):
            super().__init__(seed=42)

    class MockCell:
        """Mock cell with coordinate attribute"""

        def __init__(self, coordinate):
            self.coordinate = coordinate

    model = DummyModel()
    agents = LLMAgent.create_agents(
        model,
        n=1,
        reasoning=ReActReasoning,
        system_prompt="Test",
        vision=0,
        internal_state=["test"],
    )
    agent = agents[0]
    agent.memory = ShortTermMemory(agent=agent, n=5, display=True)

    # Set pos=None and add cell
    agent.pos = None
    agent.cell = MockCell(coordinate=(3, 4))

    monkeypatch.setattr(agent.memory, "add_to_memory", lambda *args, **kwargs: None)
    obs = agent.generate_obs()

    # Should use cell.coordinate
    assert obs.self_state["location"] == (3, 4)


def test_safer_cell_access_agent_without_cell_or_pos(monkeypatch):
    """Test safer cell access: agent without pos or cell returns None"""
    monkeypatch.setenv("GEMINI_API_KEY", "dummy")

    class DummyModel(Model):
        def __init__(self):
            super().__init__(seed=42)

    model = DummyModel()
    agents = LLMAgent.create_agents(
        model,
        n=1,
        reasoning=ReActReasoning,
        system_prompt="Test",
        vision=0,
        internal_state=["test"],
    )
    agent = agents[0]
    agent.memory = ShortTermMemory(agent=agent, n=5, display=True)

    # Ensure no pos or cell
    agent.pos = None
    if hasattr(agent, "cell"):
        delattr(agent, "cell")

    monkeypatch.setattr(agent.memory, "add_to_memory", lambda *args, **kwargs: None)
    obs = agent.generate_obs()

    # Should return None gracefully
    assert obs.self_state["location"] is None


def test_safer_cell_access_neighbor_with_cell_no_pos(monkeypatch):
    """Test safer cell access: neighbor with cell but pos=None uses cell.coordinate"""
    monkeypatch.setenv("GEMINI_API_KEY", "dummy")

    class DummyModel(Model):
        def __init__(self):
            super().__init__(seed=42)
            self.grid = MultiGrid(3, 3, torus=False)

    class MockCell:
        """Mock cell with coordinate attribute"""

        def __init__(self, coordinate):
            self.coordinate = coordinate

    model = DummyModel()
    agents = LLMAgent.create_agents(
        model,
        n=2,
        reasoning=ReActReasoning,
        system_prompt="Test",
        vision=-1,
        internal_state=["test"],
    )

    agent = agents[0]
    neighbor = agents[1]
    agent.unique_id = 1
    neighbor.unique_id = 2

    agent.memory = ShortTermMemory(agent=agent, n=5, display=True)
    neighbor.memory = ShortTermMemory(agent=neighbor, n=5, display=True)

    model.grid.place_agent(agent, (1, 1))

    # Neighbor has cell but no pos
    neighbor.pos = None
    neighbor.cell = MockCell(coordinate=(2, 2))

    monkeypatch.setattr(agent.memory, "add_to_memory", lambda *args, **kwargs: None)
    obs = agent.generate_obs()

    # Should use neighbor's cell.coordinate
    assert obs.local_state["LLMAgent 2"]["position"] == (2, 2)


def test_safer_cell_access_neighbor_without_cell_or_pos(monkeypatch):
    """Test safer cell access: neighbor without pos or cell returns None"""
    monkeypatch.setenv("GEMINI_API_KEY", "dummy")

    class DummyModel(Model):
        def __init__(self):
            super().__init__(seed=42)
            self.grid = MultiGrid(3, 3, torus=False)

    model = DummyModel()
    agents = LLMAgent.create_agents(
        model,
        n=2,
        reasoning=ReActReasoning,
        system_prompt="Test",
        vision=-1,
        internal_state=["test"],
    )

    agent = agents[0]
    neighbor = agents[1]
    agent.unique_id = 1
    neighbor.unique_id = 2

    agent.memory = ShortTermMemory(agent=agent, n=5, display=True)
    neighbor.memory = ShortTermMemory(agent=neighbor, n=5, display=True)

    model.grid.place_agent(agent, (1, 1))

    # Neighbor has no pos or cell
    neighbor.pos = None
    if hasattr(neighbor, "cell"):
        delattr(neighbor, "cell")

    monkeypatch.setattr(agent.memory, "add_to_memory", lambda *args, **kwargs: None)
    obs = agent.generate_obs()

    # Should return None gracefully
    assert obs.local_state["LLMAgent 2"]["position"] is None


def test_generate_obs_with_continuous_space(monkeypatch):
    """Test neighbor detection with ContinuousSpace"""
    monkeypatch.setenv("GEMINI_API_KEY", "dummy")

    from mesa.space import ContinuousSpace

    class ContinuousModel(Model):
        def __init__(self):
            super().__init__(seed=42)
            self.space = ContinuousSpace(x_max=10.0, y_max=10.0, torus=False)

    model = ContinuousModel()

    # Create agents
    agents = LLMAgent.create_agents(
        model,
        n=3,
        reasoning=ReActReasoning,
        system_prompt="Test",
        vision=2.0,  # Radius
        internal_state=["test"],
    )

    agent = agents[0]
    nearby_neighbor = agents[1]
    far_neighbor = agents[2]

    agent.unique_id = 1
    nearby_neighbor.unique_id = 2
    far_neighbor.unique_id = 3

    # Set up memory
    for a in agents:
        a.memory = ShortTermMemory(agent=a, n=5, display=True)

    # Place agents
    model.space.place_agent(agent, (5.0, 5.0))
    model.space.place_agent(nearby_neighbor, (6.0, 5.0))  # Distance ~1.0
    model.space.place_agent(far_neighbor, (9.0, 9.0))  # Distance ~5.66

    monkeypatch.setattr(agent.memory, "add_to_memory", lambda *args, **kwargs: None)
    obs = agent.generate_obs()

    # Should see nearby neighbor but not far one
    assert len(obs.local_state) == 1
    assert "LLMAgent 2" in obs.local_state
    assert "LLMAgent 3" not in obs.local_state


def test_generate_obs_vision_all_agents(monkeypatch):
    """Test vision=-1 to see all agents in simulation"""
    monkeypatch.setenv("GEMINI_API_KEY", "dummy")

    class DummyModel(Model):
        def __init__(self):
            super().__init__(seed=42)
            self.grid = MultiGrid(10, 10, torus=False)

    model = DummyModel()

    # Create multiple agents
    agents = LLMAgent.create_agents(
        model,
        n=4,
        reasoning=ReActReasoning,
        system_prompt="Test",
        vision=-1,  # See ALL agents
        internal_state=["test"],
    )

    for idx, a in enumerate(agents):
        a.unique_id = idx + 1
        a.memory = ShortTermMemory(agent=a, n=5, display=True)
        model.grid.place_agent(a, (idx, idx))

    agent = agents[0]
    monkeypatch.setattr(agent.memory, "add_to_memory", lambda *args, **kwargs: None)
    obs = agent.generate_obs()

    # Should see ALL other 3 agents (not self)
    assert len(obs.local_state) == 3
    assert "LLMAgent 2" in obs.local_state
    assert "LLMAgent 3" in obs.local_state
    assert "LLMAgent 4" in obs.local_state


def test_generate_obs_no_grid_with_vision(monkeypatch):
    """Test fallback when model has no grid/space but vision > 0"""
    monkeypatch.setenv("GEMINI_API_KEY", "dummy")

    class NoGridModel(Model):
        def __init__(self):
            super().__init__(seed=42)
            # No grid or space defined

    model = NoGridModel()

    agents = LLMAgent.create_agents(
        model,
        n=2,
        reasoning=ReActReasoning,
        system_prompt="Test",
        vision=5,  # Has vision but no grid to use it with
        internal_state=["test"],
    )

    agent = agents[0]
    agent.unique_id = 1
    agent.memory = ShortTermMemory(agent=agent, n=5, display=True)

    monkeypatch.setattr(agent.memory, "add_to_memory", lambda *args, **kwargs: None)

    # Should not crash, should return empty neighbors
    obs = agent.generate_obs()

    assert len(obs.local_state) == 0  # No neighbors found (fallback path)


def test_generate_obs_with_orthogonal_grid(monkeypatch):
    """Test neighbor detection with OrthogonalMooreGrid"""
    monkeypatch.setenv("GEMINI_API_KEY", "dummy")

    from mesa.discrete_space import OrthogonalMooreGrid

    class OrthogonalModel(Model):
        def __init__(self):
            super().__init__(seed=42)
            self.grid = OrthogonalMooreGrid([3, 3], torus=False)

    model = OrthogonalModel()

    agents = LLMAgent.create_agents(
        model,
        n=2,
        reasoning=ReActReasoning,
        system_prompt="Test",
        vision=1,
        internal_state=["test"],
    )

    agent = agents[0]
    neighbor = agents[1]

    agent.unique_id = 1
    neighbor.unique_id = 2

    agent.memory = ShortTermMemory(agent=agent, n=5, display=True)
    neighbor.memory = ShortTermMemory(agent=neighbor, n=5, display=True)

    # Place agents in adjacent cells
    model.grid.place_agent(agent, model.grid.all_cells[4])  # Center
    model.grid.place_agent(neighbor, model.grid.all_cells[5])  # Adjacent

    monkeypatch.setattr(agent.memory, "add_to_memory", lambda *args, **kwargs: None)
    obs = agent.generate_obs()

    # Should detect neighbor via cell.connections
    assert len(obs.local_state) == 1
    assert "LLMAgent 2" in obs.local_state
