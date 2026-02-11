from mesa.agent import Agent
from mesa.model import Model
from mesa.space import (
    ContinuousSpace,
)

from mesa_llm import Plan
from mesa_llm.memory.st_lt_memory import STLTMemory
from mesa_llm.module_llm import ModuleLLM
from mesa_llm.reasoning.reasoning import (
    EnvironmentalState,
    Observation,
    Reasoning,
)
from mesa_llm.tools.tool_manager import ToolManager


class LLMAgent(Agent):
    """
    LLMAgent manages an LLM backend and optionally connects to a memory module.

    Parameters:
        model (Model): The mesa model the agent in linked to.
        llm_model (str): The model to use for the LLM in the format 'provider/model'. Defaults to 'gemini/gemini-2.0-flash'.
        system_prompt (str | None): Optional system prompt to be used in LLM completions.
        reasoning (str): Optional reasoning method to be used in LLM completions.

    Attributes:
        llm (ModuleLLM): The internal LLM interface used by the agent.
        memory (Memory | None): The memory module attached to this agent, if any.

    """

    def __init__(
        self,
        model: Model,
        reasoning: type[Reasoning],
        llm_model: str = "gemini/gemini-2.0-flash",
        system_prompt: str | None = None,
        vision: float | None = None,
        internal_state: list[str] | str | None = None,
        step_prompt: str | None = None,
        # Environmental perception parameters
        perceive_environment: bool = True,
        perceive_objects: bool | str = False,  # False, "all", "pickable", "static"
        max_visible_cells: int = 10,
        global_env_attributes: list[str] | None = None,
    ):
        super().__init__(model=model)

        self.model = model
        self.step_prompt = step_prompt
        self.llm = ModuleLLM(llm_model=llm_model, system_prompt=system_prompt)

        self.memory = STLTMemory(
            agent=self,
            short_term_capacity=5,
            consolidation_capacity=2,
            llm_model=llm_model,
        )

        self.tool_manager = ToolManager()
        self.vision = vision
        self.reasoning = reasoning(agent=self)
        self.system_prompt = system_prompt
        self.is_speaking = False
        self._current_plan = None  # Store current plan for formatting

        # display coordination
        self._step_display_data = {}

        if isinstance(internal_state, str):
            internal_state = [internal_state]
        elif internal_state is None:
            internal_state = []

        self.internal_state = internal_state

        # Environmental perception configuration
        self.perceive_environment = perceive_environment
        self.perceive_objects = perceive_objects
        self.max_visible_cells = max_visible_cells
        self.global_env_attributes = global_env_attributes or [
            "weather",
            "season",
            "time_of_day",
            "temperature",
            "day",
            "hour",
            "market_price",
            "global_event",
        ]

    def __str__(self):
        return f"LLMAgent {self.unique_id}"

    def apply_plan(self, plan: Plan) -> list[dict]:
        """
        Execute the plan in the simulation.
        """
        # Store current plan for display
        self._current_plan = plan

        # Execute tool calls
        tool_call_resp = self.tool_manager.call_tools(
            agent=self, llm_response=plan.llm_plan
        )

        # Add to memory
        self.memory.add_to_memory(
            type="action",
            content={
                k: v
                for tool_call in tool_call_resp
                for k, v in tool_call.items()
                if k not in ["tool_call_id", "role"]
            },
        )

        return tool_call_resp

    def generate_obs(self) -> Observation:
        """
        Generate comprehensive observation including environmental data.

        Returns an Observation with:
        - step: Current simulation step
        - self_state: Agent's own state
        - local_state: Nearby agents
        - environmental_state: Environmental perception (if enabled)
        """
        step = self.model.steps

        # Get self state
        self_state = {
            "agent_unique_id": self.unique_id,
            "system_prompt": self.system_prompt,
            "location": self._get_agent_location(),
            "internal_state": self.internal_state,
        }

        # Get local agents (neighbors)
        local_state = self._get_local_state()

        # Get environmental state
        environment_state = None
        if self.perceive_environment:
            environment_state = self._get_environmental_state()

        # Add to memory
        self.memory.add_to_memory(
            type="observation",
            content={
                "self_state": self_state,
                "local_state": local_state,
                "environment_state": environment_state,
            },
        )

        return Observation(
            step=step,
            self_state=self_state,
            local_state=local_state,
            environment_state=environment_state,
        )

    def _get_agent_location(self) -> tuple | None:
        """Get agent's current position."""
        if self.pos is not None:
            return self.pos
        if hasattr(self, "cell") and self.cell is not None:
            return self.cell.coordinate
        return None

    def _get_location_of(self, agent) -> tuple | None:
        """Get location of another agent."""
        if hasattr(agent, "pos") and agent.pos is not None:
            return agent.pos
        if hasattr(agent, "cell") and agent.cell is not None:
            return agent.cell.coordinate
        return None

    def _get_local_state(self) -> dict:
        """Get neighboring agents within vision."""
        neighbors = self._get_neighbors()
        local_state = {}

        for agent in neighbors:
            key = f"{type(agent).__name__} {agent.unique_id}"
            local_state[key] = {
                "position": self._get_location_of(agent),
                "internal_state": [
                    s
                    for s in getattr(agent, "internal_state", [])
                    if not str(s).startswith("_")
                ],
            }

        return local_state

    def _get_neighbors(self) -> list:
        """Get agents within vision radius."""
        if self.vision is None or self.vision == 0:
            return []

        # Vision = -1 means all agents
        if self.vision == -1:
            return [a for a in self.model.agents if a is not self]

        pos = self._get_agent_location()
        if pos is None:
            return []

        # Get from grid/space based on type
        grid = getattr(self.model, "grid", None)
        space = getattr(self.model, "space", None)

        # Handle OrthogonalGrid (uses cells not positions)
        if hasattr(self, "cell") and self.cell and hasattr(self.cell, "neighborhood"):
            # Get neighbors from cell connections using BFS
            neighbors = []
            radius = int(self.vision)

            # BFS to get all cells within radius
            visited = {self.cell}
            queue = [(self.cell, 0)]

            while queue:
                current_cell, dist = queue.pop(0)
                # Look at neighbors if within radius
                if dist < radius:
                    for (
                        neighbor_cell
                    ) in current_cell.neighborhood:  # It's a property, not a method!
                        if neighbor_cell not in visited:
                            visited.add(neighbor_cell)
                            queue.append((neighbor_cell, dist + 1))

                            # Add agents from discovered neighbor cell
                            if hasattr(neighbor_cell, "agents"):
                                for agent in neighbor_cell.agents:
                                    if agent is not self:
                                        neighbors.append(agent)

            return neighbors

        # Handle standard grids with get_neighbors
        if grid and hasattr(grid, "get_neighbors"):
            try:
                return grid.get_neighbors(
                    pos, radius=int(self.vision), include_center=False
                )
            except:
                pass

        # Handle continuous space
        if space and isinstance(space, ContinuousSpace):
            try:
                return space.get_neighbors(
                    pos, radius=self.vision, include_center=False
                )
            except:
                pass

        return []

    def _get_environmental_state(self) -> EnvironmentalState | None:
        """Collect all environmental data."""
        pos = self._get_agent_location()
        if pos is None:
            return None

        current_cell = self._get_current_cell_data(pos)
        visible_cells = self._get_visible_cells(pos)
        statistics = self._calculate_statistics(current_cell, visible_cells)
        global_env = self._get_global_environment()

        return EnvironmentalState(
            current_cell=current_cell,
            visible_cells=visible_cells,
            statistics=statistics,
            global_environment=global_env,
        )

    def _get_current_cell_data(self, pos: tuple) -> dict:
        """Get data about current cell."""
        return {
            "position": str(pos),
            "properties": self._get_cell_properties(pos),
            "objects": self._get_cell_objects(pos) if self.perceive_objects else [],
            "navigable": self._is_cell_navigable(pos),
            "agent_count": 1,  # Self
        }

    def _get_visible_cells(self, center_pos: tuple) -> dict[str, dict]:
        """Get data about cells within vision."""
        visible = {}

        if self.vision is None or self.vision <= 0:
            return visible

        # Get cells in radius
        cells = self._get_cells_in_radius(center_pos, self.vision)

        # Limit and process
        for i, cell_pos in enumerate(cells):
            if i >= self.max_visible_cells:
                break
            if cell_pos == center_pos:
                continue

            distance, direction, bearing = self._calculate_spatial_info(
                center_pos, cell_pos
            )

            visible[str(cell_pos)] = {
                "distance": round(distance, 2),
                "direction": direction,
                "relative_bearing": round(bearing, 1),
                "properties": self._get_cell_properties(cell_pos),
                "objects": self._get_cell_objects(cell_pos)
                if self.perceive_objects
                else [],
                "navigable": self._is_cell_navigable(cell_pos),
                "agent_count": self._count_agents_at(cell_pos),
            }

        return visible

    def _get_cells_in_radius(self, center: tuple, radius: float) -> list[tuple]:
        """Get all cell positions within radius."""
        cells = []
        r = int(radius)

        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                if dx == 0 and dy == 0:
                    continue
                if (dx**2 + dy**2) ** 0.5 <= radius:
                    cells.append((center[0] + dx, center[1] + dy))

        return cells

    def _get_cell_properties(self, pos: tuple) -> dict:
        """Get PropertyLayer values at position."""
        properties = {}

        grid = getattr(self.model, "grid", None)
        if not grid:
            return properties

        # Try modern API first (Mesa's standard public API)
        if hasattr(grid, "properties"):
            for name, layer in grid.properties.items():
                if name == "empty":
                    continue
                try:
                    if hasattr(layer, "data"):
                        # PropertyLayer with data attribute
                        value = layer.data[pos]
                    else:
                        # Direct dict-like access
                        value = layer[pos]
                    properties[name] = self._convert_property_value(value)
                except (IndexError, KeyError, TypeError):
                    properties[name] = None

        return properties

    def _convert_property_value(self, value):
        """Convert numpy types to Python types."""
        try:
            import numpy as np

            if isinstance(value, (np.integer, np.floating)):
                return float(value)
            if isinstance(value, np.bool_):
                return bool(value)
        except ImportError:
            pass
        return value

    def _get_cell_objects(self, pos: tuple) -> list[dict]:
        """Detect non-agent objects at position."""
        objects = []

        # Try model.objects dict
        if hasattr(self.model, "objects"):
            model_objects = self.model.objects
            if isinstance(model_objects, dict):
                objs_at_pos = model_objects.get(pos, [])
                for obj in objs_at_pos:
                    obj_data = self._object_to_dict(obj)
                    if self._should_include_object(obj_data):
                        objects.append(obj_data)

        # Try cell.agents (filter out Agents)
        cell = self._get_cell_at(pos)
        if cell and hasattr(cell, "agents"):
            for entity in cell.agents:
                # Skip actual agents
                if not isinstance(entity, Agent):
                    obj_data = self._object_to_dict(entity)
                    if self._should_include_object(obj_data):
                        objects.append(obj_data)

        return objects

    def _get_cell_at(self, pos: tuple):
        """Get cell object at position."""
        grid = getattr(self.model, "grid", None)
        if grid and hasattr(grid, "get_cell"):
            try:
                return grid.get_cell(pos)
            except:
                pass
        return None

    def _object_to_dict(self, obj) -> dict:
        """Convert object to dictionary."""
        if hasattr(obj, "to_dict"):
            return obj.to_dict()

        obj_dict = {"type": getattr(obj, "type", type(obj).__name__)}

        # Common object attributes
        for attr in [
            "pickable",
            "blocks_movement",
            "state",
            "id",
            "value",
            "nutrition",
        ]:
            if hasattr(obj, attr):
                obj_dict[attr] = getattr(obj, attr)

        return obj_dict

    def _should_include_object(self, obj_data: dict) -> bool:
        """Filter objects based on config."""
        if self.perceive_objects is False:
            return False
        if self.perceive_objects == "all":
            return True
        if self.perceive_objects == "pickable":
            return obj_data.get("pickable", False)
        if self.perceive_objects == "static":
            return not obj_data.get("pickable", False)
        return True

    def _is_cell_navigable(self, pos: tuple) -> bool:
        """Check if cell can be moved to."""
        grid = getattr(self.model, "grid", None)

        # Check walkable property layer
        if grid and hasattr(grid, "properties"):
            if "walkable" in grid.properties:
                try:
                    walkable_layer = grid.properties["walkable"]
                    if hasattr(walkable_layer, "data"):
                        return bool(walkable_layer.data[pos])
                    else:
                        return bool(walkable_layer[pos])
                except (IndexError, KeyError):
                    pass

        # Check cell is not blocked
        cell = self._get_cell_at(pos)
        if cell and hasattr(cell, "blocked"):
            return not cell.blocked

        # Default: assume navigable
        return True

    def _count_agents_at(self, pos: tuple) -> int:
        """Count agents at position."""
        count = 0
        for agent in self.model.agents:
            agent_pos = self._get_location_of(agent)
            if agent_pos == pos:
                count += 1
        return count

    def _calculate_spatial_info(self, from_pos: tuple, to_pos: tuple) -> tuple:
        """Calculate distance, direction, and bearing."""
        import math

        dx = to_pos[0] - from_pos[0]
        dy = to_pos[1] - from_pos[1]

        distance = math.sqrt(dx**2 + dy**2)

        # Cardinal direction
        direction = self._get_cardinal_direction(dx, dy)

        # Bearing (0=North, 90=East, 180=South, 270=West)
        bearing = (math.degrees(math.atan2(dx, -dy)) + 360) % 360

        return distance, direction, bearing

    def _get_cardinal_direction(self, dx: int, dy: int) -> str:
        """Convert delta to direction."""
        if dx == 0 and dy == 0:
            return "here"

        # Normalize to -1, 0, 1
        nx = 0 if dx == 0 else (1 if dx > 0 else -1)
        ny = 0 if dy == 0 else (1 if dy > 0 else -1)

        directions = {
            (0, -1): "north",
            (1, -1): "northeast",
            (1, 0): "east",
            (1, 1): "southeast",
            (0, 1): "south",
            (-1, 1): "southwest",
            (-1, 0): "west",
            (-1, -1): "northwest",
        }

        return directions.get((nx, ny), "unknown")

    def _calculate_statistics(self, current_cell: dict, visible_cells: dict) -> dict:
        """Calculate aggregate statistics."""
        stats = {}

        # Collect numeric properties
        all_props = {}

        # Add current cell properties
        for prop, value in current_cell.get("properties", {}).items():
            if isinstance(value, (int, float)) and value is not None:
                all_props.setdefault(prop, []).append(value)

        # Add visible cell properties
        for cell_data in visible_cells.values():
            for prop, value in cell_data.get("properties", {}).items():
                if isinstance(value, (int, float)) and value is not None:
                    all_props.setdefault(prop, []).append(value)

        # Calculate stats for each property
        for prop, values in all_props.items():
            if values:
                stats[prop] = {
                    "max": max(values),
                    "min": min(values),
                    "avg": round(sum(values) / len(values), 2),
                    "count": len(values),
                }

        # Agent count statistics
        total_agents = current_cell.get("agent_count", 0)
        for cell_data in visible_cells.values():
            total_agents += cell_data.get("agent_count", 0)
        stats["agent_count"] = {"total_visible": total_agents}

        return stats

    def _get_global_environment(self) -> dict:
        """Get model-level environment variables."""
        global_env = {}

        # Try environment object
        if hasattr(self.model, "environment"):
            env = self.model.environment
            if hasattr(env, "to_dict"):
                global_env.update(env.to_dict())
            elif isinstance(env, dict):
                global_env.update(env)
            else:
                # Try to extract attributes
                for attr in dir(env):
                    if not attr.startswith("_"):
                        try:
                            global_env[attr] = getattr(env, attr)
                        except:
                            pass

        # Try individual attributes
        for attr in self.global_env_attributes:
            if hasattr(self.model, attr) and attr not in global_env:
                global_env[attr] = getattr(self.model, attr)

        return global_env

    def send_message(self, message: str, recipients: list[Agent]) -> str:
        """
        Send a message to the recipients.
        """
        for recipient in [*recipients, self]:
            recipient.memory.add_to_memory(
                type="message",
                content={
                    "message": message,
                    "sender": self,
                    "recipients": recipients,
                },
            )

        return f"{self} → {recipients} : {message}"

    def pre_step(self):
        """
        This is some code that is executed before the step method of the child agent is called.
        """
        self.memory.process_step(pre_step=True)

    def post_step(self):
        """
        This is some code that is executed after the step method of the child agent is called.
        It functions because of the __init_subclass__ method that creates a wrapper around the step method of the child agent.
        """
        self.memory.process_step()

    async def astep(self):
        """
        Default asynchronous step method for parallel agent execution.
        Subclasses should override this method for custom async behavior.
        If not overridden, falls back to calling the synchronous step() method.
        """
        self.pre_step()

        if hasattr(self, "step") and self.__class__.step != LLMAgent.step:
            self.step()

        self.post_step()

    def __init_subclass__(cls, **kwargs):
        """
        Wrapper - allows to automatically integrate code to be executed after the step method of the child agent (created by the user) is called.
        """
        super().__init_subclass__(**kwargs)
        # only wrap if subclass actually defines its own step
        user_step = cls.__dict__.get("step")
        user_astep = cls.__dict__.get("astep")

        if user_step:

            def wrapped(self, *args, **kwargs):
                """
                This is the wrapper that is used to integrate the pre_step and post_step methods into the step method of the child agent.
                """
                LLMAgent.pre_step(self, *args, **kwargs)
                result = user_step(self, *args, **kwargs)
                LLMAgent.post_step(self, *args, **kwargs)
                return result

            cls.step = wrapped

        if user_astep:

            async def awrapped(self, *args, **kwargs):
                """
                Async wrapper for astep method.
                """
                self.pre_step()
                result = await user_astep(self, *args, **kwargs)
                self.post_step()
                return result

            cls.astep = awrapped
