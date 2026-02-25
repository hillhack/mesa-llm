from mesa import Agent, Model
from mesa.space import MultiGrid
from mesa.time import RandomActivation

from mesa_llm.llm_agent import LLMAgent
from mesa_llm.reasoning.react import ReActReasoning


class Fruit(Agent):
    """A pickable resource."""

    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.type = "fruit"
        self.pickable = True
        self.nutrition = 10


class Boulder(Agent):
    """An obstacle that blocks movement."""

    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.type = "boulder"
        self.blocks_movement = True
        self.pickable = False


class SmartForagerAgent(LLMAgent):
    """An LLM-powered agent that navigates and forages."""

    def __init__(self, unique_id, model, **kwargs):
        super().__init__(
            model=model,
            reasoning=ReActReasoning,
            vision=3,
            perceive_objects="all",
            perceive_environment=True,
            **kwargs,
        )
        self.energy = 50

    def step(self):
        self.energy -= 1  # Base metabolism

        # Generate observation including new perception data
        obs = self.generate_obs()

        # Prepare the step prompt with energy status
        prompt = f"Your current energy is {self.energy}. Find food and avoid obstacles."

        # Reason and plan
        plan = self.reasoning.plan(prompt=prompt, obs=obs)

        # Execute actions
        self.apply_plan(plan)

        # Print status for demonstration
        print(f"Agent {self.unique_id} @ {self.pos} | Energy: {self.energy}")

    def eat(self):
        """Action for eating fruit at the current location."""
        cell_agents = self.model.grid.get_cell_list_contents([self.pos])
        for obj in cell_agents:
            if isinstance(obj, Fruit):
                self.energy += obj.nutrition
                self.model.grid.remove_agent(obj)
                self.model.schedule.remove(obj)
                print(f"Agent {self.unique_id} ate a fruit!")
                return "Ate fruit"
        return "No fruit here"


class SmartForagerModel(Model):
    """A survival model with resources and obstacles."""

    def __init__(self, width=10, height=10, num_agents=1, num_fruit=10, num_boulders=5):
        super().__init__()
        self.grid = MultiGrid(width, height, torus=False)
        self.schedule = RandomActivation(self)

        # Add Boulders (Obstacles)
        for _ in range(num_boulders):
            b = Boulder(self.next_id(), self)
            pos = self.grid.find_empty()
            if pos:
                self.grid.place_agent(b, pos)

        # Add Fruit (Resources)
        for _ in range(num_fruit):
            f = Fruit(self.next_id(), self)
            pos = self.grid.find_empty()
            if pos:
                self.grid.place_agent(f, pos)
                self.schedule.add(f)

        # Add Forager Agent
        for _ in range(num_agents):
            a = SmartForagerAgent(self.next_id(), self)
            pos = self.grid.find_empty()
            if pos:
                self.grid.place_agent(a, pos)
                self.schedule.add(a)

    def step(self):
        self.schedule.step()


if __name__ == "__main__":
    # Note: Requires MESA_LLM_API_KEY environment variable if not using a local LLM
    model = SmartForagerModel()
    print("Starting Smart Forager Simulation...")
    for i in range(5):
        print(f"\n--- Step {i} ---")
        model.step()
