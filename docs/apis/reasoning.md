# Reasoning System

The reasoning system in Mesa-LLM provides different cognitive strategies for agents to analyze situations, make decisions, and plan actions. It forms the core intelligence layer that transforms observations into actionable plans using structured thinking approaches. The reasoning module enables agents to process environmental observations and memory context into executable action plans through various cognitive frameworks.

---
### class Observation(step : int, self_state : dict, local_state : dict, environment_state : EnvironmentalState = None)
A structured snapshot containing the agent's current step, self-state (internal attributes, location, system context), local-state (neighboring agents), and optionally environmental-state (grid properties, objects, navigability).

**Attributes:**
- **step** (int) - Current simulation step number
- **self_state** (dict) - Agent's internal attributes, location, and system context
- **local_state** (dict) - Neighboring agents detected within vision radius
- **environment_state** (EnvironmentalState | None) - Detailed environmental perception data

---
### class EnvironmentalState()
Detailed perception data about the agent's surroundings and the global simulation state.

**Attributes:**
- **current_cell** (dict) - Properties, objects, and navigability of the agent's current location
- **visible_cells** (dict) - Mapping of position strings to data for cells within vision radius (distance, direction, bearing, etc.)
- **statistics** (dict) - Aggregated stats for numeric properties across all visible cells
- **global_environment** (dict) - Model-level global variables configured for perception

---
### class Plan(step : int, llm_plan : object, ttl : int = 1)
An LLM-generated plan containing the step number, complete LLM response with tool calls, and a time-to-live (TTL) indicating how many steps the plan remains valid. Plans encapsulate both reasoning content and executable actions.

**Attributes:**
- **step** (int) - Step number when plan was created
- **llm_plan** (object) - Complete LLM response object with tool calls
- **ttl** (int) - Time-to-live indicating validity duration

---
### class Reasoning(agent : LLMAgent)
Abstract base class providing the interface for all reasoning strategies, with both synchronous `plan()` and asynchronous `aplan()` methods for parallel execution scenarios.

**Attributes:**
- **agent** (LLMAgent reference)

**Methods:**
- **abstract plan(prompt, obs=None, ttl=1, selected_tools=None)** → *Plan* - Generate synchronous plan
- **async aplan(prompt, obs=None, ttl=1, selected_tools=None)** → *Plan* - Generate asynchronous plan

**Reasoning Flow:**
1. Agent generates **observation** of current situation through `generate_obs()`
2. Reasoning strategies access **memory** to inform decisions
3. Selected reasoning approach processes observation and memory into a structured **plan**
4. Plans are automatically converted to **tool schemas** for LLM function calling
5. Tool manager **executes the planned actions** in the simulation environment

## Built-in Reasoning Strategies

### class CoTReasoning(agent : LLMAgent)
Chain of Thought reasoning with explicit step-by-step analysis before action execution. Uses structured numbered thoughts followed by tool execution. Integrates memory context for informed decision-making.

**Attributes:**
- **agent** (LLMAgent reference)

**Methods:**
- **plan(prompt, obs=None, ttl=1, selected_tools=None)** → *Plan* - Generate synchronous plan with CoT reasoning
- **async aplan(prompt, obs=None, ttl=1, selected_tools=None)** → *Plan* - Generate asynchronous plan with CoT reasoning

**Reasoning Format:**
```
Thought 1: [Initial reasoning based on observation]
Thought 2: [How memory informs the situation]
Thought 3: [Possible alternatives or risks]
Thought 4: [Final decision and justification]
Action: [The action you decide to take]
```

---
### class ReActReasoning(agent : LLMAgent)
Reasoning + Acting with alternating reasoning and action in flexible conversational format. Combines thinking and acting in natural language flow. Less structured than CoT but incorporates memory and communication history.

**Attributes:**
- **agent** (LLMAgent reference)

**Methods:**
- **plan(prompt, obs=None, ttl=1, selected_tools=None)** → *Plan* - Generate synchronous plan with ReAct reasoning
- **async aplan(prompt, obs=None, ttl=1, selected_tools=None)** → *Plan* - Generate asynchronous plan with ReAct reasoning

---
### class ReWOOReasoning(agent : LLMAgent)
Reasoning Without Observation for multi-step planning without environmental feedback. Enables multi-step planning without requiring immediate environmental feedback. Plans remain valid across multiple simulation steps with extended TTL. Reduces computational overhead through strategic long-term thinking.

**Attributes:**
- **agent** (LLMAgent reference)
- **remaining_tool_calls** (int) - Number of tool calls remaining in current plan
- **current_plan** (Plan) - Currently active multi-step plan
- **current_obs** (Observation) - Last observation used for planning

**Methods:**
- **plan(prompt, obs=None, ttl=1, selected_tools=None)** → *Plan* - Generate synchronous plan with ReWOO reasoning
- **async aplan(prompt, obs=None, ttl=1, selected_tools=None)** → *Plan* - Generate asynchronous plan with ReWOO reasoning

## Usage in Mesa Simulations

```python
from mesa_llm.llm_agent import LLMAgent
from mesa_llm.reasoning.cot import CoTReasoning

class MyAgent(LLMAgent):
    def __init__(self, model, **kwargs):
        super().__init__(
            model=model,
            reasoning=CoTReasoning,  # Specify reasoning strategy
            **kwargs
        )

    def step(self):
        # Generate observation and create plan using reasoning strategy
        obs = self.generate_obs()
        plan = self.reasoning.plan(
            obs=obs,
            selected_tools=["move_one_step", "speak_to"]
        )
        self.apply_plan(plan)

# Strategy-specific configurations
from mesa_llm.reasoning.react import ReActReasoning
from mesa_llm.reasoning.rewoo import ReWOOReasoning

# For ReWOO with multi-step planning
plan = self.reasoning.plan(obs=obs, ttl=3)  # Plan valid for 3 steps

# Parallel reasoning execution
async def astep(self):
    obs = self.generate_obs()
    plan = await self.reasoning.aplan(
        prompt=self.step_prompt,
        obs=obs,
        selected_tools=["move_one_step", "arrest_citizen"]
    )
    self.apply_plan(plan)
```