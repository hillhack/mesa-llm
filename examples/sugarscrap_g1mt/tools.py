from typing import TYPE_CHECKING

from examples.sugarscrap_g1mt.agents import (
    Resource,
    Trader,
    trader_tool_manager,
)
from mesa_llm.tools.tool_decorator import tool

if TYPE_CHECKING:
    from mesa_llm.llm_agent import LLMAgent

if TYPE_CHECKING:
    from mesa_llm.llm_agent import LLMAgent


@tool(tool_manager=trader_tool_manager)
def move_to_best_resource(agent: "LLMAgent") -> str:
    """
    Move the agent to the best resource cell within its vision range.

        Args:
            agent: Provided automatically

        Returns:
            A string confirming the new position of the agent.
    """

    best_cell = None
    best_amount = -1

    x, y = agent.pos
    vision = agent.vision

    for dx in range(-vision, vision + 1):
        for dy in range(-vision, vision + 1):
            nx, ny = x + dx, y + dy
            if not agent.model.grid.out_of_bounds((nx, ny)):
                cell_contents = agent.model.grid.get_cell_list_contents((nx, ny))
                for obj in cell_contents:
                    if isinstance(obj, Resource) and obj.current_amount > best_amount:
                        best_amount = obj.current_amount
                        best_cell = (nx, ny)

    if best_cell:
        agent.model.grid.move_agent(agent, best_cell)
        return f"agent {agent.unique_id} moved to {best_cell}."
    else:
        return f"agent {agent.unique_id} found no resources to move to."


@tool(tool_manager=trader_tool_manager)
def propose_trade(
    agent: "LLMAgent", other_agent_id: int, sugar_amount: int, spice_amount: int
) -> str:
    """
    Propose a trade to another agent.

        Args:
            other_agent_id: The unique id of the other agent to trade with.
            sugar_amount: The amount of sugar to offer.
            spice_amount: The amount of spice to offer.
            agent: Provided automatically

        Returns:
            A string confirming the trade proposal.
    """
    other_agent = next(
        (a for a in agent.model.agents if a.unique_id == other_agent_id), None
    )

    if other_agent is None:
        return f"Agent {other_agent} not found."

    if not isinstance(other_agent, Trader):
        return f"agent {other_agent_id} is not a valid trader."

    # Simple trade acceptance logic for demonstration
    if other_agent.calculate_mrs() > agent.calculate_mrs():
        agent.sugar -= sugar_amount
        agent.spice += spice_amount
        other_agent.sugar += sugar_amount
        other_agent.spice -= spice_amount
        return f"agent {agent.unique_id} traded {sugar_amount} sugar for {spice_amount} spice with agent {other_agent_id}."
    else:
        return f"agent {other_agent_id} rejected the trade proposal from agent {agent.unique_id}."
