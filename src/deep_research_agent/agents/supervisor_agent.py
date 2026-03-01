"""Multi-agent supervisor for coordinating research across multiple specialized agents.

This module implements a supervisor pattern where:
1. A supervisor agent coordinates research activities and delegates tasks
2. Multiple researcher agents work on specific sub-topics independently
3. Results are aggregated and compressed for final reporting

The supervisor uses parallel research execution to improve efficiency while
maintaining isolated context windows for each research topic.
"""

from typing_extensions import Literal
from langgraph.prebuilt import ToolNode

from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
)
from langgraph.graph import StateGraph, START, END

from src.deep_research_agent.state import SupervisorState
from src.deep_research_agent.tools.conduct_research_tool import conduct_research
from src.deep_research_agent.tools.research_complete_tool import research_complete
from src.deep_research_agent.tools.think_tool import think_tool
from src.utils.helpers import get_prompt_template, get_today_str
from src.config import ROOT_DIR
from src.utils.models import get_model
from langchain_core.runnables import RunnableConfig


class SupervisorResearchAgent:
    def __init__(self, interleaved_thinking=True, agent_reasoning="low"):
        self.supervisor_agent_system_prompt = get_prompt_template(
            ROOT_DIR / "src/deep_research_agent/prompts/lead_researcher_prompt.jinja"
        )
        self.tools = [conduct_research, research_complete]
        self.interleaved_thinking = interleaved_thinking
        if self.interleaved_thinking:
            self.tools.append(think_tool)
        self.tool_node = ToolNode(
            tools=self.tools, name="tool_node", messages_key="supervisor_messages"
        )
        self.model = get_model(reasoning=agent_reasoning)
        self.model_with_tools = self.model.bind_tools(self.tools)

    async def llm_call(self, state: SupervisorState, config: RunnableConfig) -> dict:
        """Analyze state and decide next actions."""
        try:
            supervisor_messages = state.get("supervisor_messages", [])

            # Prepare system message with current date and constraints
            system_message = self.supervisor_agent_system_prompt.render(
                date=get_today_str(),
                max_concurrent_research_units=config.get("configurable", {}).get(
                    "max_concurrent_researchers", 3
                ),
                max_researcher_iterations=config.get("configurable", {}).get(
                    "max_researcher_iterations", 6
                ),
                interleaved_thinking=self.interleaved_thinking,
            )
            messages = [SystemMessage(content=system_message)] + supervisor_messages
            model_response = await self.model_with_tools.ainvoke(messages)
            return {
                "supervisor_messages": [model_response],
                "is_llm_call_error": False,
                "num_retry_llm_call_node": 0,
            }
        except Exception as e:
            message = HumanMessage(
                content=f"The LLM threw the following error: {str(e)}"
            )
            return {
                "supervisor_messages": [message],
                "is_llm_call_error": True,
                "num_retry_llm_call_node": state.get("num_retry_llm_call_node", 0) + 1,
            }

    async def should_continue(
        self, state: SupervisorState, config: RunnableConfig
    ) -> Literal["tool_node", "__end__", "llm_call"]:
        """Decide whether to continue research or end."""
        last_message = state["supervisor_messages"][-1]

        if state.get("is_llm_call_error", False):
            if state["num_retry_llm_call_node"] > config.get("configurable", {}).get(
                "max_llm_call_retry"
            ):
                return "__end__"
            else:
                return "llm_call"

        if last_message.tool_calls:
            for tool_call in last_message.tool_calls:
                if tool_call.get("name", "") == "research_complete":
                    return "__end__"
            return "tool_node"

        return "__end__"

    def build_agent_graph(self) -> StateGraph:
        """Constructs and returns the research agent workflow graph."""
        agent_builder = StateGraph(SupervisorState)
        agent_builder.add_node("llm_call", self.llm_call)
        agent_builder.add_node("tool_node", self.tool_node)

        agent_builder.add_edge(START, "llm_call")
        agent_builder.add_conditional_edges(
            "llm_call",
            self.should_continue,
            {
                "tool_node": "tool_node",
                "__end__": END,
                "llm_call": "llm_call",
            },
        )
        agent_builder.add_edge("tool_node", "llm_call")

        return agent_builder.compile()
