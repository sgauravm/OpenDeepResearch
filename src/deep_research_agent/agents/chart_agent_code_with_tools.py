import re
from typing import Any, Literal, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig

from langgraph.graph import END, START, StateGraph

from src.config import PROMPTS_DIR
from src.deep_research_agent.tools.plotly_python_code_executer_tool import (
    execute_plotly_chart_python_code,
    data_not_provided,
)
from src.utils.helpers import get_prompt_template
from src.utils.models import get_model
from langgraph.prebuilt import ToolNode
from langgraph.graph import MessagesState


load_dotenv()


class ChartAgentState(MessagesState):
    """Workflow state."""

    chart_json: dict
    chart_name: str
    is_code_error: bool
    data_provided: bool
    num_code_correction: int
    python_code: str


class ChartAgent:
    def __init__(self):
        self.model = get_model(reasoning="low")
        self.tools = [execute_plotly_chart_python_code, data_not_provided]
        self.model_with_tools = self.model.bind_tools(self.tools)
        self.tool_node = ToolNode(
            tools=self.tools, name="tool_node", messages_key="messages"
        )
        self.chart_agent_prompt_template = get_prompt_template(
            str(PROMPTS_DIR / "plotly_chart_agent_system_prompt.jinja")
        )

    def llm_call(self, state: ChartAgentState) -> dict:
        """Analyze state and decide next actions."""

        model_response = self.model_with_tools.invoke(
            [SystemMessage(content=self.chart_agent_prompt_template.render())]
            + state["messages"]
        )
        return {"messages": [model_response]}

    def should_continue(self, state: ChartAgentState) -> Literal["llm_call", "__end__"]:
        """Determine if the chart generation should continue."""
        # INSERT_YOUR_CODE
        messages = state.get("messages", [])
        if not messages:
            return "__end__"
        last_message = messages[-1]
        # For langchain_core.messages AIMessage/ToolMessage - tool_calls may be in additional_kwargs or as tool_calls attribute
        tool_calls = []
        if hasattr(last_message, "tool_calls"):
            tool_calls = getattr(last_message, "tool_calls", [])
        elif isinstance(last_message, dict):
            tool_calls = last_message.get("tool_calls", [])
        elif hasattr(last_message, "additional_kwargs"):
            # e.g., for AIMessage, 'tool_calls' is under additional_kwargs for OpenAI-like models
            tool_calls = last_message.additional_kwargs.get("tool_calls", [])
        if not tool_calls:
            return "__end__"
        # TODO: Max code correction iteration should be part of config
        if (
            state.get("is_code_error", False)
            and state.get("num_code_correction", 0) < 3
        ):
            return "llm_call"
        else:
            return "__end__"

    def build_agent_graph(self) -> StateGraph:
        """Build the agent graph."""
        graph = StateGraph(ChartAgentState)
        graph.add_node("llm_call", self.llm_call)
        graph.add_node("tool_node", self.tool_node)
        graph.add_edge(START, "llm_call")
        graph.add_edge("llm_call", "tool_node")
        graph.add_conditional_edges(
            "tool_node",
            self.should_continue,
            {
                "llm_call": "llm_call",
                "__end__": END,
            },
        )
        return graph.compile()
