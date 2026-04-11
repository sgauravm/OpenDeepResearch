"""Plotly code-generation chart agent (used by the "code" chart mode).

This is the legacy chart backend, kept as an opt-in alternative to the
default structured renderer. It has the LLM write plotly code which is
executed in a subprocess sandbox; on errors, it retries up to a few times.

Differences from the earlier version:

- The separate `design_chart` node has been removed. It expanded a rough
  instruction into a verbose styling spec which pushed small models
  further from correct code. The agent now goes directly from the
  incoming instruction (a ``HumanMessage`` in ``state["messages"]``) to
  code generation.
- Error context passed back to the model on retry is wider than before;
  see ``code_mode_error_context_lines`` in ``CHART_AGENT_CONFIG``.
"""

from typing import Literal

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, MessagesState, START, StateGraph
from langgraph.prebuilt import ToolNode

from src.config import PROMPTS_DIR
from src.deep_research_agent.tools.plotly_python_code_executer_tool import (
    data_not_provided,
    execute_plotly_chart_python_code,
)
from src.utils.helpers import get_prompt_template
from src.utils.models import get_model


load_dotenv()


class ChartAgentState(MessagesState):
    """Workflow state for the plotly code chart agent."""

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
        """Generate plotly code based on the conversation so far."""
        model_response = self.model_with_tools.invoke(
            [SystemMessage(content=self.chart_agent_prompt_template.render())]
            + state["messages"]
        )
        return {"messages": [model_response]}

    def should_continue(
        self, state: ChartAgentState
    ) -> Literal["llm_call", "__end__"]:
        """Decide whether to retry code generation after a tool call."""
        # Successfully generated chart: done.
        chart_json = state.get("chart_json")
        if chart_json and isinstance(chart_json, dict) and len(chart_json) > 0:
            return "__end__"

        # Data not provided: done.
        if not state.get("data_provided", True):
            return "__end__"

        # Hard cap on loop length.
        messages = state.get("messages", [])
        if len(messages) > 12:
            return "__end__"

        # Retry up to 3 times when errors accumulate.
        error_count = sum(
            1
            for m in messages
            if hasattr(m, "content") and "Failed to execute" in str(m.content)
        )
        if 0 < error_count < 4:
            return "llm_call"

        return "__end__"

    def build_agent_graph(self) -> StateGraph:
        """Build the code-generation graph.

        Entry contract: callers put a ``HumanMessage`` containing the chart
        instruction in ``state["messages"]`` before invoking.
        """
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
