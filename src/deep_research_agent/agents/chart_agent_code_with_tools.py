from typing import Literal

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph, MessagesState
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
    """Workflow state."""

    # Initial chart instruction from user/section writer
    initial_instruction: str
    # Enhanced detailed instruction for Plotly
    detailed_instruction: str
    # Whether to use detailed instruction (True) or initial instruction (False)
    use_detailed_instruction: bool
    chart_json: dict
    chart_name: str
    is_code_error: bool
    data_provided: bool
    num_code_correction: int
    python_code: str


class ChartAgent:
    def __init__(self):
        self.model = get_model(reasoning="low")
        self.design_model = get_model(reasoning="medium")
        self.tools = [execute_plotly_chart_python_code, data_not_provided]
        self.model_with_tools = self.model.bind_tools(self.tools)
        self.tool_node = ToolNode(
            tools=self.tools, name="tool_node", messages_key="messages"
        )
        self.chart_agent_prompt_template = get_prompt_template(
            str(PROMPTS_DIR / "plotly_chart_agent_system_prompt.jinja")
        )
        self.chart_design_prompt_template = get_prompt_template(
            str(PROMPTS_DIR / "chart_design_prompt.jinja")
        )

    def design_chart(self, state: ChartAgentState) -> dict:
        """
        Transform initial chart instruction into a detailed, beautiful Plotly specification.

        This node analyzes the initial instruction and creates comprehensive styling
        and data formatting guidelines for generating visually appealing static charts.
        If use_detailed_instruction is False, skips the design step and uses initial instruction.
        """
        initial_instruction = state.get("initial_instruction", "")
        use_detailed = state.get("use_detailed_instruction", True)

        if not initial_instruction:
            # Fallback to messages if initial_instruction not provided
            messages = state.get("messages", [])
            if messages and isinstance(messages[0], HumanMessage):
                initial_instruction = messages[0].content

        # If not using detailed instruction, pass initial instruction directly
        if not use_detailed:
            return {
                "detailed_instruction": initial_instruction,
                "messages": [HumanMessage(content=initial_instruction)],
            }

        # Generate detailed instruction for beautiful static charts
        design_prompt = self.chart_design_prompt_template.render(
            initial_instruction=initial_instruction
        )

        response = self.design_model.invoke([SystemMessage(content=design_prompt)])
        detailed_instruction = response.content

        return {
            "detailed_instruction": detailed_instruction,
            "messages": [HumanMessage(content=detailed_instruction)],
        }

    def llm_call(self, state: ChartAgentState) -> dict:
        """Generate Plotly code based on the detailed instruction."""
        model_response = self.model_with_tools.invoke(
            [SystemMessage(content=self.chart_agent_prompt_template.render())]
            + state["messages"]
        )
        return {"messages": [model_response]}

    def should_continue(self, state: ChartAgentState) -> Literal["llm_call", "__end__"]:
        """Determine if the chart generation should continue."""
        # If chart was successfully generated, we're done
        chart_json = state.get("chart_json")
        if chart_json and isinstance(chart_json, dict) and len(chart_json) > 0:
            return "__end__"

        # If data was not provided, we're done
        if not state.get("data_provided", True):
            return "__end__"

        # Count messages to prevent infinite loops
        messages = state.get("messages", [])
        if len(messages) > 12:
            return "__end__"

        # Count error messages to determine retry count
        error_count = sum(
            1 for m in messages
            if hasattr(m, "content") and "Failed to execute" in str(m.content)
        )

        # Retry up to 3 times on errors
        if error_count > 0 and error_count < 4:
            return "llm_call"

        # Default: end
        return "__end__"

    def build_agent_graph(self) -> StateGraph:
        """Build the agent graph with design and generation nodes."""
        graph = StateGraph(ChartAgentState)

        # Add nodes
        graph.add_node("design_chart", self.design_chart)
        graph.add_node("llm_call", self.llm_call)
        graph.add_node("tool_node", self.tool_node)

        # Define edges
        graph.add_edge(START, "design_chart")
        graph.add_edge("design_chart", "llm_call")
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
