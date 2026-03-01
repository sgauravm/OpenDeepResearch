"""Section Writer Agent Implementation.

This agent writes a single section of a research report with chart creation capability.
It analyzes source content, creates charts when appropriate, and produces
well-structured markdown content with proper citations.
"""

from typing import Literal, Annotated, Sequence

from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel

from src.config import ROOT_DIR
from src.utils.helpers import get_prompt_template, get_today_str
from src.utils.models import get_model
from src.deep_research_agent.tools.create_chart_tool import create_chart


class SectionWriterState(BaseModel):
    """Internal state for the section writer agent."""

    messages: Annotated[Sequence[BaseMessage], add_messages] = []
    section_content: str = ""
    is_complete: bool = False


class SectionWriterAgent:
    """Agent for writing report sections with chart creation capability."""

    def __init__(
        self,
        reasoning: Literal["low", "medium", "high"] = "low",
    ):
        self.model = get_model(reasoning=reasoning)

        # Tools for section writer agent
        self.tools = [create_chart]
        self.model_with_tools = self.model.bind_tools(self.tools)
        self.tool_node = ToolNode(tools=self.tools, name="section_writer_tools")

        self.system_template = get_prompt_template(
            ROOT_DIR / "src/deep_research_agent/prompts/section_writer_system.jinja"
        )
        self.human_template = get_prompt_template(
            ROOT_DIR / "src/deep_research_agent/prompts/section_writer.jinja"
        )

    def _llm_call(self, state: dict) -> dict:
        """Call the LLM with tools."""
        messages = state.get("messages", [])
        response = self.model_with_tools.invoke(messages)
        return {"messages": [response]}

    def _should_continue(self, state: dict) -> Literal["tools", "extract_content"]:
        """Determine if we should call tools or extract final content."""
        messages = state.get("messages", [])
        if not messages:
            return "extract_content"

        last_message = messages[-1]

        # Check for tool calls
        tool_calls = []
        if hasattr(last_message, "tool_calls"):
            tool_calls = getattr(last_message, "tool_calls", [])
        elif isinstance(last_message, dict):
            tool_calls = last_message.get("tool_calls", [])
        elif hasattr(last_message, "additional_kwargs"):
            tool_calls = last_message.additional_kwargs.get("tool_calls", [])

        if tool_calls:
            return "tools"
        return "extract_content"

    def _extract_content(self, state: dict) -> dict:
        """Extract the final section content from the last AI message."""
        messages = state.get("messages", [])
        content = ""

        # Find the last AI message that contains the section content
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and msg.content:
                # Skip messages that only contain tool calls
                tool_calls = getattr(msg, "tool_calls", [])
                if not tool_calls and msg.content:
                    content = msg.content
                    break

        return {"section_content": content, "is_complete": True}

    def build_agent_graph(self):
        """Build the section writer agent graph."""
        agent_builder = StateGraph(dict)

        agent_builder.add_node("llm_call", self._llm_call)
        agent_builder.add_node("tools", self.tool_node)
        agent_builder.add_node("extract_content", self._extract_content)

        agent_builder.add_edge(START, "llm_call")
        agent_builder.add_conditional_edges(
            "llm_call",
            self._should_continue,
            {
                "tools": "tools",
                "extract_content": "extract_content",
            },
        )
        agent_builder.add_edge("tools", "llm_call")
        agent_builder.add_edge("extract_content", END)

        return agent_builder.compile()

    def write_section(
        self,
        research_brief: str,
        section_names: str,
        cur_section: str,
        section_description: str,
        previous_section: str,
        source_content: str,
    ) -> str:
        """
        Write a section of the research report.

        Args:
            research_brief: The original research instruction
            section_names: List of all section names in order
            cur_section: Name of the current section to write
            section_description: Writing instructions for this section
            previous_section: Text of the previous section for style consistency
            source_content: Source material for this section

        Returns:
            The written section content in markdown format
        """
        # Build system prompt
        system_prompt = self.system_template.render()

        # Build human message with section details
        human_message = self.human_template.render(
            research_brief=research_brief,
            section_names=section_names,
            cur_section=cur_section,
            section_description=section_description,
            previous_section=previous_section,
            source_content=source_content,
            date=get_today_str(),
        )

        # Build and run the agent
        agent = self.build_agent_graph()

        initial_state = {
            "messages": [
                SystemMessage(content=system_prompt),
                HumanMessage(content=human_message),
            ],
            "section_content": "",
            "is_complete": False,
        }

        result = agent.invoke(initial_state)
        content = result.get("section_content", "")

        # Ensure section starts with proper heading
        if content and not content.strip().startswith("#"):
            content = f"# {cur_section}\n\n{content}"

        return content
