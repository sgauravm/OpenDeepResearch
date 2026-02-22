"""Research Agent Implementation as a Class."""

from typing import Literal
from pydantic import BaseModel
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, SystemMessage, filter_messages
from langchain.chat_models import init_chat_model
from langgraph.config import get_stream_writer
from langgraph.prebuilt import ToolNode
from langchain.tools import ToolRuntime
from langchain_core.runnables import RunnableConfig

from src.deep_research_agent.state import ResearcherOutputState, ResearcherState
from src.deep_research_agent.tools.search_tool import web_search
from src.deep_research_agent.tools.think_tool import think_tool
from src.deep_research_agent.tools.research_complete_tool import research_complete
from src.utils.helpers import get_prompt_template, get_today_str
from src.config import ROOT_DIR
from src.utils.models import get_model
from src.types import ContentType


class ResearcherAgent:
    """Encapsulates the research agent with tools, models, and workflow."""

    def __init__(
        self,
        interleaved_thinking: bool = True,
        agent_reasoning: Literal["low", "medium", "high"] | None = "medium",
    ):
        # Load prompts
        self.research_agent_system_prompt = get_prompt_template(
            ROOT_DIR
            / "src/deep_research_agent/prompts/research_agent_system_prompt.jinja"
        )
        self.interleaved_thinking = interleaved_thinking

        # Initialize tools
        self.tools = [web_search, research_complete]
        if self.interleaved_thinking:
            self.tools.append(think_tool)

        self.tools_by_name = {tool.name: tool for tool in self.tools}
        self.tool_node = ToolNode(
            tools=self.tools, name="tool_node", messages_key="researcher_messages"
        )

        # Initialize models
        self.model = get_model(reasoning=agent_reasoning)
        self.model_with_tools = self.model.bind_tools(self.tools)

    # ===== Node Implementations =====
    def llm_call(self, state: ResearcherState) -> dict:
        """Analyze state and decide next actions."""
        try:
            model_response = self.model_with_tools.invoke(
                [
                    SystemMessage(
                        content=self.research_agent_system_prompt.render(
                            date=get_today_str(),
                            interleaved_thinking=self.interleaved_thinking,
                        )
                    )
                ]
                + state["researcher_messages"]
            )
            return {
                "researcher_messages": [model_response],
                "is_llm_call_error": False,
                "num_retry_llm_call_node": 0,
            }
        except Exception as e:
            message = HumanMessage(
                content=f"The LLM threw the following error: {str(e)}"
            )
            return {
                "researcher_messages": [message],
                "is_llm_call_error": True,
                "num_retry_llm_call_node": state.get("num_retry_llm_call_node", 0) + 1,
            }

    def compress_research(self, state: ResearcherState) -> dict:
        """Concatenate all collected content from research_notes for the supervisor."""
        writer = get_stream_writer()
        writer(
            {
                "content_type": ContentType.COMPRESSION_START,
                "content": "Assembling research findings.",
                "node_name": "compress_research",
            }
        )

        research_notes = state.get("research_notes", {})
        descriptions = [
            f"{filename}: {note.get('description', '')}"
            for filename, note in research_notes.items()
            if isinstance(note, dict) and note.get("description")
        ]

        if descriptions:
            compressed_research = (
                "=== INFORMATION COLLECTED BY RESEARCH AGENT ===\n\n"
                + "\n".join(f"- {d}" for d in descriptions)
            )
        else:
            compressed_research = "=== INFORMATION COLLECTED BY RESEARCH AGENT ===\n\nNo content collected."

        # TODO: This might not be required
        raw_notes = [
            str(m.content)
            for m in filter_messages(
                state["researcher_messages"], include_types=["tool", "ai"]
            )
        ]

        writer(
            {
                "content_type": ContentType.COMPRESSION_STOP,
                "content": "Done assembling research findings",
                "node_name": "compress_research",
            }
        )

        return {
            "compressed_research": compressed_research,
            "raw_notes": ["\n".join(raw_notes)],
            "num_web_search_calls": 0,
        }

    # ===== Routing Logic =====
    def should_continue(
        self, state: ResearcherState, config: RunnableConfig
    ) -> Literal["tool_node", "compress_research", "__end__", "llm_call"]:
        """Decide whether to continue research or compress results."""
        last_message = state["researcher_messages"][-1]
        if state.get("is_llm_call_error", False):
            if state["num_retry_llm_call_node"] > config.get("configurable", {}).get(
                "max_llm_call_retry"
            ):
                return "compress_research"
            else:
                return "llm_call"

        if last_message.tool_calls:
            for tool_call in last_message.tool_calls:
                if tool_call.get("name", "") == "research_complete":
                    return "compress_research"
            return "tool_node"

        return "compress_research"

    # ===== Agent Graph Builder =====
    def build_agent_graph(self) -> StateGraph:
        """Constructs and returns the research agent workflow graph."""
        agent_builder = StateGraph(ResearcherState, output_schema=ResearcherOutputState)
        agent_builder.add_node("llm_call", self.llm_call)
        agent_builder.add_node("tool_node", self.tool_node)
        agent_builder.add_node("compress_research", self.compress_research)

        agent_builder.add_edge(START, "llm_call")
        agent_builder.add_conditional_edges(
            "llm_call",
            self.should_continue,
            {
                "tool_node": "tool_node",
                "compress_research": "compress_research",
                "llm_call": "llm_call",
            },
        )
        agent_builder.add_edge("tool_node", "llm_call")
        agent_builder.add_edge("compress_research", END)

        return agent_builder.compile()
