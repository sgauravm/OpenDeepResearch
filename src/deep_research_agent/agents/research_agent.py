"""Research Agent Implementation as a Class."""

from typing import Literal
from pydantic import BaseModel
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    ToolMessage,
    filter_messages,
)
from langchain.chat_models import init_chat_model
from langgraph.config import get_stream_writer
from langgraph.prebuilt import ToolNode
from langchain.tools import ToolRuntime
from langchain_core.runnables import RunnableConfig

from src.deep_research_agent.state import ResearcherOutputState, ResearcherState
from src.deep_research_agent.tools.search_tool import web_search
from src.deep_research_agent.tools.think_tool import think_tool
from src.utils.helpers import get_prompt_template, get_today_str
from src.config import ROOT_DIR
from src.utils.models import get_model


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
        self.compress_research_system_prompt = get_prompt_template(
            ROOT_DIR
            / "src/deep_research_agent/prompts/compress_research_system_prompt.jinja"
        )
        self.compress_research_human_prompt = get_prompt_template(
            ROOT_DIR
            / "src/deep_research_agent/prompts/compress_research_human_prompt.jinja"
        )
        self.interleaved_thinking = interleaved_thinking

        # Initialize tools
        self.tools = [web_search]
        if self.interleaved_thinking:
            self.tools.append(think_tool)

        self.tools_by_name = {tool.name: tool for tool in self.tools}
        self.tool_node = ToolNode(
            tools=self.tools, name="tool_node", messages_key="researcher_messages"
        )

        # Initialize models
        self.model = get_model(reasoning=agent_reasoning)
        self.model_with_tools = self.model.bind_tools(self.tools)
        self.summarization_model = get_model()
        self.compress_model = get_model(max_tokens=32000)

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
        """Compress research messages into a summary."""
        writer = get_stream_writer()
        writer({"type": "node_info", "message": "Compressing research findings."})

        system_message = self.compress_research_system_prompt.render(
            date=get_today_str()
        )
        researcher_messages = state.get("researcher_messages", [])
        if len(researcher_messages) > 0:
            researcher_messages = researcher_messages[:-1]
        messages = (
            [SystemMessage(content=system_message)]
            + researcher_messages
            + [
                HumanMessage(
                    content=self.compress_research_human_prompt.render(
                        research_topic=state["research_topic"]
                    )
                )
            ]
        )

        response = self.compress_model.invoke(messages)
        raw_notes = [
            str(m.content)
            for m in filter_messages(
                state["researcher_messages"], include_types=["tool", "ai"]
            )
        ]

        return {
            "compressed_research": str(response.content),
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
