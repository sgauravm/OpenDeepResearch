import streamlit as st
import uuid
import dotenv
import asyncio
from collections import defaultdict
from enum import Enum
from typing import Dict, Any, Generator, Optional, List, Union

# --- External Imports (Preserved from your original code) ---
# Ensure your 'src' folder exists in the same directory as this file
from langchain_core.messages import HumanMessage, ToolMessage, AIMessage
from langgraph.graph.state import CompiledStateGraph
from langchain_core.runnables import RunnableConfig
import itertools
import time

# Assuming these are your custom modules
try:
    from src.deep_research_agent.agents.research_agent import ResearcherAgent
    from src.deep_research_agent.agents.scoping_agent import ScopingAgent
    from src.utils.stream import run_async_generator
    from src.utils.models import get_model
    from src.config import FINAL_AGENT_CONFIG
    from src.deep_research_agent.agents.final_deep_research_agent import (
        DeepResearchAgent,
    )
    from src.types import Payload, ContentType
except ImportError as e:
    st.error(
        f"Error importing local modules: {e}. Please ensure 'src' directory structure is correct."
    )
    st.stop()

# Load environment variables
dotenv.load_dotenv()

# ==========================================
# 1. Constants & Enums (For Safety)
# ==========================================


class AgentType(str, Enum):
    MAIN = "main"
    SUPERVISOR = "supervisor"
    RESEARCHER = "researcher"


class ToolName(str, Enum):
    THINK = "think_tool"
    RESEARCH = "conduct_research"
    WEB_SEARCH = "web_search"
    COMPLETE = "research_complete"


# ==========================================
# 2. Logic Processor (Handles Graph Events)
# ==========================================


class StreamEventProcessor:
    """
    Handles the raw events coming from LangGraph and converts them
    into clean Payloads for the UI.
    """

    def __init__(self):
        self.reason_streaming_dict = defaultdict(lambda: False)

    @staticmethod
    def _get_agent_type(agent_path: list) -> str:
        if not agent_path:
            return AgentType.MAIN
        elif "supervisor_subgraph" in agent_path[-1]:
            return AgentType.SUPERVISOR
        return AgentType.RESEARCHER

    def _get_payload_from_tool_call(
        self, tool_call: dict, agent_type: str
    ) -> Optional[Dict]:
        name = tool_call["name"]
        args = tool_call.get("args", {})
        content = None

        if name == ToolName.THINK:
            content = f"Agent Reflection: {args.get('reflection')}"
        elif name == ToolName.RESEARCH:
            content = f"Delegating to Researcher subagent to research on topic: {args.get('research_topic')}"
        elif name == ToolName.WEB_SEARCH:
            content = f"Conducting web search on topic: {args.get('query')}"
        elif name == ToolName.COMPLETE:
            content = "Generating final report"
        else:
            return None

        return {
            "content": content,
            "content_type": ContentType.TOOL_CALLED,
            "tool_name": name,
            "agent_type": agent_type,
        }

    def _extract_sub_agent_messages(self, chunk: dict, agent_type: str) -> list:
        """Helper to extract message lists from different dictionary structures."""
        # Determine the key based on agent type
        msg_key = (
            "supervisor_messages"
            if agent_type == AgentType.SUPERVISOR
            else "researcher_messages"
        )

        # Determine the node source (LLM vs Tool)
        if "llm_call" in chunk:
            return chunk["llm_call"].get(msg_key, [])
        elif "tool_node" in chunk:
            return chunk["tool_node"].get(msg_key, [])

        return []

    def _handle_main_updates(
        self, chunk: dict, agent_type: str
    ) -> Generator[Dict, None, None]:
        """Handles top-level updates (Clarification requests or Final Reports)."""
        clarify = chunk.get("clarify_with_user")
        final = chunk.get("final_report_generation")

        content = None
        if clarify and clarify.get("messages"):
            content = clarify["messages"][0].content
        elif final:
            content = final.get("final_report", "")

        if content:
            yield {
                "content": content,
                "content_type": ContentType.ASSISTANT_MESSAGE,
                "agent_type": agent_type,
            }

    def _handle_sub_agent_updates(
        self, chunk: dict, agent_type: str
    ) -> Generator[Dict, None, None]:
        """Handles sub-agent tool calls and completions."""
        messages = self._extract_sub_agent_messages(chunk, agent_type)

        for msg in messages:
            # 1. Handle Tool Calls (The request to do work)
            if isinstance(msg, AIMessage) and msg.tool_calls:
                for tc in msg.tool_calls:
                    payload = self._get_payload_from_tool_call(tc, agent_type)
                    if payload:
                        yield payload

            # 2. Handle Tool Output (The work is done)
            if isinstance(msg, ToolMessage):
                yield {
                    "content": "",
                    "content_type": ContentType.TOOL_CALL_COMPLETE,
                    "tool_name": msg.name,
                    "agent_type": agent_type,
                }

    def _handle_reasoning_stream(
        self, chunk, node_name: str, agent_type: str
    ) -> Optional[Dict]:
        """Detects if reasoning (Chain of Thought) is starting, ongoing, or stopping."""
        msg_chunk, metadata = chunk
        key = f"{node_name}_{agent_type}"

        if metadata.get("langgraph_node") != node_name:
            return None

        reasoning = msg_chunk.additional_kwargs.get("reasoning_content", "")

        # Start or Continue Reasoning
        if reasoning:
            if not self.reason_streaming_dict[key]:
                self.reason_streaming_dict[key] = True
                c_type = ContentType.START_STREAM_REASON
            else:
                c_type = ContentType.STREAMING_REASON

            return {
                "content": reasoning,
                "content_type": c_type,
                "agent_type": agent_type,
                "node_name": node_name,
            }

        # Stop Reasoning
        if (msg_chunk.content or msg_chunk.tool_calls) and self.reason_streaming_dict[
            key
        ]:
            self.reason_streaming_dict[key] = False
            return {
                "content": "",
                "content_type": "stop_stream_reason",
                "agent_type": agent_type,
                "node_name": node_name,
            }

        return None

    async def process_stream(
        self, agent_graph: CompiledStateGraph, state: dict, config: RunnableConfig
    ):
        """Main generator that yields clean payloads."""
        async for subagent, mode, chunk in agent_graph.astream(
            input=state,
            stream_mode=["messages", "updates", "values", "custom"],
            config=config,
            subgraphs=True,
        ):
            agent_type = self._get_agent_type(subagent)

            # --- Modularized Update Handling ---
            if mode == "updates":
                if agent_type == AgentType.MAIN:
                    for payload in self._handle_main_updates(chunk, agent_type):
                        yield payload
                else:
                    for payload in self._handle_sub_agent_updates(chunk, agent_type):
                        yield payload

            # --- Message Handling (Reasoning) ---
            if mode == "messages":
                payload = self._handle_reasoning_stream(chunk, "llm_call", agent_type)
                if payload:
                    yield payload

            if mode == "custom":
                if chunk.get("content_type") == ContentType.COMPRESSION_START:
                    yield {
                        "content": "Compressing research.",
                        "content_type": ContentType.TOOL_CALLED,
                        "tool_name": chunk["node_name"],
                        "agent_type": agent_type,
                    }
                elif chunk.get("content_type") == ContentType.COMPRESSION_STOP:
                    yield {
                        "content": "",
                        "content_type": ContentType.TOOL_CALL_COMPLETE,
                        "tool_name": chunk["node_name"],
                        "agent_type": agent_type,
                    }

            # --- Value Handling ---
            if mode == "values" and chunk:
                yield {"content": chunk, "content_type": "response"}


# ==========================================
# 3. UI Manager (Handles Streamlit Rendering)
# ==========================================


class StreamlitUI:
    def __init__(self):
        self.processor = StreamEventProcessor()
        self.placeholder_dict = {}
        self.reason_accum = ""
        self.current_status_container = None
        self.current_chat_container = None
        self.reason_status_container = None
        self.reason_placeholder = None
        self.agent = None

    def initialize_session(self):
        if "messages" not in st.session_state:
            st.session_state.messages = []
        if "thread" not in st.session_state or st.session_state.thread is None:
            st.session_state.thread = uuid.uuid4().hex[:8]
            # Initialize Agent
            st.session_state.agent = DeepResearchAgent(
                agent_reasoning=self.reasoning_mode,
                interleaved_thinking=self.interleaved_thinking,
            ).build_agent_graph()

    def render_sidebar(self):
        with st.sidebar:
            st.header("Configuration")
            if st.button("New Chat"):
                st.session_state.messages = []
                st.session_state.thread = None
                st.rerun()
            self.show_reasoning = st.toggle("Show reasoning", value=True)
            self.interleaved_thinking = st.toggle(
                "Enable interleaved thinking", value=True
            )

            self.max_web_search_calls = st.number_input(
                "Max Web Search Calls", min_value=1, max_value=5, value=3, step=1
            )

            self.max_web_search_results = st.number_input(
                "Max Web Search Results", min_value=1, max_value=3, value=1, step=1
            )

            self.max_researcher_iterations = st.number_input(
                "Max Researcher Iterations", min_value=1, max_value=5, value=3, step=1
            )
            reasoning_options = {
                "Low": "low",
                "Medium": "medium",
                "High": "high",
            }

            label = st.selectbox(
                "Reasoning level", list(reasoning_options.keys()), index=1
            )
            self.reasoning_mode = reasoning_options[label]

    def render_chat_history(self):
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    def _ensure_status_container(self):
        if self.current_chat_container is None:
            self.current_chat_container = st.chat_message("assistant")
            with self.current_chat_container:
                self.current_status_container = st.status(
                    label="Processing...", expanded=True
                )

    def handle_payload(self, payload: Dict):
        c_type = payload["content_type"]

        # 1. Final Assistant Message
        if c_type == ContentType.ASSISTANT_MESSAGE:
            if self.current_status_container:
                self.current_status_container.update(
                    label="Finished", state="complete", expanded=False
                )

            with st.chat_message("assistant"):
                st.markdown(payload["content"])

            st.session_state.messages.append(
                {"role": "assistant", "content": payload["content"]}
            )
            # Reset UI state handles
            self.current_chat_container = None
            self.current_status_container = None

        # 2. Tool Calls (Status Updates)
        elif c_type == ContentType.TOOL_CALLED:
            self._ensure_status_container()
            with self.current_status_container:
                t_name = payload["tool_name"]
                content = payload["content"]

                if t_name == ToolName.THINK:
                    st.markdown(self._get_html_text(content), unsafe_allow_html=True)
                elif t_name == ToolName.COMPLETE:
                    st.write(":blue[Generating final report...]")
                else:
                    p = st.empty()
                    p.markdown(f":gray[:small[ ⏳ *{content}*]]")
                    # Store placeholder to update later
                    key = f"{t_name}_{payload['agent_type']}"
                    self.placeholder_dict[key] = {"text": content, "placeholder": p}

        # 3. Tool Complete
        elif c_type == ContentType.TOOL_CALL_COMPLETE:
            key = f"{payload['tool_name']}_{payload['agent_type']}"
            if key in self.placeholder_dict:
                data = self.placeholder_dict[key]
                data["placeholder"].markdown(f":gray[:small[ ✅ *{data['text']}*]]")
                del self.placeholder_dict[key]

        # 4. Reasoning Streaming
        elif c_type == ContentType.START_STREAM_REASON and self.show_reasoning:
            self._ensure_status_container()
            with self.current_status_container:
                self.reason_status_container = st.status(
                    label="🧠 Thinking...", state="running", expanded=True
                )
                with self.reason_status_container:
                    self.reason_placeholder = st.empty()
            self.reason_accum += payload["content"]
            self._update_reasoning_ui()

        elif c_type == ContentType.STREAMING_REASON and self.show_reasoning:
            self.reason_accum += payload["content"]
            self._update_reasoning_ui()

        elif (
            c_type == "stop_stream_reason" and self.show_reasoning
        ):  # or ContentType.STOP_STREAM_REASON if exists
            if self.reason_status_container:
                self.reason_status_container.update(
                    label="🧠 Thoughts", state="complete", expanded=False
                )
            self.reason_accum = ""

    def _get_html_text(self, text):
        return f"""<div style="color: gray; font-size: 0.85em; font-style: italic; white-space: pre-wrap;">{text}</div>"""

    def _update_reasoning_ui(self):
        if self.reason_placeholder:
            html = self._get_html_text(self.reason_accum)
            self.reason_placeholder.markdown(html, unsafe_allow_html=True)


# ==========================================
# 4. Main Application Loop
# ==========================================


def main():
    st.set_page_config(page_title="Deep Research Agent", layout="centered")
    st.title("Deep Research Agent")

    ui = StreamlitUI()
    ui.render_sidebar()
    ui.initialize_session()
    ui.render_chat_history()

    # User Input
    if prompt := st.chat_input("Type your query here..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Config for this run
        config = {
            "configurable": {
                "thread_id": st.session_state.thread,
                "max_web_search_calls": ui.max_web_search_calls,
                "max_web_search_results": ui.max_web_search_results,
                "max_llm_call_retry": FINAL_AGENT_CONFIG["max_llm_call_retry"],
                "max_researcher_iterations": ui.max_researcher_iterations,
                "max_concurrent_researchers": FINAL_AGENT_CONFIG[
                    "max_concurrent_researchers"
                ],
                "interleaved_thinking": ui.interleaved_thinking,
                "agent_reasoning": ui.reasoning_mode,
            },
        }

        state = {"messages": [HumanMessage(content=prompt)]}

        # Run execution loop
        # We wrap the generator to run in the main thread context
        stream_generator = ui.processor.process_stream(
            agent_graph=st.session_state.agent, state=state, config=config
        )

        # Iterate through the async generator
        for payload in run_async_generator(stream_generator):
            ui.handle_payload(payload)


if __name__ == "__main__":
    main()
