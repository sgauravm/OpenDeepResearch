import streamlit as st
import uuid
import dotenv
from typing import Dict
from langchain_core.messages import HumanMessage

# Assuming these are your custom modules
try:
    from src.utils.stream import run_async_generator, StreamEventProcessor
    from src.config import FINAL_AGENT_CONFIG
    from src.deep_research_agent.agents.final_deep_research_agent import (
        DeepResearchAgent,
    )
    from src.types import ContentType, ToolName
except ImportError as e:
    st.error(
        f"Error importing local modules: {e}. Please ensure 'src' directory structure is correct."
    )
    st.stop()

# Load environment variables
dotenv.load_dotenv()


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
