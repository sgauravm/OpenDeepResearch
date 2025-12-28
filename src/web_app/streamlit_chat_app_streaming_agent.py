import random
import streamlit as st
import uuid
import dotenv
from src.deep_research_agent.agents.research_agent import ResearcherAgent
from src.deep_research_agent.agents.scoping_agent import ScopingAgent
from langchain_core.messages import HumanMessage, ToolMessage, AIMessage
from src.utils.stream import (
    # response_generator_emulator,
    # graph_stream_generator,
    run_async_generator,
)
from src.utils.models import get_model
from langgraph.graph.state import CompiledStateGraph
from langchain_core.runnables import RunnableConfig
import time
from langchain_core.language_models.chat_models import BaseChatModel
from src.config import FINAL_AGENT_CONFIG
from src.deep_research_agent.agents.final_deep_research_agent import DeepResearchAgent
from src.types import Payload, ContentType

dotenv.load_dotenv()


######################################################
# top 5 indian restaurants in toronto
from collections import defaultdict


def _get_agent_type(agent):
    if len(agent) == 0:
        agent_type = "main"
    elif "supervisor_subgraph" in agent[-1]:
        agent_type = "supervisor"
    else:
        agent_type = "researcher"
    return agent_type


def _get_payload_from_tool_call(tool_call, agent_type) -> Payload | None:
    name = tool_call["name"]
    args = tool_call.get("args", {})

    if name == "think_tool":
        content = f"Agent Reflection: {args.get('reflection')}"
    elif name == "conduct_research":
        content = f"Conducting research on topic: {args.get('research_topic')}"
    elif name == "web_search":
        content = f"Conducting web search on topic: {args.get('query')}"
    elif name == "research_complete":
        content = "Generating final report"
    else:
        return None  # directly return None for unknown tool

    return {
        "content": content,
        "content_type": ContentType.TOOL_CALLED,
        "tool_name": name,
        "agent_type": agent_type,
    }


def _get_messages(chunk, agent_type):
    message_key = (
        "supervisor_messages" if agent_type == "supervisor" else "researcher_messages"
    )
    if "llm_call" in chunk:
        messages = chunk["llm_call"].get(message_key, [])
    elif "tool_node" in chunk:
        messages = chunk["tool_node"].get(message_key, [])
    else:
        messages = []
    return messages


def _handle_updates(chunk, agent_type):
    if agent_type == "main":
        yield from _handle_main_updates(chunk, agent_type)
    else:
        yield from _handle_other_agent_updates(chunk, agent_type)


def _handle_main_updates(chunk, agent_type):
    content = None
    # 1. check clarification
    clarify = chunk.get("clarify_with_user")
    if clarify and clarify.get("messages"):
        content = clarify["messages"][0].content

    # 2. check final report
    final = chunk.get("final_report_generation")
    if final:
        content = final.get("final_report", "")
    if content:
        yield {
            "content": content,
            "content_type": ContentType.ASSISTANT_MESSAGE,
            "agent_type": agent_type,
        }


def _handle_other_agent_updates(chunk, agent_type):
    messages = _get_messages(chunk, agent_type)

    for message in messages:

        if isinstance(message, AIMessage) and message.tool_calls:
            for tool_call in message.tool_calls:
                payload = _get_payload_from_tool_call(tool_call, agent_type)
                if payload:
                    yield payload

        if isinstance(message, ToolMessage):
            yield {
                "content": "",
                "content_type": ContentType.TOOL_CALL_COMPLETE,
                "tool_name": message.name,
                "agent_type": agent_type,
            }


class HandleStreaming:
    def __init__(self):
        self.reason_streaming_dict = defaultdict(lambda: False)

    def get_reason_payload(self, chunk, node_name, agent_type):
        msg_chunk, metadata = chunk
        payload = None
        streaming_dict_key = f"{node_name}_{agent_type}"
        # Start reason streaming
        if (
            metadata["langgraph_node"] == node_name
            and msg_chunk.additional_kwargs.get("reasoning_content", "") != ""
        ):
            if not self.reason_streaming_dict[streaming_dict_key]:
                self.reason_streaming_dict[streaming_dict_key] = True
                content_type = ContentType.START_STREAM_REASON
            else:
                content_type = ContentType.STREAMING_REASON

            payload = {
                "content": msg_chunk.additional_kwargs.get("reasoning_content", ""),
                "content_type": content_type,
                "agent_type": agent_type,
                "node_name": node_name,
            }

        # Stop reason streaming
        if metadata["langgraph_node"] == node_name and (
            msg_chunk.content != "" or msg_chunk.tool_calls
        ):
            if self.reason_streaming_dict[streaming_dict_key]:
                self.reason_streaming_dict[streaming_dict_key] = False
                payload = {
                    "content": "",
                    "content_type": "stop_stream_reason",
                    "agent_type": agent_type,
                    "node_name": node_name,
                }
        return payload


async def graph_stream_generator(
    agent: CompiledStateGraph,
    state: dict,
    config: RunnableConfig | None = None,
    subgraphs: bool = True,
):
    handle_streaming = HandleStreaming()
    async for subagent, mode, chunk in agent.astream(
        input=state,
        stream_mode=["messages", "updates", "values"],
        config=config,
        subgraphs=subgraphs,
    ):
        agent_type = _get_agent_type(subagent)

        if mode == "updates":
            for payload in _handle_updates(chunk, agent_type):
                yield payload

        if mode == "messages":
            payload = handle_streaming.get_reason_payload(chunk, "llm_call", agent_type)
            if payload:
                yield payload

        if mode == "values" and chunk:
            yield {"content": chunk, "content_type": "response"}


######################################################


think_tool_required = st.sidebar.selectbox(
    "Do you want to include think tool?", ("Yes", "No")
)

reset_clicked = st.sidebar.button("New Chat")


# Initialize session state
if "messages" not in st.session_state or reset_clicked:
    st.session_state.messages = []

if "thread" not in st.session_state or reset_clicked:
    st.session_state.thread = uuid.uuid4().hex[:8]
    st.session_state.agent = DeepResearchAgent(
        agent_reasoning=FINAL_AGENT_CONFIG["agent_reasoning"],
        interleaved_thinking=FINAL_AGENT_CONFIG["interleaved_thinking"],
    ).build_agent_graph()

config = {
    "configurable": {
        "thread_id": st.session_state.thread,
        "max_web_search_calls": FINAL_AGENT_CONFIG["max_web_search_calls"],
        "max_web_search_results": FINAL_AGENT_CONFIG["max_web_search_results"],
        "max_llm_call_retry": FINAL_AGENT_CONFIG["max_llm_call_retry"],
        "max_researcher_iterations": FINAL_AGENT_CONFIG["max_researcher_iterations"],
        "max_concurrent_researchers": FINAL_AGENT_CONFIG["max_concurrent_researchers"],
        "interleaved_thinking": FINAL_AGENT_CONFIG["interleaved_thinking"],
        "agent_reasoning": FINAL_AGENT_CONFIG["agent_reasoning"],
    },
}

st.title("Deep Research Agent")


# Display old messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
if prompt := st.chat_input("Type your query here..."):
    # Save + display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    reason_placeholder = None
    reason_status = None
    msg_accum = ""
    reason_accum = ""
    reason_streaming = False

    final_response = None

    placeholder_dict = {}
    update_text = ""
    state = {"messages": [HumanMessage(content=prompt)]}

    # Assistant Updates
    # update_container = st.chat_message("assistant")
    # with update_container:
    #     status_container = st.status(label="Updates", expanded=True)

    update_container = None
    status_container = None
    generating_report = False

    for payload in run_async_generator(
        graph_stream_generator(
            agent=st.session_state.agent, state=state, config=config, subgraphs=True
        )
    ):
        if payload["content_type"] == ContentType.ASSISTANT_MESSAGE:
            if status_container:
                status_container.update(
                    label="Finished", state="complete", expanded=True
                )
            with st.chat_message("assistant"):
                st.markdown(payload["content"])
                st.session_state.messages.append(
                    {"role": "assistant", "content": payload["content"]}
                )
            update_container = None
            status_container = None
        else:
            if update_container is None:
                update_container = st.chat_message("assistant")
                with update_container:
                    status_container = st.status(label="Updates", expanded=True)

            if update_container and status_container:

                # Update to show tool is called
                if payload["content_type"] == ContentType.TOOL_CALLED:

                    with status_container:
                        placeholder = st.empty()
                        text_content = payload["content"]
                        if payload["tool_name"] == "think_tool":
                            placeholder.markdown(f":gray[:small[ 💭 *{text_content}*]]")
                        elif payload["tool_name"] == "research_complete":
                            generating_report = True
                        else:
                            placeholder.markdown(f":gray[:small[ ⏳ *{text_content}*]]")
                            placeholder_dict[
                                f"{payload["tool_name"]}_{payload["agent_type"]}"
                            ] = {
                                "text_content": text_content,
                                "placeholder": placeholder,
                            }

                    status_container.update(state="running")

                # Update status of tool called to finished
                if payload["content_type"] == ContentType.TOOL_CALL_COMPLETE:
                    key_name = f"{payload["tool_name"]}_{payload["agent_type"]}"
                    if key_name in placeholder_dict:
                        placeholder_dict[key_name]["placeholder"].markdown(
                            f":gray[:small[  ✅ *{placeholder_dict[key_name]["text_content"]}*]]"
                        )
                        del placeholder_dict[key_name]

                # Start Stream reason
                if payload["content_type"] in [
                    ContentType.STREAMING_REASON,
                    ContentType.START_STREAM_REASON,
                ]:
                    if payload["content_type"] == ContentType.START_STREAM_REASON:
                        with status_container:
                            reason_status = st.status(
                                label="🧠 Thinking...", state="running", expanded=True
                            )
                            with reason_status:
                                reason_placeholder = st.empty()

                    reason_accum += payload["content"]
                    html_block = f"""
                                    <div style="
                                        color: gray;
                                        font-size: 0.85em;
                                        font-style: italic;
                                        white-space: pre-wrap;
                                    ">{reason_accum}</div>
                                """
                    with reason_status:
                        reason_placeholder.markdown(html_block, unsafe_allow_html=True)

                if payload["content_type"] == ContentType.STOP_STREAM_REASON:
                    reason_accum = ""
                    reason_status.update(
                        label="🧠 Thoughts", state="complete", expanded=False
                    )

                # Final get final state response
                if payload["content_type"] == "response":

                    final_response = payload["content"]
