import random
import streamlit as st
import uuid
import dotenv
from src.deep_research_agent.agents.research_agent import ResearcherAgent
from src.deep_research_agent.agents.scoping_agent import ScopingAgent
from langchain_core.messages import HumanMessage
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

dotenv.load_dotenv()


######################################################
async def graph_stream_generator(
    agent: CompiledStateGraph, state: dict, config: RunnableConfig | None = None
):
    async for mode, chunk in agent.astream(
        input=state, stream_mode=["messages", "custom", "values"], config=config
    ):
        payload = None
        # Print custom values
        if mode == "custom":
            msg = chunk.get("message", None)
            if msg:
                payload = {"content": msg, "content_type": "updates"}

        # Print llm messages from selected nodes
        if mode == "messages":
            if chunk[0].content != "":

                payload = {
                    "content": chunk[0].content,
                    "content_type": "stream_messages",
                }

        # Save final response
        if mode == "values" and chunk:
            payload = {"content": chunk, "content_type": "response"}
        if payload:
            yield payload


##
async def response_generator_emulator(query: str, model: BaseChatModel | None = None):
    updates = [
        "Initializing system components...",
        "Loading configuration files...",
        "Establishing database connection...",
        "Running validation checks...",
        "Startup complete. System is ready.",
    ]

    response = random.choice(
        [
            "Hello there! How can I assist you today?",
            "Hi, human! Is there anything I can help you with?",
            "Do you need help?",
        ]
    )
    # Yield updates
    for i in range(len(updates)):
        update = random.choice(updates)
        payload = {"content": update, "content_type": "updates"}
        yield payload
        time.sleep(0.5)
    if model:
        async for chunk in model.astream(query):
            # Check if it has reasoning content
            if chunk.additional_kwargs.get("reasoning_content", "") != "":
                payload = {
                    "content": chunk.additional_kwargs.get("reasoning_content", ""),
                    "content_type": "stream_reason",
                    "chunk_position": chunk.chunk_position,
                }
                yield payload
            else:
                if chunk.content != "":
                    payload = {
                        "content": chunk.content,
                        "content_type": "stream_message",
                        "chunk_position": chunk.chunk_position,
                    }
                    yield payload

    else:
        for word in response.split():
            payload = {"content": word + " ", "content_type": "stream_messages"}
            yield payload
            time.sleep(0.05)


######################################################

# For emulator
model = get_model(reasoning="medium")

think_tool_required = st.sidebar.selectbox(
    "Do you want to include think tool?", ("Yes", "No")
)

reset_clicked = st.sidebar.button("New Chat")


interleaved_thinking = True if think_tool_required == "Yes" else False
# researcher_agent = ResearcherAgent(
#     interleaved_thinking=interleaved_thinking, agent_reasoning="medium"
# ).build_agent_graph()

agent = ScopingAgent().build_agent_graph()


# Initialize session state
if "messages" not in st.session_state or reset_clicked:
    st.session_state.messages = []

if "thread" not in st.session_state or reset_clicked:
    st.session_state.thread = uuid.uuid4().hex[:8]

config = {
    "configurable": {
        "thread_id": st.session_state.thread,
        "max_web_search_calls": 5,
        "max_web_search_results": 1,
        "max_llm_call_retry": 3,
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

    # Assistant response
    with st.chat_message("assistant"):

        status_container = st.status(label="Updates", expanded=True)
        message_placeholder = st.empty()
        reason_placeholder = None
        reason_status = None
        msg_accum = ""
        reason_accum = ""
        reason_streaming = False

        final_response = None
        state = {"messages": [HumanMessage(content=prompt)]}

        prev_placeholder = None
        update_text = ""
        for payload in run_async_generator(response_generator_emulator(prompt, model)):
            if prev_placeholder:
                prev_placeholder.markdown(f":gray[:small[  ✅ *Done {update_text}*]]")
                prev_placeholder = None

            if payload["content_type"] == "updates":
                # Append each update as a new line in the updates container
                with status_container:
                    cur_placeholder = st.empty()
                    if prev_placeholder:
                        prev_placeholder.markdown(
                            f":gray[:small[  ✅ *Done {update_text}*]]"
                        )
                        prev_placeholder = None
                    update_text = payload["content"]
                    cur_placeholder.markdown(f":gray[:small[ ⏳ *{update_text}*]]")
                    prev_placeholder = cur_placeholder
                status_container.update(state="running")

            elif payload["content_type"] == "stream_reason":
                with status_container:
                    if reason_placeholder is None:
                        reason_streaming = True
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
                        reason_placeholder.markdown(
                            html_block.strip(), unsafe_allow_html=True
                        )

            elif payload["content_type"] == "stream_message":
                if reason_streaming:
                    reason_streaming = False
                    reason_placeholder = None
                    reason_accum = ""
                    reason_status.update(
                        label="🧠 Thoughts", state="complete", expanded=False
                    )
                    reason_status = None
                msg_accum += payload["content"]
                message_placeholder.markdown(msg_accum)

            elif payload["content_type"] == "response":

                final_response = payload["content"]

            # Optionally store final assistant message in history
            st.session_state.messages.append(
                {"role": "assistant", "content": msg_accum}
            )

            if prev_placeholder:
                prev_placeholder.markdown(f":gray[:small[  ✅ *Done {update_text}*]]")
        status_container.update(label="Finished", state="complete", expanded=True)
