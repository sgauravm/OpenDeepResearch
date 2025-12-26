import streamlit as st
import uuid
import dotenv
from src.deep_research_agent.agents.research_agent import ResearcherAgent
from src.deep_research_agent.agents.scoping_agent import ScopingAgent
from langchain_core.messages import HumanMessage
from src.utils.stream import (
    response_generator_emulator,
    graph_stream_generator,
    run_async_generator,
)
from src.utils.models import get_model

dotenv.load_dotenv()

# # For emulator
# model = get_model(reasoning="low")

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
        # Container for all updates (they stay)
        # Collapsible container for all updates
        with st.expander("Updates", expanded=True):
            updates_container = st.container()
        # updates_container = st.container()
        # Placeholder for the streaming final message (single line, updated in place)
        message_placeholder = st.empty()
        msg_accum = ""

        final_response = None
        state = {"messages": [HumanMessage(content=prompt)]}
        for payload in run_async_generator(
            graph_stream_generator(agent, state, config)
        ):
            if payload["content_type"] == "updates":
                # Append each update as a new line in the updates container
                with updates_container:
                    st.markdown(f":gray[:small[ -  *{payload['content']}*]]")
            elif payload["content_type"] == "stream_messages":
                msg_accum += payload["content"]
                # Stream the main message below the updates, in a single element
                message_placeholder.markdown(msg_accum)
            elif payload["content_type"] == "response":
                final_response = payload["content"]

            # Optionally store final assistant message in history
            st.session_state.messages.append(
                {"role": "assistant", "content": msg_accum}
            )
