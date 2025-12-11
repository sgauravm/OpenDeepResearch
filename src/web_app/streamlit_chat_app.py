import streamlit as st
from src.deep_research_agent.agents.scoping import scope_graph
import uuid
import dotenv

dotenv.load_dotenv()

st.title("Deep Research Agent")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "thread" not in st.session_state:
    st.session_state.thread = {"configurable": {"thread_id": uuid.uuid4().hex[:8]}}

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
        # TODO: Add try except when agent is finalized
        # Build state for the agent
        state = {"messages": [{"role": "user", "content": prompt}]}

        # Invoke agent
        result = scope_graph.invoke(state, config=st.session_state.thread)

        # Extract assistant message (dict-safe)
        last_msg = result["messages"][-1]
        assistant_message = (
            last_msg["content"] if isinstance(last_msg, dict) else last_msg.content
        )

        st.markdown(assistant_message)

    # Save assistant message
    st.session_state.messages.append(
        {"role": "assistant", "content": assistant_message}
    )
