import streamlit as st
import uuid
import dotenv
from src.deep_research_agent.agents.research_agent import ResearcherAgent
from src.deep_research_agent.agents.scoping_agent import ScopingAgent
from src.deep_research_agent.agents.final_deep_research_agent import DeepResearchAgent
from langchain_core.messages import HumanMessage
from src.utils.stream import (
    response_generator_emulator,
    graph_stream_generator,
    run_async_generator,
)
from src.utils.models import get_model
from src.config import FINAL_AGENT_CONFIG
import asyncio

dotenv.load_dotenv()


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

    # Assistant response
    with st.chat_message("assistant"):
        state = {"messages": [HumanMessage(content=prompt)]}
        with st.spinner("🧠 Thinking..."):
            result = asyncio.run(st.session_state.agent.ainvoke(state, config=config))
        if result.get("final_report", "") != "":
            assistant_response = result.get("final_report", "")
        else:
            assistant_response = result.get("messages")[-1].content
        st.markdown(assistant_response)
        st.session_state.messages.append(
            {"role": "assistant", "content": assistant_response}
        )
