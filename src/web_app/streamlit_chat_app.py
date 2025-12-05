import streamlit as st
from src.deep_research_agent.agents.scoping import scope_graph

st.title("Deep Research Agent")

thread = {"configurable": {"thread_id": "1"}}

if "messages" not in st.session_state:
    st.session_state.messages = []


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Type your query here..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        state = {"messages": [{"role": "user", "content": prompt}]}
        result = scope_graph.invoke(state, config=thread)
        response = st.markdown(result["messages"][-1].content)

    st.session_state.messages.append({"role": "assistant", "content": response})
