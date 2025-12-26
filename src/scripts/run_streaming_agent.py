from src.deep_research_agent.agents.research_agent import ResearcherAgent
from src.deep_research_agent.agents.supervisor_agent import SupervisorResearchAgent
from src.utils.stream import graph_stream_print
from langchain_core.messages import HumanMessage
from src.config import SUPERVISOR_AGENT_CONFIG
import asyncio
import dotenv
from src.config import FINAL_AGENT_CONFIG

dotenv.load_dotenv()


# Example brief
research_brief = """
I want a curated list of notable coffee shops across all neighborhoods in Toronto, including each shop’s address, operating hours, signature specialty drinks, typical price range, overall ambiance, and aggregated customer reviews and ratings. I have no specific cost or location constraints, so consider all price ranges and all neighborhoods. Please gather this information from primary or reputable sources (e.g., official shop websites, trusted review platforms like Yelp or Google Reviews) and present it in a clear, organized format (table or summary).
"""

interleaved_thinking = True
agent_reasoning = "low"
research_supervisor = SupervisorResearchAgent(
    interleaved_thinking=interleaved_thinking, agent_reasoning=agent_reasoning
).build_agent_graph()

config = {
    "configurable": {
        "thread_id": "1",
        "max_web_search_calls": FINAL_AGENT_CONFIG["max_web_search_calls"],
        "max_web_search_results": FINAL_AGENT_CONFIG["max_web_search_results"],
        "max_llm_call_retry": FINAL_AGENT_CONFIG["max_llm_call_retry"],
        "max_researcher_iterations": FINAL_AGENT_CONFIG["max_researcher_iterations"],
        "max_concurrent_researchers": FINAL_AGENT_CONFIG["max_concurrent_researchers"],
        "interleaved_thinking": FINAL_AGENT_CONFIG["interleaved_thinking"],
        "agent_reasoning": FINAL_AGENT_CONFIG["agent_reasoning"],
    },
}

state = {
    "supervisor_messages": [HumanMessage(content=research_brief)],
}

result = asyncio.run(
    graph_stream_print(
        agent=research_supervisor, state=state, config=config, subgraphs=True
    )
)
