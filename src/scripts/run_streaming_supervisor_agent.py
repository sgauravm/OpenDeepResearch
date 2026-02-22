import json
from src.deep_research_agent.agents.research_agent import ResearcherAgent
from src.deep_research_agent.agents.supervisor_agent import SupervisorResearchAgent
from src.utils.stream import graph_stream_print
from langchain_core.messages import HumanMessage, BaseMessage
from src.config import SUPERVISOR_AGENT_CONFIG
import asyncio
import dotenv
from src.config import FINAL_AGENT_CONFIG

dotenv.load_dotenv()


def _make_json_serializable(obj):
    """Recursively convert result to JSON-serializable format (handles LangChain messages)."""
    if isinstance(obj, BaseMessage):
        return obj.model_dump()
    if isinstance(obj, dict):
        return {k: _make_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_json_serializable(item) for item in obj]
    return obj


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
        "max_web_search_calls": 4,
        "max_web_search_results": 3,
        "max_llm_call_retry": 3,
        "max_researcher_iterations": 3,
        "max_concurrent_researchers": 3,
        "interleaved_thinking": True,
        "agent_reasoning": "low",
    },
}

state = {
    "supervisor_messages": [HumanMessage(content=research_brief)],
    "visited_urls": [],
    "research_notes": {},
}

result = asyncio.run(
    graph_stream_print(
        agent=research_supervisor, state=state, config=config, subgraphs=True
    )
)

# # Save the result in to a json file
# with open("result.json", "w") as f:
#     json.dump(_make_json_serializable(result), f)
