from src.deep_research_agent.agents.research_agent import ResearcherAgent
from src.utils.stream import graph_stream_print
from langchain_core.messages import HumanMessage
import asyncio
import dotenv
from src.config import RESEARCHER_AGENT_CONFIG

dotenv.load_dotenv()


# Example brief
research_brief = """
I want a curated list of notable coffee shops across all neighborhoods in Toronto, including each shop’s address, operating hours, signature specialty drinks, typical price range, overall ambiance, and aggregated customer reviews and ratings. I have no specific cost or location constraints, so consider all price ranges and all neighborhoods. Please gather this information from primary or reputable sources (e.g., official shop websites, trusted review platforms like Yelp or Google Reviews) and present it in a clear, organized format (table or summary).
"""
research_brief = """
I want to identify the top 5 tourist spots in Toronto, taking into account factors such as visitor popularity, cultural significance, accessibility, and overall visitor experience. I have not specified any constraints on budget, time, or particular interests, so all price ranges and time frames should be considered. Please use official tourism websites (e.g., Tourism Toronto), reputable travel guides (e.g., Lonely Planet, TripAdvisor reviews), and recent visitor reviews as primary sources for this research.
"""

interleaved_thinking = True
agent_reasoning = "low"
research_agent = ResearcherAgent(
    interleaved_thinking=interleaved_thinking, agent_reasoning=agent_reasoning
).build_agent_graph()

config = {
    "configurable": {
        "thread_id": "1",
        "max_web_search_calls": 4,
        "max_web_search_results": 3,
        "max_llm_call_retry": 3,
    },
}

state = {
    "researcher_messages": [HumanMessage(content=research_brief)],
}

result = asyncio.run(
    graph_stream_print(
        agent=research_agent, state=state, config=config, subgraphs=False
    )
)

if result and "compressed_research" in result:
    print("\n\n=== COMPRESSED RESEARCH OUTPUT ===\n")
    print(result["compressed_research"])
