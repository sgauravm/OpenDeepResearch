from langchain.tools import ToolRuntime, tool
from langchain_core.messages import HumanMessage, ToolMessage
from pydantic import BaseModel, Field
from src.deep_research_agent.agents.research_agent import ResearcherAgent
from src.config import RESEARCHER_AGENT_CONFIG
from langgraph.types import Command
from langgraph.config import get_stream_writer


class ResearchTopic(BaseModel):
    research_topic: str = Field(
        description="The topic to research. Should be a single topic, and should be described in high detail (at least a paragraph).",
    )


@tool(
    description="Tool for delegating a research task to a specialized sub-agent.",
    args_schema=ResearchTopic,
)
async def conduct_research(
    research_topic: str,
    runtime: ToolRuntime,
):
    writer = get_stream_writer()
    writer(
        {
            "message_type": "tool_info",
            "message": f"Calling researcher subagent with topic: {research_topic}",
        }
    )

    config = runtime.config.get("configurable", {})
    researcher_agent = ResearcherAgent(
        interleaved_thinking=config.get("interleaved_thinking", True),
        agent_reasoning=config.get("agent_reasoning", "low"),
    ).build_agent_graph()

    researcher_state = {
        "researcher_messages": [HumanMessage(content=research_topic)],
        "research_topic": research_topic,
    }

    research_iterations = runtime.state.get("research_iterations", 0)
    if research_iterations >= config.get("max_researcher_iterations", 5):
        return f"Maximum number of conduct research calls ({research_iterations}) reached. Cannot perform more searches. Now complete the research process."
    try:
        result = await researcher_agent.ainvoke(input=researcher_state)
        tool_message = result.get(
            "compressed_research", "Error synthesizing research report"
        )
        raw_notes = result.get("raw_notes", [])

    except Exception as e:
        tool_message = (
            f"The following error occured while conducting research: {str(e)}"
        )
        raw_notes = []

    return Command(
        update={
            "supervisor_messages": [
                ToolMessage(
                    content=tool_message,
                    tool_name="conduct_research",
                    tool_call_id=runtime.tool_call_id,
                )
            ],
            "raw_notes": raw_notes,
            "research_iterations": research_iterations + 1,
        }
    )
