from langchain.tools import ToolRuntime, tool
from langchain_core.messages import filter_messages, BaseMessage
from langgraph.types import Command
from langgraph.graph import END
from langgraph.config import get_stream_writer


@tool
async def research_complete(runtime: ToolRuntime):
    """Use this tool to indicate that research is finished."""
    writer = get_stream_writer()
    writer(
        {
            "message_type": "tool_info",
            "message": f"Research completed.",
        }
    )
    return {}
