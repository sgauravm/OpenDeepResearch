from langchain_core.tools import tool
from langgraph.config import get_stream_writer
from pydantic import BaseModel, Field


class ThinkInput(BaseModel):
    reflection: str = Field(
        description=(
            "Your detailed reflection on research progress, findings, gaps, and next steps decision whether to continue searching or stop based on `Hard Limits` provided in system prompt"
        )
    )


@tool(args_schema=ThinkInput)
def think_tool(reflection: ThinkInput) -> str:
    """
    STRICT JSON TOOL.
    Tool for strategic reflection on research progress and decisions.
    Use this tool after each web_search to pause, analyze results, and plan next steps.

    Output rules (MANDATORY):
    - Call this tool with ONLY valid JSON
    - Do NOT include explanations, reasoning, or extra text
    - JSON must match exactly:
      {"reflection": "<string>"}
    """
    writer = get_stream_writer()
    writer(
        {
            "type": "tool_info",
            "message": f"Researcher Reflection: {reflection}",
            "tool_name": "think_tool",
        }
    )
    return f"Reflection recorded: {reflection}"
