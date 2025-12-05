from typing_extensions import Annotated
import ollama
from ollama import WebSearchResponse
from src.utils.models import MODEL
from src.utils.helpers import get_prompt_template, get_today_str
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import InjectedToolArg, InjectedToolCallId, tool
import uuid, base64
import os
from src.research_planning_offloading_agent.state import DeepAgentState
from langgraph.prebuilt import InjectedState

from langgraph.types import Command
from src.config import ROOT_DIR

SUMMARIZE_WEB_SEARCH_TEMPLATE = get_prompt_template(
    os.path.join(
        ROOT_DIR,
        "src/research_planning_offloading_agent/prompts/summarize_web_search.jinja",
    )
)


class Summary(BaseModel):
    """Schema for webpage content summarization."""

    filename: str = Field(description="Name of the file to store.")
    summary: str = Field(description="Key learnings from the webpage.")


def summarize_webpage_content(webpage_content: str) -> Summary:
    """Summarize webpage content using the configured summarization model.

    Args:
        webpage_content: Raw webpage content to summarize

    Returns:
        Summary object with filename and summary
    """
    try:
        # Set up structured output model for summarization
        structured_model = MODEL.with_structured_output(Summary)

        # Generate summary
        summary_and_filename = structured_model.invoke(
            [
                HumanMessage(
                    content=SUMMARIZE_WEB_SEARCH_TEMPLATE.render(
                        webpage_content=webpage_content, date=get_today_str()
                    )
                )
            ]
        )

        return summary_and_filename

    except Exception as e:
        print(f"Error during summarization: {e}")
        # Return a basic summary object on failure
        return Summary(
            filename="search_result.md",
            summary=(
                webpage_content[:1000] + "..."
                if len(webpage_content) > 1000
                else webpage_content
            ),
        )


def process_search_results(results: WebSearchResponse) -> list[dict]:
    """Process search results by summarizing content where available.

    Args:
        results: Tavily search results dictionary

    Returns:
        List of processed results with summaries
    """
    processed_results = []

    for result in results.results:

        summary_obj = summarize_webpage_content(result.content)

        # uniquify file names
        uid = (
            base64.urlsafe_b64encode(uuid.uuid4().bytes)
            .rstrip(b"=")
            .decode("ascii")[:8]
        )
        name, ext = os.path.splitext(summary_obj.filename)
        summary_obj.filename = f"{name}_{uid}{ext}"

        processed_results.append(
            {
                "url": result.url,
                "title": result.title,
                "summary": summary_obj.summary,
                "filename": summary_obj.filename,
                "raw_content": result.content,
            }
        )

    return processed_results


@tool(parse_docstring=True)
def web_search_tool(
    query: str,
    state: Annotated[DeepAgentState, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
    max_results: Annotated[int, InjectedToolArg] = 1,
) -> Command:
    """Search web and save detailed results to files while returning minimal context.

    Performs web search and saves full content to files for context offloading.
    Returns only essential information to help the agent decide on next steps.

    Args:
        query: Search query to execute
        state: Injected agent state for file storage
        tool_call_id: Injected tool call identifier
        max_results: Maximum number of results to return (default: 1)

    Returns:
        Command that saves full results to files and provides minimal summary
    """
    # Execute search
    search_results = ollama.web_search(query, max_results=max_results)

    # Process and summarize results
    processed_results = process_search_results(search_results)

    # Save each result to a file and prepare summary
    files = state.get("files", {})
    saved_files = []
    summaries = []

    for i, result in enumerate(processed_results):
        # Use the AI-generated filename from summarization
        filename = result["filename"]

        # Create file content with full details
        file_content = f"""# Search Result: {result['title']}

**URL:** {result['url']}
**Query:** {query}
**Date:** {get_today_str()}

## Summary
{result['summary']}

## Raw Content
{result['raw_content'] if result['raw_content'] else 'No raw content available'}
"""

        files[filename] = file_content
        saved_files.append(filename)
        summaries.append(f"- {filename}: {result['summary']}...")

    # Create minimal summary for tool message - focus on what was collected
    summary_text = f"""🔍 Found {len(processed_results)} result(s) for '{query}':

{chr(10).join(summaries)}
"""

    return Command(
        update={
            "files": files,
            "messages": [ToolMessage(summary_text, tool_call_id=tool_call_id)],
        }
    )
