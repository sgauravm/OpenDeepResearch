import asyncio
import os
from typing import Annotated, Any, List, Literal

import ollama
from ollama import WebSearchResponse
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, ToolMessage
from langgraph.types import Command
from langchain.tools import ToolRuntime, tool
from src.utils.models import get_model
from src.utils.helpers import split_text_by_words
from src.config import ROOT_DIR
from src.utils.helpers import get_prompt_template, get_today_str
from langgraph.config import get_stream_writer


SUMMARIZE_WEBPAGE_PROMPT_TEMPLATE = get_prompt_template(
    os.path.join(
        ROOT_DIR,
        "src/deep_research_agent/prompts/summarize_webpage_prompt.jinja",
    )
)
MODEL = get_model()


class Summary(BaseModel):
    """Schema for webpage content summarization."""

    summary: str = Field(
        description="Concise summary of the webpage content. Strictly should not be less than 300 words. Do not repeat or loop text."
    )
    key_excerpts: str = Field(
        description="Important quotes and excerpts from the content. Strictly should not be less than 300 words. Do not repeat or loop text."
    )


def ollama_search_multiple(
    search_queries: List[str], max_results: int = 2
) -> List[dict]:
    """Perform search using Ollama web search API for multiple queries.

    Args:
        search_queries: List of search queries to execute
        max_results: Maximum number of results per query
        topic: Topic filter for search results
        include_raw_content: Whether to include raw webpage content

    Returns:
        List of search result dictionaries
    """

    search_docs = []
    for query in search_queries:
        result = ollama.web_search(query, max_results=max_results)
        search_docs.append(result)

    return search_docs


async def summarize_chunk(chunk: str):
    structured_model = MODEL.with_structured_output(Summary)
    return await structured_model.ainvoke(
        [
            HumanMessage(
                content=SUMMARIZE_WEBPAGE_PROMPT_TEMPLATE.render(
                    webpage_content=chunk, date=get_today_str()
                )
            )
        ]
    )


async def summarize_long_content(
    content: str, chunk_size=2000, overlap_size=100
) -> Summary:
    chunks = split_text_by_words(content, chunk_size, overlap_size)
    structured_model = MODEL.with_structured_output(Summary)

    partials = await asyncio.gather(*[summarize_chunk(chunk) for chunk in chunks])

    combined_summary = "\n".join(p.summary for p in partials)
    combined_excerpts = "\n".join(p.key_excerpts for p in partials)

    return Summary(summary=combined_summary, key_excerpts=combined_excerpts)


async def summarize_webpage_content(webpage_content: str) -> str:
    """Summarize webpage content using the configured summarization model.

    Args:
        webpage_content: Raw webpage content to summarize

    Returns:
        Formatted summary with key excerpts
    """
    try:
        webpage_content = " ".join(webpage_content.split(" ")[0:8000])
        # Set up structured output model for summarization
        writer = get_stream_writer()
        writer(
            {
                "message_type": "tool_info",
                "message": f"Summarizing web content of length {len(webpage_content.split())} words.",
            }
        )

        # Generate summary
        summary = await summarize_long_content(webpage_content)

        # Format summary with clear structure
        formatted_summary = (
            f"<summary>\n{summary.summary}\n</summary>\n\n"
            f"<key_excerpts>\n{summary.key_excerpts}\n</key_excerpts>"
        )

        return formatted_summary

    except Exception as e:
        print(f"Failed to summarize webpage: {str(e)}")
        return (
            webpage_content[:1000] + "..."
            if len(webpage_content) > 1000
            else webpage_content
        )


def deduplicate_search_results(search_results: List[WebSearchResponse]) -> dict:
    """Deduplicate search results by URL to avoid processing duplicate content.

    Args:
        search_results: List of search result dictionaries

    Returns:
        Dictionary mapping URLs to unique results
    """
    unique_results = {}

    for response in search_results:
        for result in response.results:
            url = result.url
            if url not in unique_results:
                unique_results[url] = result

    return unique_results


async def process_search_results(unique_results: dict[str, Any]) -> dict[str, Any]:
    """Process search results by summarizing content in parallel."""

    async def process_single(url: str, result: Any):
        content = await summarize_webpage_content(result.content)
        return url, {"title": result.title, "content": content}

    # Create a list of coroutines
    tasks = [process_single(url, result) for url, result in unique_results.items()]

    # Run all tasks concurrently
    results = await asyncio.gather(*tasks)

    # Convert list of tuples back to dict
    summarized_results = dict(results)

    return summarized_results


def format_search_output(summarized_results: dict) -> str:
    """Format search results into a well-structured string output.

    Args:
        summarized_results: Dictionary of processed search results

    Returns:
        Formatted string of search results with clear source separation
    """
    if not summarized_results:
        return "No valid search results found. Please try different search queries or use a different search API."

    formatted_output = "Search results: \n\n"

    for i, (url, result) in enumerate(summarized_results.items(), 1):
        formatted_output += f"\n\n--- SOURCE {i}: {result['title']} ---\n"
        formatted_output += f"URL: {url}\n\n"
        formatted_output += f"SUMMARY:\n{result['content']}\n\n"
        formatted_output += "-" * 80 + "\n"

    return formatted_output


def _get_non_visited_urls(visted_urls, search_results):
    for url in visted_urls:
        if url in search_results:
            del search_results[url]
    return search_results


class ToolRuntimeModel(BaseModel):
    tool_call_id: str
    state: dict = {}


@tool(parse_docstring=True)
async def web_search(
    query: str,
    runtime: ToolRuntime,
):
    """Fetch results from ollama web search API with content summarization.

    Args:
        query: A single search query to execute

    Returns:
        Formatted string of search results with summaries
    """
    # Getting context data
    max_results = runtime.config.get("configurable", {}).get(
        "max_web_search_results", 1
    )
    max_web_search_calls = runtime.config.get("configurable", {}).get(
        "max_web_search_calls", 5
    )
    num_web_search_calls = runtime.state.get("num_web_search_calls", 0)

    num_web_search_calls += 1
    # Checking for max tool call limit
    if num_web_search_calls > max_web_search_calls:
        return f"Maximum number of web search calls ({max_web_search_calls}) reached. Cannot perform more searches. Answer based on the information collected so far."

    writer = get_stream_writer()
    writer(
        {
            "message_type": "tool_info",
            "message": f"Executing web search for query: {query}",
        }
    )
    # Execute search for single query
    search_results = ollama_search_multiple(
        [query],  # Convert single query to list for the internal function
        max_results=max_results,
    )

    # Deduplicate results by URL to avoid processing duplicate content
    unique_results = deduplicate_search_results(search_results)

    # Exclude already visited URLs
    unique_result = _get_non_visited_urls(
        runtime.state.get("visited_urls", []), unique_results
    )

    if len(unique_result) == 0:
        tool_message = "No new search results found. All URLs have been previously visited. Try a different query."
        return Command(
            update={
                "researcher_messages": [
                    ToolMessage(
                        content=tool_message,
                        name="web_search",
                        tool_call_id=runtime.tool_call_id,
                    )
                ],
                "num_web_search_calls": num_web_search_calls,
            }
        )

    # Process results with summarization
    summarized_results = await process_search_results(unique_results)

    # Format output for consumption
    tool_message = format_search_output(summarized_results)

    return Command(
        update={
            "visited_urls": list(summarized_results.keys()),
            "researcher_messages": [
                ToolMessage(
                    content=tool_message,
                    name="web_search",
                    tool_call_id=runtime.tool_call_id,
                )
            ],
            "num_web_search_calls": num_web_search_calls,
        }
    )
