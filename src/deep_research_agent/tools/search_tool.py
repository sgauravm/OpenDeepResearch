import os
from typing import Annotated, List, Literal

import ollama
from ollama import WebSearchResponse
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage
from langchain_core.tools import InjectedToolArg, tool
from src.utils.models import get_model
from src.config import ROOT_DIR
from src.utils.helpers import get_prompt_template, get_today_str


SUMMARIZE_WEBPAGE_PROMPT_TEMPLATE = get_prompt_template(
    os.path.join(
        ROOT_DIR,
        "src/deep_research_agent/prompts/summarize_webpage_prompt.jinja",
    )
)
MODEL = get_model()


class Summary(BaseModel):
    """Schema for webpage content summarization."""

    summary: str = Field(description="Concise summary of the webpage content")
    key_excerpts: str = Field(
        description="Important quotes and excerpts from the content"
    )


def ollama_search_multiple(
    search_queries: List[str], max_results: int = 3
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

    # Execute searches sequentially. Note: yon can use AsyncTavilyClient to parallelize this step.
    search_docs = []
    for query in search_queries:
        result = ollama.web_search(query, max_results=max_results)
        search_docs.append(result)

    return search_docs


def summarize_webpage_content(webpage_content: str) -> str:
    """Summarize webpage content using the configured summarization model.

    Args:
        webpage_content: Raw webpage content to summarize

    Returns:
        Formatted summary with key excerpts
    """
    try:
        # Set up structured output model for summarization
        structured_model = MODEL.with_structured_output(Summary)

        # Generate summary
        summary = structured_model.invoke(
            [
                HumanMessage(
                    content=SUMMARIZE_WEBPAGE_PROMPT_TEMPLATE.render(
                        webpage_content=webpage_content, date=get_today_str()
                    )
                )
            ]
        )

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


def process_search_results(unique_results: dict) -> dict:
    """Process search results by summarizing content where available.

    Args:
        unique_results: Dictionary of unique search results

    Returns:
        Dictionary of processed results with summaries
    """
    summarized_results = {}

    for url, result in unique_results.items():

        content = summarize_webpage_content(result.content)

        summarized_results[url] = {"title": result.title, "content": content}

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


@tool(parse_docstring=True)
def web_search(
    query: str,
    max_results: Annotated[int, InjectedToolArg] = 3,
) -> str:
    """Fetch results from ollama web search API with content summarization.

    Args:
        query: A single search query to execute
        max_results: Maximum number of results to return

    Returns:
        Formatted string of search results with summaries
    """
    # Execute search for single query
    search_results = ollama_search_multiple(
        [query],  # Convert single query to list for the internal function
        max_results=max_results,
    )

    # Deduplicate results by URL to avoid processing duplicate content
    unique_results = deduplicate_search_results(search_results)

    # Process results with summarization
    summarized_results = process_search_results(unique_results)

    # Format output for consumption
    return format_search_output(summarized_results)
