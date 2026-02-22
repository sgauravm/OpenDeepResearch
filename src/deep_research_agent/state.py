import operator
from typing import TypedDict
from typing_extensions import Optional, Annotated, Sequence

from langchain_core.messages import BaseMessage
from langgraph.graph import MessagesState
from langgraph.graph.message import add_messages


class AgentInputState(MessagesState):
    """Input state for the full agent - only contains messages from user input."""

    pass


class AgentState(MessagesState):
    """
    Main state for the full multi-agent research system.

    Extends MessagesState with additional fields for research coordination.
    Note: Some fields are duplicated across different state classes for proper
    state management between subgraphs and the main workflow.
    """

    # Research brief generated from user conversation history
    research_brief: Optional[str]
    # Messages exchanged with the supervisor agent for coordination
    supervisor_messages: Annotated[Sequence[BaseMessage], add_messages]
    # Raw unprocessed research notes collected during the research phase
    raw_notes: Annotated[list[str], operator.add] = []
    # Processed and structured notes ready for report generation
    notes: Annotated[list[str], operator.add] = []
    # Final formatted research report
    final_report: str
    research_notes: Annotated[dict[str, dict[str, dict]], add_dict]
    visited_urls: Annotated[list[str], operator.add]


def add_dict(dict1: dict, dict2: dict) -> dict:
    return {**dict1, **dict2}


class ResearcherState(TypedDict):
    """
    State for the research agent containing message history and research metadata.

    This state tracks the researcher's conversation, iteration count for limiting
    tool calls, the research topic being investigated, compressed findings,
    and raw research notes for detailed analysis.
    """

    researcher_messages: Annotated[Sequence[BaseMessage], add_messages]
    tool_call_iterations: int
    research_topic: str
    compressed_research: str
    visited_urls: Annotated[list[str], operator.add]
    num_web_search_calls: int
    num_retry_llm_call_node: int
    is_llm_call_error: bool
    research_notes: Annotated[dict[str, dict[str, dict]], add_dict]


class ResearcherOutputState(TypedDict):
    """
    Output state for the research agent containing final research results.

    This represents the final output of the research process with compressed
    research findings and all raw notes from the research process.
    """

    compressed_research: str
    researcher_messages: Annotated[Sequence[BaseMessage], add_messages]
    research_notes: Annotated[dict[str, dict[str, dict]], add_dict]
    visited_urls: Annotated[list[str], operator.add]


class SupervisorState(TypedDict):
    """
    State for the multi-agent research supervisor.

    Manages coordination between supervisor and research agents, tracking
    research progress and accumulating findings from multiple sub-agents.
    """

    # Messages exchanged with supervisor for coordination and decision-making
    supervisor_messages: Annotated[Sequence[BaseMessage], add_messages]
    # Detailed research brief that guides the overall research direction
    research_brief: str

    # Counter tracking the number of research iterations performed
    research_iterations: int = 0

    num_retry_llm_call_node: int
    is_llm_call_error: bool
    research_iterations: int
    research_notes: Annotated[dict[str, dict[str, dict]], add_dict]
    visited_urls: Annotated[list[str], operator.add]


class ResearchWriterState(TypedDict):
    """
    State for the research writer agent containing research brief, research notes, section writing plan, section texts, and final research text.
    """

    research_brief: str
    research_notes: Annotated[dict[str, dict[str, dict]], add_dict]
    section_texts: Annotated[list[str], operator.add]
    final_research_text: str
    report_plan: Annotated[list[dict], operator.add]
    current_section_index: int
