from src.utils.helpers import get_prompt_template
from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolCallId, tool
import os
from src.research_planning_offloading_agent.state import DeepAgentState, Todo
from typing import Annotated
from langgraph.prebuilt import InjectedState
from langchain.agents import create_agent
from langgraph.types import Command
from typing_extensions import TypedDict
from src.utils.notebook_utils import format_messages, stream_agent
from src.config import ROOT_DIR


WRITE_TODO_DESCRIPTION_TEMPLATE = get_prompt_template(
    os.path.join(
        ROOT_DIR,
        "src/research_planning_offloading_agent/prompts/write_todo_description.jinja",
    )
)


TODO_USAGE_INSTRUCTION_TEMPLATE = get_prompt_template(
    os.path.join(
        ROOT_DIR,
        "src/research_planning_offloading_agent/prompts/todo_usage_instruction.jinja",
    )
)


@tool(description=WRITE_TODO_DESCRIPTION_TEMPLATE.render(), parse_docstring=True)
def write_todos(
    todos: list[Todo], tool_call_id: Annotated[str, InjectedToolCallId]
) -> Command:
    """Create or update the agent's TODO list for task planning and tracking.

    Args:
        todos: List of Todo items with content and status
        tool_call_id: Tool call identifier for message response

    Returns:
        Command to update agent state with new TODO list
    """
    return Command(
        update={
            "todos": todos,
            "messages": [
                ToolMessage(f"Updated todo list to {todos}", tool_call_id=tool_call_id)
            ],
        }
    )


@tool(parse_docstring=True)
def read_todos(
    state: Annotated[DeepAgentState, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> str:
    """Read the current TODO list from the agent state.

    This tool allows the agent to retrieve and review the current TODO list
    to stay focused on remaining tasks and track progress through complex workflows.

    Args:
        state: Injected agent state containing the current TODO list
        tool_call_id: Injected tool call identifier for message tracking

    Returns:
        Formatted string representation of the current TODO list
    """
    todos = state.get("todos", [])
    if not todos:
        return "No todos currently in the list."

    result = "Current TODO List:\n"
    for i, todo in enumerate(todos, 1):
        status_emoji = {"pending": "⏳", "in_progress": "🔄", "completed": "✅"}
        emoji = status_emoji.get(todo["status"], "❓")
        result += f"{i}. {emoji} {todo['content']} ({todo['status']})\n"

    return result.strip()
