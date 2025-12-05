import os
from src.research_planning_offloading_agent.state import DeepAgentState
from langchain.agents import create_agent
from src.config import ROOT_DIR
from src.research_planning_offloading_agent.tools.file_tools import (
    ls,
    read_file,
    write_file,
)
from src.research_planning_offloading_agent.tools.todo_tools import (
    write_todos,
    read_todos,
)
from src.research_planning_offloading_agent.tools.task_tools import create_task_tool
from src.research_planning_offloading_agent.tools.search_tools import web_search_tool
from src.research_planning_offloading_agent.tools.common_tools import think_tool
from src.utils.models import MODEL
from src.utils.helpers import get_prompt_template, get_today_str


TODO_USAGE_INSTRUCTION_TEMPLATE = get_prompt_template(
    os.path.join(
        ROOT_DIR,
        "src/research_planning_offloading_agent/prompts/todo_usage_instruction.jinja",
    )
)

FILE_USAGE_INSTRUCTION_TEMPLATE = get_prompt_template(
    os.path.join(
        ROOT_DIR,
        "src/research_planning_offloading_agent/prompts/file_usage_instruction.jinja",
    )
)

SUBAGENT_USAGE_INSTRUCTION_TEMPLATE = get_prompt_template(
    os.path.join(
        ROOT_DIR,
        "src/research_planning_offloading_agent/prompts/subagent_instruction.jinja",
    )
)
RESEARCHER_INSTRUCTIONS_TEMPLATE = get_prompt_template(
    os.path.join(
        ROOT_DIR,
        "src/research_planning_offloading_agent/prompts/researcher_instructions.jinja",
    )
)


research_sub_agent = {
    "name": "research-agent",
    "description": "Delegate research to the sub-agent researcher. Only give this researcher one topic at a time.",
    "prompt": RESEARCHER_INSTRUCTIONS_TEMPLATE.render(date=get_today_str()),
    "tools": ["web_search_tool", "think_tool"],
}

built_in_tools = [ls, read_file, write_file, write_todos, read_todos]
sub_agent_tools = [web_search_tool, think_tool]
# Create task tool to delegate tasks to sub-agents
task_tool = create_task_tool(
    sub_agent_tools, [research_sub_agent], MODEL, DeepAgentState
)
delegation_tools = [task_tool]
all_tools = sub_agent_tools + built_in_tools + delegation_tools

# Limits
max_concurrent_research_units = 3
max_researcher_iterations = 3

SUBAGENT_INSTRUCTIONS = SUBAGENT_USAGE_INSTRUCTION_TEMPLATE.render(
    max_concurrent_research_units=max_concurrent_research_units,
    max_researcher_iterations=max_researcher_iterations,
    date=get_today_str(),
)

INSTRUCTIONS = (
    "# TODO MANAGEMENT\n"
    + TODO_USAGE_INSTRUCTION_TEMPLATE.render()
    + "\n\n"
    + "=" * 80
    + "\n\n"
    + "# FILE SYSTEM USAGE\n"
    + FILE_USAGE_INSTRUCTION_TEMPLATE.render()
    + "\n\n"
    + "=" * 80
    + "\n\n"
    + "# SUB-AGENT DELEGATION\n"
    + SUBAGENT_INSTRUCTIONS
)

agent = create_agent(
    MODEL, all_tools, system_prompt=INSTRUCTIONS, state_schema=DeepAgentState
)
