import os
from src.research_planning_offloading_agent.state import DeepAgentState
from langchain.agents import create_agent
from src.research_planning_offloading_agent.state import DeepAgentState
from src.research_planning_offloading_agent.state import DeepAgentState
from src.research_planning_offloading_agent.tools.search_tools import web_search_tool
from src.research_planning_offloading_agent.tools.common_tools import think_tool
from src.utils.helpers import get_prompt_template, get_today_str
from src.utils.models import MODEL
from src.config import ROOT_DIR

RESEARCHER_INSTRUCTIONS_TEMPLATE = get_prompt_template(
    os.path.join(
        ROOT_DIR,
        "src/research_planning_offloading_agent/prompts/researcher_instructions.jinja",
    )
)

SUB_AGENT_TOOLS = [web_search_tool, think_tool]

researcher_agent = create_agent(
    MODEL,
    system_prompt=RESEARCHER_INSTRUCTIONS_TEMPLATE.render(date=get_today_str()),
    tools=SUB_AGENT_TOOLS,
    state_schema=DeepAgentState,
)
