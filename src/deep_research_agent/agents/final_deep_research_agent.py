from src.deep_research_agent.agents.supervisor_agent import SupervisorResearchAgent
from src.utils.models import get_model
from src.utils.helpers import get_today_str, get_prompt_template
from src.deep_research_agent.state import AgentState, AgentInputState
from pydantic import BaseModel, Field
from src.config import ROOT_DIR
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    get_buffer_string,
    SystemMessage,
)
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command
import os
from typing import Literal
from langgraph.checkpoint.memory import InMemorySaver

from IPython.display import Image, display
from langchain_core.runnables import RunnableConfig


# Structured output schema
class ClarifyWithUser(BaseModel):
    """Schema for user clarification decision and questions."""

    need_clarification: bool = Field(
        description="Whether the user needs to be asked a clarifying question.",
    )
    question: str = Field(
        description="A question to ask the user to clarify the report scope",
    )
    verification: str = Field(
        description="Verify message that we will start research after the user has provided the necessary information.",
    )


class ResearchQuestion(BaseModel):
    """Schema for structured research brief generation."""

    research_brief: str = Field(
        description="A research question that will be used to guide the research.",
    )


class DeepResearchAgent:
    def __init__(
        self,
        agent_reasoning: Literal["low", "medium", "high"] = "medium",
        interleaved_thinking: bool = True,
    ):
        self.model = get_model(reasoning="medium")
        self.writer_model = get_model(reasoning="medium")
        self.supervisor_reasoning = agent_reasoning
        self.interleaved_thinking = interleaved_thinking

        self.clarify_with_user_template = get_prompt_template(
            os.path.join(
                ROOT_DIR,
                "src/deep_research_agent/prompts/clarify_with_user_instruction.jinja",
            )
        )
        self.write_research_brief_template = get_prompt_template(
            os.path.join(
                ROOT_DIR,
                "src/deep_research_agent/prompts/write_research_brief_from_messages.jinja",
            )
        )

        self.write_final_report_template = get_prompt_template(
            os.path.join(
                ROOT_DIR,
                "src/deep_research_agent/prompts/final_report_generation_prompt.jinja",
            )
        )

    async def clarify_with_user(
        self,
        state: AgentState,
    ) -> Command[Literal["write_research_brief", "__end__"]]:
        """
        Determine if the user's request contains sufficient information to proceed with research.

        Uses structured output to make deterministic decisions and avoid hallucination.
        Routes to either research brief generation or ends with a clarification question.
        """
        structured_output_model = self.model.with_structured_output(ClarifyWithUser)
        # Invoke the model with clarification instructions
        response = structured_output_model.invoke(
            [
                SystemMessage(
                    content=self.clarify_with_user_template.render(
                        messages=get_buffer_string(messages=state["messages"]),
                        date=get_today_str(),
                    )
                )
            ]
        )

        # Alternately can also go to a dedicated node to ask human for clarification question
        # Route based on clarification need
        if response.need_clarification:
            return Command(
                goto=END, update={"messages": [AIMessage(content=response.question)]}
            )
        else:
            return Command(
                goto="write_research_brief",
                update={"messages": [AIMessage(content=response.verification)]},
            )

    async def write_research_brief(self, state: AgentState):
        """
        Transform the conversation history into a comprehensive research brief.

        Uses structured output to ensure the brief follows the required format
        and contains all necessary details for effective research.
        """
        # Set up structured output model
        structured_output_model = self.model.with_structured_output(ResearchQuestion)

        # Generate research brief from conversation history
        response = structured_output_model.invoke(
            [
                SystemMessage(
                    content=self.write_research_brief_template.render(
                        messages=get_buffer_string(state.get("messages", [])),
                        date=get_today_str(),
                    )
                )
            ]
        )

        # Update state with generated research brief and pass it to the supervisor
        return {
            "research_brief": response.research_brief,
            "supervisor_messages": [
                HumanMessage(content=f"{response.research_brief}.")
            ],
        }

    async def supervisor_subgraph(self, state: AgentState, config: RunnableConfig):
        research_supervisor = SupervisorResearchAgent(
            interleaved_thinking=self.interleaved_thinking,
            agent_reasoning=self.supervisor_reasoning,
        ).build_agent_graph()

        supervisor_state = {
            "supervisor_messages": [
                HumanMessage(content=state.get("research_brief", ""))
            ],
        }
        result = await research_supervisor.ainvoke(supervisor_state)
        return {
            "notes": result.get("notes", []),
            "raw_notes": result.get("raw_notes", []),
        }

    async def final_report_generation(self, state: AgentState):
        """
        Final report generation node.

        Synthesizes all research findings into a comprehensive final report
        """

        notes = state.get("notes", [])
        if len(notes) == 0:
            return {
                "final_report": "The report could not be generated.",
                "messages": ["The report could not be generated."],
            }

        findings = "\n".join(notes)

        final_report_prompt = self.write_final_report_template.render(
            research_brief=state.get("research_brief", ""),
            findings=findings,
            date=get_today_str(),
        )

        final_report = await self.writer_model.ainvoke(
            [HumanMessage(content=final_report_prompt)]
        )

        return {
            "final_report": final_report.content,
            "messages": ["Here is the final report: " + final_report.content],
        }

    def build_agent_graph(self):
        builder = StateGraph(AgentState, input_schema=AgentInputState)
        builder.add_node("clarify_with_user", self.clarify_with_user)
        builder.add_node("write_research_brief", self.write_research_brief)
        builder.add_node("supervisor_subgraph", self.supervisor_subgraph)
        builder.add_node("final_report_generation", self.final_report_generation)
        builder.add_edge(START, "clarify_with_user")
        builder.add_edge("write_research_brief", "supervisor_subgraph")
        builder.add_edge("supervisor_subgraph", "final_report_generation")
        builder.add_edge("final_report_generation", END)

        checkpointer = InMemorySaver()
        return builder.compile(checkpointer=checkpointer)
