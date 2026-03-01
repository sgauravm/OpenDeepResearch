import os
from datetime import datetime
from pathlib import Path
from typing import Literal

from IPython.display import Image, display
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    get_buffer_string,
)
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command
from pydantic import BaseModel, Field

from src.config import ROOT_DIR
from src.deep_research_agent.agents.research_report_writer_agent import (
    ResearchWriterAgent,
)
from src.deep_research_agent.agents.supervisor_agent import SupervisorResearchAgent
from src.deep_research_agent.state import AgentInputState, AgentState
from src.utils.helpers import get_prompt_template, get_today_str
from src.utils.models import get_model
import markdown
from weasyprint import HTML, CSS

# Output directory for saved reports
OUTPUT_DIR = ROOT_DIR / "output"


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

        self.research_writer_agent = ResearchWriterAgent(
            planner_reasoning="medium",
            writer_reasoning="low",
        ).build_agent_graph()

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
            "research_notes": state.get("research_notes", {}),
            "visited_urls": state.get("visited_urls", []),
        }
        result = await research_supervisor.ainvoke(supervisor_state)
        return {
            "research_notes": result.get("research_notes", {}),
            "visited_urls": result.get("visited_urls", []),
        }

    async def final_report_generation(self, state: AgentState):
        """
        Final report generation node.

        Uses ResearchWriterAgent to synthesize all research findings into a
        comprehensive final report via planner, section writer, and final doc pipeline.
        Saves the generated report as a markdown file.
        """
        research_notes = state.get("research_notes", {})
        research_brief = state.get("research_brief", "")

        if len(research_notes) == 0:
            return {
                "final_report": "The report could not be generated.",
                "messages": ["The report could not be generated."],
            }

        writer_state = {
            "research_brief": research_brief,
            "research_notes": research_notes,
        }
        result = await self.research_writer_agent.ainvoke(writer_state)
        final_research_text = result.get(
            "final_research_text", "The report could not be generated."
        )

        # Save the report as a markdown file
        self._save_report(final_research_text)

        return {
            "final_report": final_research_text,
            "messages": [final_research_text],
        }

    def _save_report(self, report_content: str) -> Path:
        """Save the generated report as markdown and PDF files."""
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"research_report_{timestamp}"

        # Save markdown
        md_filepath = OUTPUT_DIR / f"{base_filename}.md"
        md_filepath.write_text(report_content, encoding="utf-8")
        print(f"Markdown report saved to: {md_filepath}")

        # Save PDF
        pdf_filepath = OUTPUT_DIR / f"{base_filename}.pdf"
        self._save_pdf(report_content, pdf_filepath)

        return md_filepath

    def _save_pdf(self, markdown_content: str, pdf_path: Path):
        """Convert markdown to PDF and save."""
        try:

            # Convert markdown to HTML
            html_content = markdown.markdown(
                markdown_content,
                extensions=["tables", "fenced_code", "toc"],
            )

            # Create full HTML document with styling
            charts_dir = OUTPUT_DIR / "charts"
            full_html = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="utf-8">
                <style>
                    body {{
                        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
                        line-height: 1.6;
                        max-width: 800px;
                        margin: 0 auto;
                        padding: 20px;
                        color: #333;
                    }}
                    h1 {{ color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
                    h2 {{ color: #34495e; margin-top: 30px; }}
                    h3 {{ color: #7f8c8d; }}
                    table {{
                        border-collapse: collapse;
                        width: 100%;
                        margin: 20px 0;
                    }}
                    th, td {{
                        border: 1px solid #ddd;
                        padding: 12px;
                        text-align: left;
                    }}
                    th {{ background-color: #3498db; color: white; }}
                    tr:nth-child(even) {{ background-color: #f9f9f9; }}
                    code {{
                        background-color: #f4f4f4;
                        padding: 2px 6px;
                        border-radius: 3px;
                        font-family: 'Courier New', monospace;
                    }}
                    pre {{
                        background-color: #f4f4f4;
                        padding: 15px;
                        border-radius: 5px;
                        overflow-x: auto;
                    }}
                    img {{
                        max-width: 100%;
                        height: auto;
                        margin: 20px 0;
                    }}
                    blockquote {{
                        border-left: 4px solid #3498db;
                        margin: 20px 0;
                        padding-left: 20px;
                        color: #666;
                    }}
                </style>
            </head>
            <body>
                {html_content}
            </body>
            </html>
            """

            # Replace relative chart paths with absolute paths for PDF rendering
            full_html = full_html.replace('src="charts/', f'src="file://{charts_dir}/')
            full_html = full_html.replace("src='charts/", f"src='file://{charts_dir}/")

            # Generate PDF
            HTML(string=full_html, base_url=str(OUTPUT_DIR)).write_pdf(str(pdf_path))
            print(f"PDF report saved to: {pdf_path}")

        except ImportError as e:
            print(f"PDF generation skipped: {e}")
            print("Install required packages: pip install markdown weasyprint")
        except Exception as e:
            print(f"PDF generation failed: {e}")

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
