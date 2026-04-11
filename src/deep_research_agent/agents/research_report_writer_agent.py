"""Research Report Writer Agent Implementation.

This module implements a research report writer with a three-stage pipeline:
1. Planner node: Creates a structured report plan from research findings
2. Section writer node: Uses the section writer agent to write each section
3. Final doc node: Concatenates all sections and appends source references
"""

from typing import Literal

from langchain_core.messages import SystemMessage
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field

from src.config import ROOT_DIR
from src.utils.helpers import get_prompt_template, get_today_str
from src.utils.models import get_model
from src.deep_research_agent.state import ResearchWriterState
from src.deep_research_agent.agents.section_writer_agent import SectionWriterAgent


SectionMode = Literal["writing_instruction", "actual_content"]


def create_report_planner_schema(valid_source_filenames: list[str]):
    """Create ReportPlannerSchema with source_file_name_list constrained to valid filenames.

    The schema is generated dynamically so the LLM's structured output only accepts
    source file names that exist in research_notes. Uses Literal for JSON schema enum.
    """
    if valid_source_filenames:
        # Literal["a", "b", "c"] at runtime: Literal[tuple(...)] unpacks correctly
        SourceFileLiteral = Literal[tuple(valid_source_filenames)]
    else:
        SourceFileLiteral = str  # fallback when no research notes

    class ReportPlan(BaseModel):
        """Schema for a single section in the report plan."""

        section_name: str = Field(
            description="The name of the section (H1 level, do not include '#')."
        )
        section_mode: SectionMode = Field(
            description="Mode of the section: either 'writing_instruction' or 'actual_content'."
        )
        writing_instruction: str = Field(
            description=(
                "Direct, imperative instructions for the section writer. "
                "Must be non-empty if section_mode is 'writing_instruction'. "
                "Must be empty string if section_mode is 'actual_content'."
            )
        )
        actual_content: str = Field(
            description=(
                "Publication-ready final content of the section. "
                "Must be non-empty if section_mode is 'actual_content'. "
                "Must be empty string if section_mode is 'writing_instruction'."
            )
        )
        source_file_name_list: list[SourceFileLiteral] = Field(
            description=(
                "List of source file names directly required for this section. "
                "Must be non-empty when section_mode is 'writing_instruction'. "
                "Must be an empty list [] when section_mode is 'actual_content'. "
                f"Valid names: {valid_source_filenames}."
            )
        )

    class ReportPlannerSchema(BaseModel):
        """Schema for the full report plan output."""

        report_plan: list[ReportPlan] = Field(
            description="Ordered list of report sections forming the complete structured report plan."
        )

    return ReportPlannerSchema


class ResearchWriterAgent:
    """Encapsulates the research report writer agent with planner, section writer, and final doc nodes."""

    def __init__(
        self,
        planner_reasoning: Literal["low", "medium", "high"] = "medium",
        writer_reasoning: Literal["low", "medium", "high"] = "medium",
    ):
        # Base model without structured output; schema is bound dynamically in planner_node
        # from research_notes keys so source_file_name_list uses Literal[valid_filenames]
        self.planner_model = get_model(reasoning=planner_reasoning)

        # Section writer agent for writing sections with chart capability
        self.section_writer_agent = SectionWriterAgent(reasoning=writer_reasoning)

        self.report_writing_planner_template = get_prompt_template(
            ROOT_DIR / "src/deep_research_agent/prompts/report_writing_planner.jinja"
        )

    # ===== Node Implementations =====

    def _build_research_findings(self, research_notes: dict) -> str:
        """Build research findings string for the planner from research_notes."""
        lines = []
        for idx, (filename, note) in enumerate(research_notes.items(), start=1):
            if isinstance(note, dict):
                desc = note.get("description", "No description")
                lines.append(f"[{idx}]. {filename}: {desc}")
        # TODO: Return empty text and at that time go to end
        return "\n".join(lines) if lines else "No research findings available."

    def planner_node(self, state: ResearchWriterState) -> dict:
        """Create a structured report plan from research brief and findings."""
        research_notes = state.get("research_notes", {})
        research_brief = state.get("research_brief", "")

        if not research_notes:
            return {
                "report_plan": [],
                "current_section_index": 0,
            }

        research_findings = self._build_research_findings(research_notes)
        prompt = self.report_writing_planner_template.render(
            research_brief=research_brief,
            research_findings=research_findings,
            date=get_today_str(),
        )

        # Schema is dynamic: source_file_name_list uses Literal[valid_filenames]
        valid_filenames = list(research_notes.keys())
        ReportPlannerSchema = create_report_planner_schema(valid_filenames)
        structured_planner = self.planner_model.with_structured_output(
            ReportPlannerSchema
        )
        result = structured_planner.invoke([SystemMessage(content=prompt)])

        # Convert Pydantic models to dicts for state
        report_plan = result.model_dump()["report_plan"]

        return {"report_plan": report_plan, "current_section_index": 0}

    def _build_source_content(
        self, source_file_names: list[str], research_notes: dict
    ) -> str:
        """Build source content string with global indices for citation (format: [index_N]).
        Indices match the final sources section order (research_notes key order).
        """
        filename_to_global_idx = {
            fn: idx for idx, fn in enumerate(research_notes.keys(), start=1)
        }
        parts = []
        for filename in source_file_names:
            if filename in research_notes:
                note = research_notes[filename]
                if isinstance(note, dict):
                    content = note.get("content", "")
                    global_idx = filename_to_global_idx.get(filename, 0)
                    parts.append(f"SOURCE [{global_idx}]:\n{content}\n---\n\n")
        return "".join(parts)

    def section_writer_node(self, state: ResearchWriterState) -> dict:
        """Write a single section based on the report plan using the section writer agent."""
        report_plan = state.get("report_plan", [])
        current_index = state.get("current_section_index", 0)
        section_texts = state.get("section_texts", [])
        research_notes = state.get("research_notes", {})
        research_brief = state.get("research_brief", "")

        if current_index >= len(report_plan):
            return {"current_section_index": current_index}

        section = report_plan[current_index]
        section_mode = section.get("section_mode", "writing_instruction")

        if section_mode == "actual_content":
            # Use pre-written content directly
            content = section.get("actual_content", "")
            # Ensure it starts with the section heading
            section_name = section.get("section_name", "")
            if content and not content.strip().startswith("#"):
                content = f"# {section_name}\n\n{content}"
            elif not content:
                content = f"# {section_name}\n\n"
            return {
                "section_texts": [content],
                "current_section_index": current_index + 1,
            }

        # writing_instruction mode: use section writer agent
        section_names = "\n".join(
            [f"{i + 1}. {s['section_name']}" for i, s in enumerate(report_plan)]
        )
        cur_section_name = section.get("section_name", "")
        cur_section_instruction = section.get("writing_instruction", "")
        source_file_names = section.get("source_file_name_list", [])

        prev_section_text = (
            section_texts[-1]
            if section_texts
            else "No previous section text available."
        )

        source_text = self._build_source_content(source_file_names, research_notes)

        # Use the section writer agent to write the section
        content = self.section_writer_agent.write_section(
            research_brief=research_brief,
            section_names=section_names,
            cur_section=cur_section_name,
            section_description=cur_section_instruction,
            previous_section=prev_section_text,
            source_content=source_text,
        )

        return {
            "section_texts": [content],
            "current_section_index": current_index + 1,
        }

    def final_doc_node(self, state: ResearchWriterState) -> dict:
        """Concatenate all sections and append source references."""
        section_texts = state.get("section_texts", [])
        research_notes = state.get("research_notes", {})

        # Build main report from section_texts (they should match report_plan order)
        main_sections = "\n\n".join(section_texts) if section_texts else ""

        # Build sources section
        sources_lines = ["# Sources\n"]
        for idx, (filename, note) in enumerate(research_notes.items(), start=1):
            if isinstance(note, dict):
                url = note.get("url", "")
                title = note.get("title") or note.get("description", filename)
                sources_lines.append(f"- [{idx}] {title}: {url}")

        sources_section = "\n".join(sources_lines)
        final_research_text = (
            f"{main_sections}\n\n{sources_section}"
            if main_sections
            else f"No report content generated.\n\n{sources_section}"
        )

        return {"final_research_text": final_research_text}

    # ===== Routing Logic =====

    def should_continue_section_writing(
        self, state: ResearchWriterState
    ) -> Literal["section_writer", "final_doc"]:
        """Decide whether to write another section or proceed to final doc."""
        report_plan = state.get("report_plan", [])
        current_index = state.get("current_section_index", 0)
        if current_index >= len(report_plan):
            return "final_doc"
        return "section_writer"

    def should_run_section_writer(
        self, state: ResearchWriterState
    ) -> Literal["section_writer", "final_doc"]:
        """After planner: run section writer if we have a plan, else go to final_doc."""
        report_plan = state.get("report_plan", [])
        if not report_plan:
            return "final_doc"
        return "section_writer"

    # ===== Agent Graph Builder =====

    def build_agent_graph(self):
        """Constructs and returns the research writer workflow graph."""
        agent_builder = StateGraph(ResearchWriterState)

        agent_builder.add_node("planner", self.planner_node)
        agent_builder.add_node("section_writer", self.section_writer_node)
        agent_builder.add_node("final_doc", self.final_doc_node)

        agent_builder.add_edge(START, "planner")
        agent_builder.add_conditional_edges(
            "planner",
            self.should_run_section_writer,
            {
                "section_writer": "section_writer",
                "final_doc": "final_doc",
            },
        )
        agent_builder.add_conditional_edges(
            "section_writer",
            self.should_continue_section_writing,
            {
                "section_writer": "section_writer",
                "final_doc": "final_doc",
            },
        )
        agent_builder.add_edge("final_doc", END)

        return agent_builder.compile()
