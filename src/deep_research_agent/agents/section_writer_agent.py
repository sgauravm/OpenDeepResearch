"""Section Writer Agent.

Writes a single section of a research report. Chart creation is handled as
an explicit preparatory step — not as a tool call made by the writer model —
which keeps the pipeline deterministic across different model families.

Flow
----
    plan  →  (charts if any)  →  writer  →  END

- `plan`    Uses structured output (`SectionChartPlan`) to decide whether any
            charts are needed. Empty list is the default. When charts are
            planned, each chart gets a snake_case name, a data-complete
            description for the chart generator, and a one-sentence
            `chart_purpose` used later for narrative placement.
- `charts`  (conditional) Iterates the plan, runs the chart generator for
            each spec, and records only the successful charts. Failed or
            data-less charts are dropped silently — the writer never hears
            about them.
- `writer`  Single LLM call with NO tools bound. Given the source content,
            the section instructions, and (if any) the successful charts,
            it produces the final markdown section. Its response IS the
            section content; there is no extraction heuristic.
"""

from __future__ import annotations

from typing import Any, Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from src.config import ROOT_DIR
from src.deep_research_agent.chart_generator import generate_chart
from src.deep_research_agent.chart_renderer import ChartParams
from src.deep_research_agent.state import (
    SectionWriterOutputState,
    SectionWriterState,
)
from src.utils.helpers import get_prompt_template, get_today_str
from src.utils.models import get_model


# ---------------------------------------------------------------------------
# Structured output schema for the planner node
# ---------------------------------------------------------------------------


class ChartSpec(BaseModel):
    """Specification for a single chart to be generated for the section.

    Note: `chart_params` is a complete, typed chart specification (see
    ChartParams). Filling it directly here is more reliable than asking the
    model for a free-form description, and it lets the downstream
    dispatcher run either the structured renderer or the legacy code agent
    without needing a separate extractor step.
    """

    chart_name: str = Field(
        description=(
            "A short snake_case identifier for the chart (e.g. "
            "'finance_efficiency_gains'). No spaces, no capital letters, "
            "no special characters."
        )
    )
    chart_params: ChartParams = Field(
        description=(
            "Complete chart specification: chart type, title, axis labels, "
            "categories, series (with exact numerical values from the "
            "sources), value format, and any optional tweaks. Fill every "
            "field relevant to the chart; leave optional fields empty when "
            "not needed."
        )
    )
    chart_purpose: str = Field(
        description=(
            "One plain-prose sentence describing what the chart "
            "communicates to the reader. No numbers, no styling — just "
            "intent. Used by the writer to place the chart naturally in "
            "the narrative."
        )
    )


class SectionChartPlan(BaseModel):
    """Plan output from the chart planner node."""

    charts: list[ChartSpec] = Field(
        default_factory=list,
        description=(
            "Charts to create for this section. Return an empty list when "
            "no chart is needed — this is the common case. Only populate "
            "when the source content contains concrete numerical data that "
            "a visual would communicate materially better than prose."
        ),
    )


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class SectionWriterAgent:
    """Agent for writing report sections with optional chart creation."""

    def __init__(
        self,
        reasoning: Literal["low", "medium", "high"] = "medium",
    ):
        self.model = get_model(reasoning=reasoning)

        self.planner_template = get_prompt_template(
            ROOT_DIR / "src/deep_research_agent/prompts/section_chart_planner.jinja"
        )
        self.system_template = get_prompt_template(
            ROOT_DIR / "src/deep_research_agent/prompts/section_writer_system.jinja"
        )
        self.human_template = get_prompt_template(
            ROOT_DIR / "src/deep_research_agent/prompts/section_writer.jinja"
        )

        self._graph = self._build_graph()

    # ------------------------------------------------------------------
    # Graph nodes
    # ------------------------------------------------------------------

    def _plan_node(self, state: SectionWriterState) -> dict:
        """Decide whether any charts should be generated for this section."""
        prompt = self.planner_template.render(
            cur_section=state.get("cur_section", ""),
            section_description=state.get("section_description", ""),
            source_content=state.get("source_content", ""),
            date=get_today_str(),
        )

        structured_model = self.model.with_structured_output(SectionChartPlan)

        try:
            plan = structured_model.invoke([SystemMessage(content=prompt)])
            charts = [spec.model_dump() for spec in plan.charts]
        except Exception as e:
            # Planner failures are non-fatal — proceed with no charts.
            print(f"Chart planning failed, proceeding without charts: {e}")
            charts = []

        return {"chart_plan": charts}

    def _route_after_plan(
        self, state: SectionWriterState
    ) -> Literal["charts", "writer"]:
        """Conditional edge: skip the charts node when nothing is planned."""
        return "charts" if state.get("chart_plan") else "writer"

    def _charts_node(self, state: SectionWriterState) -> dict:
        """Generate each planned chart; keep only the successful ones.

        The planner stored each chart as a dict (via `model_dump`), so we
        reconstruct the `ChartParams` object here before handing it to the
        chart generator. Any chart that fails validation or generation is
        silently dropped — the writer never hears about failures.
        """
        chart_plan: list[dict] = state.get("chart_plan", []) or []
        successful: list[dict] = []

        for spec in chart_plan:
            chart_name = spec.get("chart_name", "")
            chart_purpose = spec.get("chart_purpose", "")
            chart_params_dict = spec.get("chart_params") or {}

            if not chart_name or not chart_params_dict:
                continue

            try:
                chart_params = ChartParams(**chart_params_dict)
            except Exception as e:
                print(
                    f"Chart '{chart_name}' skipped: invalid params: {e}"
                )
                continue

            try:
                result = generate_chart(
                    chart_name=chart_name,
                    chart_params=chart_params,
                )
            except Exception as e:
                print(f"Chart '{chart_name}' generation raised: {e}")
                continue

            if result.success and result.image_markdown:
                successful.append(
                    {
                        "chart_name": chart_name,
                        "chart_purpose": chart_purpose,
                        "image_markdown": result.image_markdown,
                    }
                )
            else:
                print(
                    f"Chart '{chart_name}' skipped: "
                    f"{result.error or 'unknown error'}"
                )

        return {"successful_charts": successful}

    def _writer_node(self, state: SectionWriterState) -> dict:
        """Write the final section. Single LLM call, no tools bound."""
        system_prompt = self.system_template.render()
        human_message = self.human_template.render(
            research_brief=state.get("research_brief", ""),
            section_names=state.get("section_names", ""),
            cur_section=state.get("cur_section", ""),
            section_description=state.get("section_description", ""),
            previous_section=state.get("previous_section", ""),
            source_content=state.get("source_content", ""),
            successful_charts=state.get("successful_charts", []),
            date=get_today_str(),
        )

        response = self.model.invoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=human_message),
            ]
        )

        content = self._msg_text(response).strip()
        return {"section_content": content, "is_complete": True}

    # ------------------------------------------------------------------
    # Graph builder
    # ------------------------------------------------------------------

    def _build_graph(self):
        builder = StateGraph(
            SectionWriterState,
            output_schema=SectionWriterOutputState,
        )
        builder.add_node("plan", self._plan_node)
        builder.add_node("charts", self._charts_node)
        builder.add_node("writer", self._writer_node)

        builder.add_edge(START, "plan")
        builder.add_conditional_edges(
            "plan",
            self._route_after_plan,
            {"charts": "charts", "writer": "writer"},
        )
        builder.add_edge("charts", "writer")
        builder.add_edge("writer", END)

        return builder.compile()

    def build_agent_graph(self):
        """Return the compiled graph (kept for API compatibility)."""
        return self._graph

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _msg_text(msg: Any) -> str:
        """Flatten a message's content into a plain string."""
        raw = getattr(msg, "content", None)
        if isinstance(raw, str):
            return raw
        if isinstance(raw, list):
            parts: list[str] = []
            for p in raw:
                if isinstance(p, str):
                    parts.append(p)
                elif isinstance(p, dict) and "text" in p:
                    parts.append(str(p.get("text", "")))
            return "".join(parts)
        return "" if raw is None else str(raw)

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def write_section(
        self,
        research_brief: str,
        section_names: str,
        cur_section: str,
        section_description: str,
        previous_section: str,
        source_content: str,
    ) -> str:
        """Write a section of the research report.

        Args:
            research_brief: The original research instruction.
            section_names: List of all section names in order.
            cur_section: Name of the current section to write.
            section_description: Writing instructions for this section.
            previous_section: Text of the previous section for style consistency.
            source_content: Source material for this section.

        Returns:
            The written section content in markdown format, guaranteed to
            start with `# {cur_section}` when any content was produced.
        """
        # Only the caller-supplied context fields are set here. Internal
        # channels (chart_plan, successful_charts, section_content,
        # is_complete) are written by the nodes themselves. This matches
        # how the other subgraphs in the project are invoked (see
        # SupervisorResearchAgent.supervisor_subgraph, ResearchWriterAgent,
        # etc.).
        initial_state = {
            "research_brief": research_brief,
            "section_names": section_names,
            "cur_section": cur_section,
            "section_description": section_description,
            "previous_section": previous_section,
            "source_content": source_content,
        }

        result = self._graph.invoke(initial_state)
        content = (result.get("section_content") or "").strip()

        # Guarantee the section starts with its heading.
        if content and not content.lstrip().startswith("# "):
            content = f"# {cur_section}\n\n{content}"

        return content
