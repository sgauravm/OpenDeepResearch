"""Chart generation dispatcher.

Two backends, selected by ``CHART_AGENT_CONFIG["mode"]`` in ``src/config.py``:

- ``"structured"`` (default): the section writer planner fills a
  ``ChartParams`` schema and the matplotlib renderer draws the figure
  directly. Zero LLM calls at generation time — the planning LLM call
  already happened in the section writer. Near-100% reliable but limited
  to the built-in chart types.

- ``"code"``: the legacy plotly code-generation agent. The same
  ``ChartParams`` is rendered to a prose description via
  ``chart_params_to_prose`` and fed to the code-generation LLM, which
  writes plotly code executed in a sandbox. More expressive but requires
  a capable coder model.

Both backends return a ``ChartResult`` so callers (the section writer's
charts node) don't need to know which mode is active.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.config import CHART_AGENT_CONFIG
from src.deep_research_agent.chart_renderer import (
    CHARTS_DIR,
    ChartParams,
    ChartResult,
    chart_params_to_prose,
    render_chart,
)


def generate_chart(chart_name: str, chart_params: ChartParams) -> ChartResult:
    """Generate a single chart using the configured backend."""
    mode = CHART_AGENT_CONFIG.get("mode", "structured")

    if mode == "structured":
        return render_chart(chart_name, chart_params)

    if mode == "code":
        return _generate_via_code_agent(chart_name, chart_params)

    return ChartResult(
        success=False,
        chart_name=chart_name,
        error=f"Unknown chart mode: {mode!r}",
    )


# ---------------------------------------------------------------------------
# Code-mode adapter
# ---------------------------------------------------------------------------


def _generate_via_code_agent(
    chart_name: str, chart_params: ChartParams
) -> ChartResult:
    """Run the legacy plotly code agent from a ChartParams spec.

    Converts the structured spec to prose (deterministic, no LLM) and runs
    the existing ChartAgent subgraph. On success, saves the plotly figure
    as PNG / HTML / JSON in the charts directory.
    """
    # Local import to avoid loading the plotly agent when running in the
    # default structured mode.
    from langchain_core.messages import HumanMessage

    from src.deep_research_agent.agents.chart_agent_code_with_tools import (
        ChartAgent,
    )

    description = chart_params_to_prose(chart_params)

    chart_agent = ChartAgent()
    chart_graph = chart_agent.build_agent_graph()

    initial_state = {
        "messages": [HumanMessage(content=description)],
        "chart_json": {},
        "chart_name": chart_name,
        "is_code_error": False,
        "data_provided": True,
        "num_code_correction": 0,
        "python_code": "",
    }

    try:
        result = chart_graph.invoke(initial_state, {"recursion_limit": 20})
    except Exception as e:
        return ChartResult(
            success=False,
            chart_name=chart_name,
            error=f"Chart agent raised: {str(e)[:200]}",
        )

    chart_json = result.get("chart_json")
    data_provided = result.get("data_provided", True)

    if not data_provided:
        return ChartResult(
            success=False,
            chart_name=chart_name,
            error="Data was not provided in the chart description.",
        )

    if not chart_json or not isinstance(chart_json, dict) or not chart_json:
        return ChartResult(
            success=False,
            chart_name=chart_name,
            error="Chart agent produced no valid figure.",
        )

    try:
        _save_plotly_chart(chart_name, chart_json)
    except Exception as e:
        return ChartResult(
            success=False,
            chart_name=chart_name,
            error=f"Chart was generated but could not be saved: {str(e)[:200]}",
        )

    return ChartResult(
        success=True,
        chart_name=chart_name,
        image_markdown=f"![{chart_name}](charts/{chart_name}.png)",
    )


def _save_plotly_chart(chart_name: str, chart_json: dict) -> None:
    """Save a plotly figure as JSON, HTML and PNG (code-mode only)."""
    import plotly.io as pio

    CHARTS_DIR.mkdir(parents=True, exist_ok=True)

    json_path: Path = CHARTS_DIR / f"{chart_name}.json"
    with open(json_path, "w") as f:
        json.dump(chart_json, f, indent=2)

    fig = pio.from_json(json.dumps(chart_json))

    html_path: Path = CHARTS_DIR / f"{chart_name}.html"
    fig.write_html(str(html_path))

    png_path: Path = CHARTS_DIR / f"{chart_name}.png"
    try:
        fig.write_image(str(png_path), width=1200, height=800, scale=2)
    except Exception as e:
        print(f"Warning: could not generate PNG for {chart_name}: {e}")
