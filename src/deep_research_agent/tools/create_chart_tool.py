"""
Chart creation tool for section writer agent.

This tool wraps the chart agent to create charts based on descriptions
provided by the section writer.
"""

import json
from pathlib import Path

from langchain.tools import tool

from src.config import ROOT_DIR
from src.deep_research_agent.agents.chart_agent_code_with_tools import ChartAgent


# Directory where charts will be saved
CHARTS_DIR = ROOT_DIR / "output" / "charts"


def save_chart(chart_name: str, chart_json: dict) -> tuple[Path, bool]:
    """Save chart JSON, HTML, and PNG using plotly.

    Returns:
        Tuple of (png_path, png_success)
    """
    import plotly.io as pio

    # Ensure charts directory exists
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)

    # Save JSON
    json_path = CHARTS_DIR / f"{chart_name}.json"
    with open(json_path, "w") as f:
        json.dump(chart_json, f, indent=2)

    # Create figure from JSON
    fig = pio.from_json(json.dumps(chart_json))

    # Save HTML (always works)
    html_path = CHARTS_DIR / f"{chart_name}.html"
    fig.write_html(str(html_path))

    # Generate PNG (requires kaleido)
    png_path = CHARTS_DIR / f"{chart_name}.png"
    png_success = False
    try:
        fig.write_image(str(png_path), width=1200, height=800, scale=2)
        png_success = True
    except Exception as e:
        print(f"Warning: Could not generate PNG for {chart_name}: {e}")
        print("Hint: Install kaleido with 'pip install kaleido' for PNG export")

    return png_path, png_success


@tool
def create_chart(chart_name: str, chart_description: str) -> str:
    """
    Create a chart based on the provided description.

    Use this tool when you need to create a visual chart for the section.
    The chart will be generated using Plotly and saved for inclusion in the report.

    Args:
        chart_name: Name of the chart in snake_case format (e.g., revenue_growth_chart,
            market_share_comparison). This will be used as the chart identifier and file name.
        chart_description: A detailed, library-agnostic description of the chart to create.
            Must include:
            - The type of chart (bar, line, pie, scatter, etc.)
            - The exact data to visualize (with actual values, not placeholders)
            - Axis labels and what they represent
            - Title for the chart
            - Any aesthetic preferences (colors, styling)
            - What insights the chart should convey

            Example: "Create a donut chart (pie chart with a hole) titled 'Market Share of Cloud Providers — 2025'. The segments and their percentage values are: 'AWS' 31%, 'Microsoft Azure' 25%, 'Google Cloud' 11%, 'Alibaba Cloud' 5%, 'IBM Cloud' 4%, 'Others' 24%. Use these colors respectively: #FF9900, #0078D4, #4285F4, #FF6A00, #054ADA, #AAAAAA. The hole size should be 0.45. Display percentage labels inside each slice in white bold font. Add a center annotation that says 'Cloud Market' in dark gray, font size 16. Pull out the 'AWS' slice slightly (pull=0.05) to emphasize it. Place the legend to the right of the chart vertically. Use a white background with no border or grid. Add a subtle drop shadow effect to the chart if possible. The chart should look sleek and suitable for a business presentation."

    Returns:
        A message indicating success or failure of chart generation.
    """
    chart_agent = ChartAgent()
    chart_graph = chart_agent.build_agent_graph()

    initial_state = {
        "messages": [],
        "initial_instruction": chart_description,
        "detailed_instruction": "",
        "use_detailed_instruction": True,
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
        return f"Chart '{chart_name}' cannot be created due to an error: {str(e)[:100]}. Continue writing the section without this chart."

    # Check if chart was successfully generated
    chart_json = result.get("chart_json")
    data_provided = result.get("data_provided", True)

    if chart_json and data_provided:
        try:
            # Save the chart
            _, png_success = save_chart(chart_name, chart_json)
            if png_success:
                return f"Successfully generated the chart '{chart_name}'. Proceed with writing the section and place the chart at the appropriate location using the markdown format: ![{chart_name}](charts/{chart_name}.png)"
            else:
                return f"Successfully generated the chart '{chart_name}' (HTML only, PNG requires kaleido). Proceed with writing the section and place the chart at the appropriate location using the markdown format: ![{chart_name}](charts/{chart_name}.png)"
        except Exception as e:
            return f"Chart '{chart_name}' was generated but could not be saved: {str(e)}. Continue writing the section without this chart."
    elif not data_provided:
        return f"Chart '{chart_name}' cannot be created: Data was not provided in the description. Continue writing the section without this chart."
    else:
        return f"Chart '{chart_name}' cannot be created due to an error. Continue writing the section without this chart."
