"""
Test the ChartAgent (tools-based) on the evaluation dataset (data/graph_description.jsonl).

Runs the agent on each description, collects the Plotly JSON output,
and saves all results to a single JSON file with ids as keys.
"""

from __future__ import annotations

import json
from pathlib import Path

from langchain_core.messages import HumanMessage
from tqdm import tqdm

from src.deep_research_agent.agents.chart_agent_code_with_tools import ChartAgent
from src.config import ROOT_DIR

# Paths
DATA_PATH = ROOT_DIR / "data" / "graph_description.jsonl"
OUTPUT_DIR = ROOT_DIR / "output" / "eval_chart_agent_with_tools"
OUTPUT_FILE = OUTPUT_DIR / "results_tools.json"


def load_dataset(path: Path) -> list[dict]:
    """Load JSONL dataset; each line has 'id' and 'description'."""
    entries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entries.append(json.loads(line))
    return entries


def run_chart_agent_with_tools(description: str) -> dict:
    """Run the chart agent with tools."""
    chart_agent = ChartAgent()
    graph = chart_agent.build_agent_graph()
    result = graph.invoke(
        {
            "messages": [
                HumanMessage(
                    content=f"Generate a plotly chart for the following description:\n {description}"
                )
            ]
        }
    )
    return result


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    entries = load_dataset(DATA_PATH)
    results: dict[int, dict] = {}

    for entry in tqdm(entries, desc="Chart agent (tools) eval"):
        idx = entry["id"]
        description = entry["description"]

        result = run_chart_agent_with_tools(description)
        result.pop("messages")

        results[idx] = result

    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=2)

    # Print num errors and numdata not provided and total example
    num_errors = sum(
        1 for result in results.values() if result.get("is_code_error", False)
    )
    num_data_not_provided = sum(
        1 for result in results.values() if not result.get("data_provided", True)
    )
    total_examples = len(results)
    print(f"Num errors: {num_errors}")
    print(f"Num data not provided: {num_data_not_provided}")
    print(f"Total examples: {total_examples}")

    print(f"\nDone. Results saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
