"""Test script for ResearchWriterAgent.

Loads research brief and research notes from data/research_notes_data.json,
runs the report writer agent, and prints the final research report.
"""

import asyncio
import json
from pathlib import Path

import dotenv

from src.config import ROOT_DIR
from src.deep_research_agent.agents.research_report_writer_agent import (
    ResearchWriterAgent,
)

dotenv.load_dotenv()


def load_research_data():
    """Load research brief and research notes from the data file."""
    data_path = ROOT_DIR / "data" / "research_notes_data.json"
    with open(data_path) as f:
        data = json.load(f)

    # Use first item from the list
    item = data[0] if isinstance(data, list) else data
    research_brief = item.get("research_brief", "")
    research_notes = item.get("research_notes", {})

    return research_brief, research_notes


async def run_with_progress(agent, state, config):
    """Run the agent with progress updates printed as each stage completes."""
    report_plan = []
    result = None

    print("Planning started...")
    async for stream_item in agent.astream(
        state,
        config=config,
        stream_mode=["updates", "values"],
    ):
        # Handle (mode, chunk) tuple when multiple stream modes
        if isinstance(stream_item, tuple):
            mode, chunk = stream_item
        else:
            mode, chunk = "values", stream_item

        if mode == "updates" and chunk:
            for node_name, updates in chunk.items():
                if node_name == "planner":
                    report_plan = updates.get("report_plan", [])
                    print("Planning done.")
                    if report_plan:
                        section_name = report_plan[0].get("section_name", "Section 1")
                        print(f"Section '{section_name}' starting...")

                elif node_name == "section_writer":
                    current_idx = updates.get("current_section_index", 0)
                    completed_idx = current_idx - 1
                    if 0 <= completed_idx < len(report_plan):
                        section_name = report_plan[completed_idx].get(
                            "section_name", f"Section {completed_idx + 1}"
                        )
                        print(f"Section '{section_name}' completed.")
                    if current_idx < len(report_plan):
                        section_name = report_plan[current_idx].get(
                            "section_name", f"Section {current_idx + 1}"
                        )
                        print(f"Section '{section_name}' starting...")

                elif node_name == "final_doc":
                    print("Final report done.")

        if mode == "values" and chunk:
            result = chunk

    return result


def main():
    research_brief, research_notes = load_research_data()

    agent = ResearchWriterAgent(
        planner_reasoning="medium",
        writer_reasoning="medium",
    ).build_agent_graph()

    state = {
        "research_brief": research_brief,
        "research_notes": research_notes,
    }

    config = {"configurable": {"thread_id": "research_writer_test"}}

    result = asyncio.run(run_with_progress(agent, state, config))

    if result and "final_research_text" in result:
        output_dir = ROOT_DIR / "output"
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / "research_report.md"
        output_path.write_text(result["final_research_text"], encoding="utf-8")
        print(f"\n=== FINAL RESEARCH REPORT ===\n")
        print(result["final_research_text"])
        print(f"\n\nReport saved to {output_path}")
    else:
        print("No report generated.")
        if result:
            print("Result keys:", list(result.keys()))


if __name__ == "__main__":
    main()
