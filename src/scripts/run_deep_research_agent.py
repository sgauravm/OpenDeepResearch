"""Command-line script to run the full Deep Research Agent.

Usage:
    python -m src.scripts.run_deep_research_agent
"""

import argparse
from datetime import datetime

import dotenv
from langchain_core.messages import HumanMessage

from src.config import ROOT_DIR, FINAL_AGENT_CONFIG
from src.deep_research_agent.agents.final_deep_research_agent import DeepResearchAgent
from src.utils.stream import run_async_generator, StreamEventProcessor
from src.types import ContentType

dotenv.load_dotenv()

OUTPUT_DIR = ROOT_DIR / "output"


def print_header(text: str):
    """Print a formatted header."""
    print("\n" + "=" * 60)
    print(text)
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Run the Deep Research Agent interactively from command line",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default settings
  python -m src.scripts.run_deep_research_agent

  # Run with high reasoning
  python -m src.scripts.run_deep_research_agent --reasoning high

  # Run with custom config
  python -m src.scripts.run_deep_research_agent --max-web-search-calls 10 --max-researcher-iterations 3

  # Run without showing reasoning output
  python -m src.scripts.run_deep_research_agent --no-reasoning
        """,
    )

    parser.add_argument(
        "-r", "--reasoning",
        choices=["low", "medium", "high"],
        default="medium",
        help="Reasoning level (default: medium)",
    )

    parser.add_argument(
        "--no-reasoning",
        action="store_true",
        help="Hide reasoning output",
    )

    parser.add_argument(
        "--interleaved-thinking",
        type=lambda x: x.lower() == "true",
        default=FINAL_AGENT_CONFIG["interleaved_thinking"],
        help=f"Enable interleaved thinking (default: {FINAL_AGENT_CONFIG['interleaved_thinking']})",
    )

    parser.add_argument(
        "--max-web-search-calls",
        type=int,
        default=FINAL_AGENT_CONFIG["max_web_search_calls"],
        help=f"Maximum web search calls per researcher (default: {FINAL_AGENT_CONFIG['max_web_search_calls']})",
    )

    parser.add_argument(
        "--max-web-search-results",
        type=int,
        default=FINAL_AGENT_CONFIG["max_web_search_results"],
        help=f"Maximum results per web search (default: {FINAL_AGENT_CONFIG['max_web_search_results']})",
    )

    parser.add_argument(
        "--max-researcher-iterations",
        type=int,
        default=FINAL_AGENT_CONFIG["max_researcher_iterations"],
        help=f"Maximum researcher iterations (default: {FINAL_AGENT_CONFIG['max_researcher_iterations']})",
    )

    args = parser.parse_args()
    show_reasoning = not args.no_reasoning

    # Initialize agent
    print_header("Deep Research Agent - CLI")
    print(f"Reasoning level: {args.reasoning}")
    print(f"Interleaved thinking: {args.interleaved_thinking}")
    print(f"Max web search calls: {args.max_web_search_calls}")
    print(f"Max researcher iterations: {args.max_researcher_iterations}")

    agent = DeepResearchAgent(
        agent_reasoning=args.reasoning,
        interleaved_thinking=args.interleaved_thinking,
    ).build_agent_graph()

    thread_id = f"cli_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    config = {
        "configurable": {
            "thread_id": thread_id,
            "max_web_search_calls": args.max_web_search_calls,
            "max_web_search_results": args.max_web_search_results,
            "max_llm_call_retry": FINAL_AGENT_CONFIG["max_llm_call_retry"],
            "max_researcher_iterations": args.max_researcher_iterations,
            "max_concurrent_researchers": FINAL_AGENT_CONFIG["max_concurrent_researchers"],
            "interleaved_thinking": args.interleaved_thinking,
            "agent_reasoning": args.reasoning,
        },
    }

    processor = StreamEventProcessor()
    final_report = None

    # Get initial query
    print("\nEnter your research query:")
    try:
        user_input = input("You: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nExiting.")
        return

    if not user_input:
        print("No query provided. Exiting.")
        return

    # Run until we get final report
    while final_report is None:
        state = {"messages": [HumanMessage(content=user_input)]}

        print(f"\n>>> Processing...")

        stream_generator = processor.process_stream(
            agent_graph=agent, state=state, config=config
        )

        clarification_question = None

        for payload in run_async_generator(stream_generator):
            c_type = payload.get("content_type")

            # Check for final report in response state
            if c_type == "response":
                response_state = payload.get("content", {})
                if response_state.get("final_report"):
                    final_report = response_state["final_report"]

            # Assistant message (could be clarification or final report)
            elif c_type == ContentType.ASSISTANT_MESSAGE:
                content = payload["content"]
                # If we don't have final_report yet, this is a clarification question
                if final_report is None:
                    clarification_question = content

            # Tool calls status
            elif c_type == ContentType.TOOL_CALLED:
                content = payload.get("content", "")
                if content:
                    print(f"  ⏳ {content}")

            # Reasoning streaming
            elif c_type == ContentType.START_STREAM_REASON and show_reasoning:
                print("\n🧠 Thinking...")
                print(f"  {payload.get('content', '')}", end="", flush=True)

            elif c_type == ContentType.STREAMING_REASON and show_reasoning:
                print(payload.get("content", ""), end="", flush=True)

            elif c_type == "stop_stream_reason" and show_reasoning:
                print("\n")

        # If we got a clarification question and no final report, ask user
        if clarification_question and final_report is None:
            print("\n" + "-" * 60)
            print("Agent asks:")
            print("-" * 60)
            print(clarification_question)
            print("-" * 60)

            try:
                user_input = input("\nYour response: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nExiting.")
                return

            if not user_input:
                user_input = "Please proceed with the research based on my original query."

    # Research complete - show results
    print_header("RESEARCH COMPLETE")

    # Show saved files
    print("\nOutput files saved:")
    for f in sorted(OUTPUT_DIR.glob("research_report_*.md"), reverse=True)[:1]:
        print(f"  Markdown: {f}")
    for f in sorted(OUTPUT_DIR.glob("research_report_*.pdf"), reverse=True)[:1]:
        print(f"  PDF: {f}")

    # Check for charts
    charts_dir = OUTPUT_DIR / "charts"
    if charts_dir.exists():
        chart_files = list(charts_dir.glob("*.png"))
        if chart_files:
            print(f"\nCharts generated: {len(chart_files)}")
            for chart in chart_files[-5:]:
                print(f"  - {chart.name}")


if __name__ == "__main__":
    main()
