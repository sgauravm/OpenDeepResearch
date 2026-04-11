from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
PROMPTS_DIR = ROOT_DIR / "src" / "deep_research_agent" / "prompts"
MODEL_CONFIG = {
    # "model_name": "gpt-oss",
    "model_name": "gemma4:e2b",
    "temperature": 0,
    "reasoning": "medium",
}

RESEARCHER_AGENT_CONFIG = {
    "max_web_search_calls": 3,
    "max_web_search_results": 1,
    "max_llm_call_retry": 2,
}

SUPERVISOR_AGENT_CONFIG = {
    "max_researcher_iterations": 3,
    "max_concurrent_researchers": 3,
    "max_llm_call_retry": 3,
}

FINAL_AGENT_CONFIG = {
    "max_web_search_calls": 5,
    "max_web_search_results": 3,
    "max_llm_call_retry": 2,
    "max_researcher_iterations": 2,
    "max_concurrent_researchers": 3,
    "interleaved_thinking": True,
    "agent_reasoning": "medium",
}

CHART_AGENT_CONFIG = {
    # "structured" renders charts deterministically from a ChartParams schema
    # using matplotlib. Near-100% reliable, fixed palette, fixed chart types.
    # Best for small general-purpose models.
    #
    # "code" uses the legacy plotly code-generation agent (runs LLM-written
    # code in a sandbox). More expressive but requires a capable code model.
    "mode": "structured",
    # 6-color palette used by the structured renderer in the order of the
    # data series. Keep the first entry as the default single-series color.
    "palette": [
        "#1e88e5",  # primary blue
        "#e74c3c",  # red
        "#1abc9c",  # teal
        "#f39c12",  # orange
        "#9b59b6",  # purple
        "#2c3e50",  # dark slate
    ],
    # Color used for a highlighted category (single bar/slice emphasis).
    "highlight_color": "#f39c12",
    # Figure sizing for the structured matplotlib renderer.
    "figsize": (10, 6),
    "figsize_pie": (8, 8),
    "dpi": 150,
    # Number of lines of traceback shown to the code-mode LLM when a code
    # execution fails. 2 is too aggressive for debugging; 8 gives enough
    # context for the model to locate and fix its own bugs.
    "code_mode_error_context_lines": 8,
}
