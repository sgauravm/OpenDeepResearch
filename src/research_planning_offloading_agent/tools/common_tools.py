from langchain_core.tools import tool


@tool(parse_docstring=True)
def think_tool(thought: str) -> str:
    """Tool for strategic reflection on research progress and decisions.

    Use this tool after each search to pause, analyze results, and plan next steps.

    When to use:
        •	After search results: What key information did I find?
        •	Before next steps: Do I have enough for a full answer?
        •	When spotting gaps: What am I still missing?
        •	Before concluding: Is the answer complete?
        •	For complex questions: Have I reached search limits?

    Reflection should cover:
        1.	Findings: What concrete information do I have?
        2.	Gaps: What crucial details are missing?
        3.	Quality: Do I have enough evidence/examples?
        4.	Decision: Continue searching or answer now?

    Args:
        thought: Your detailed reflection on research progress, findings, gaps, and next steps

    Returns:
        Confirmation that reflection was recorded for decision-making
    """
    return f"Reflection recorded: {thought}"
