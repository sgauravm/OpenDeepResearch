from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
MODEL_CONFIG = {
    "model_name": "gpt-oss",
    "temperature": 0,
    "reasoning": None,
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
    "max_web_search_calls": 1,
    "max_web_search_results": 1,
    "max_llm_call_retry": 2,
    "max_researcher_iterations": 1,
    "max_concurrent_researchers": 3,
    "interleaved_thinking": True,
    "agent_reasoning": "low",
}
