# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment

- Python `>=3.14`; dependencies managed via `uv` (see `pyproject.toml`, `uv.lock`).
- Requires a running [Ollama](https://ollama.com) instance. The default model is set in `src/config.py` (`MODEL_CONFIG["model_name"]`). Web search uses Ollama's hosted web search and requires `OLLAMA_API_KEY` (loaded via `dotenv.load_dotenv()` from `.env`).
- Install: `uv sync`. The venv lives at `.venv/`.

## Common commands

```bash
# Install / sync deps
uv sync

# Run the full deep research agent (CLI)
uv run python -m src.scripts.run_deep_research_agent
uv run python -m src.scripts.run_deep_research_agent --reasoning high --max-web-search-calls 10 --max-researcher-iterations 3
uv run python -m src.scripts.run_deep_research_agent --no-reasoning

# Run the Streamlit UI
uv run streamlit run src/web_app/streamlit_deepresearch_chat_app.py

# Run individual sub-agents / pipelines (useful when iterating on one stage)
uv run python -m src.scripts.run_streaming_supervisor_agent
uv run python -m src.scripts.run_streaming_research_agent
uv run python -m src.scripts.run_research_writer_agent
uv run python -m src.scripts.run_chart_agent_with_tools
uv run python -m src.scripts.test_section_writer_with_chart
```

There is no test suite, linter, or formatter configured. When adding new entry points, add them under `src/scripts/` and invoke them with `python -m src.scripts.<name>` so the `src.*` package imports resolve.

## Architecture

OpenDeepResearch is a hierarchical multi-agent research system built on **LangGraph**. Each "agent" is a class that exposes `build_agent_graph()` returning a compiled `StateGraph`. Agents are composed by invoking a child graph from a parent node (subgraph pattern), not by linking graphs directly — so each stage can also run standalone from its own script.

### Orchestration layers (top → bottom)

1. **`DeepResearchAgent`** (`src/deep_research_agent/agents/final_deep_research_agent.py`) — top-level orchestrator. Nodes:
   - `clarify_with_user` — uses structured output (`ClarifyWithUser`) to decide whether to end with a clarifying question or proceed.
   - `write_research_brief` — turns chat history into a `ResearchQuestion` brief.
   - `supervisor_subgraph` — constructs and `ainvoke`s a `SupervisorResearchAgent` graph, threading `research_notes` and `visited_urls` through.
   - `final_report_generation` — invokes `ResearchWriterAgent`, then saves markdown + PDF via WeasyPrint (`_save_pdf`).
   - Compiled with an `InMemorySaver` checkpointer; callers pass a `thread_id` in `config.configurable`.

2. **`SupervisorResearchAgent`** (`supervisor_agent.py`) — plans research, fans out to multiple `ResearcherAgent` children concurrently via the `conduct_research` tool, tracks iterations, and decides when to stop (via `research_complete`). Honors `max_researcher_iterations` and `max_concurrent_researchers` from runtime config.

3. **`ResearcherAgent`** (`research_agent.py`) — single researcher loop: think → search → summarize/notes → compress. Uses `search_tool`, `think_tool`, and `research_complete`. Bounded by `max_web_search_calls` / `max_web_search_results` / `max_llm_call_retry` from runtime config.

4. **`ResearchWriterAgent`** (`research_report_writer_agent.py`) — report pipeline:
   - Planner produces a `report_plan` (list of sections).
   - Section writer iterates `current_section_index`, writing sections one at a time; each section may invoke the `ChartAgent` via `create_chart_tool`.
   - Final assembly concatenates `section_texts` into `final_research_text`.

5. **`ChartAgent`** (`chart_agent_code_with_tools.py`) — code-generating agent that writes Plotly code and executes it via `plotly_python_code_executer_tool.py`. Charts land in `output/charts/` as PNG/HTML/JSON and are referenced from the final markdown (the PDF exporter rewrites `src="charts/..."` to absolute `file://` URLs so WeasyPrint can embed them).

### State flow

State classes live in `src/deep_research_agent/state.py`. Each level has its own `TypedDict`/`MessagesState` — **fields are intentionally duplicated** across `AgentState`, `SupervisorState`, `ResearcherState`, `ResearchWriterState` so subgraphs can own their own channels. Two custom reducers matter:

- `research_notes: Annotated[dict, add_dict]` — nested dict (researcher → source → note) merged across concurrent researchers. `add_dict` does a shallow `{**d1, **d2}` merge; keep keys unique (e.g. namespaced by researcher/source) or later writes will clobber earlier ones.
- `visited_urls: Annotated[list, operator.add]` — accumulates across all researchers; used for dedup at the supervisor level.

Messages use LangGraph's `add_messages` reducer. `supervisor_messages` is a separate channel from the user-facing `messages` to keep supervisor tool-call chatter out of the user transcript.

### Configuration

- **Static defaults**: `src/config.py` — `MODEL_CONFIG`, `RESEARCHER_AGENT_CONFIG`, `SUPERVISOR_AGENT_CONFIG`, `FINAL_AGENT_CONFIG`. `ROOT_DIR` and `PROMPTS_DIR` are derived from the file location.
- **Runtime overrides**: passed through `RunnableConfig["configurable"]` (see `run_deep_research_agent.py` for the full key list: `max_web_search_calls`, `max_web_search_results`, `max_llm_call_retry`, `max_researcher_iterations`, `max_concurrent_researchers`, `interleaved_thinking`, `agent_reasoning`). Agents read these from `config` inside their nodes — prefer this over hardcoding.
- **Model selection**: `src/utils/models.py::get_model` wraps `ChatOllama`. Non-`gpt-oss` models (gemma3, llama3.1, granite4, mistral) don't support the string reasoning levels — `get_llm` silently coerces `reasoning=None` for those families, and `get_model` maps `"low"` → `None` / other levels → `True`. Changing the default model in `MODEL_CONFIG` will change behavior across all agents.

### Prompts

All prompts are **Jinja2 templates** under `src/deep_research_agent/prompts/` loaded via `src.utils.helpers.get_prompt_template`. When editing prompt behavior, edit the `.jinja` file — do not inline prompt strings in Python. Templates are rendered with `messages`, `date` (`get_today_str()`), and stage-specific variables.

### Streaming

`src/utils/stream.py` provides `StreamEventProcessor` and `run_async_generator`. It normalizes LangGraph stream events into a `{content_type, content}` payload shape using `src/types.py::ContentType` (`START_STREAM_REASON`, `STREAMING_REASON`, `TOOL_CALLED`, `ASSISTANT_MESSAGE`, `RESPONSE`, etc.). Both the CLI (`run_deep_research_agent.py`) and Streamlit app consume this same event stream — if you add a new event kind, add it to `ContentType` and handle it in both UIs.

### Output

Reports are written to `output/` as `research_report_YYYYMMDD_HHMMSS.md` + `.pdf`. Charts in `output/charts/`. The `output/` directory is created on demand by `DeepResearchAgent._save_report`.
