# OpenDeepResearch

OpenDeepResearch is a powerful autonomous research agent that performs multi-step deep research on complex topics. It leverages local Large Language Models via Ollama, conducts web searches to gather real-time information, and generates comprehensive research reports with visualizations.

## Features

- **Autonomous Deep Research**: Plans and executes multi-step research with clarifying questions
- **Multi-Agent Architecture**: Supervisor coordinates multiple researcher agents for comprehensive coverage
- **Web Search Integration**: Real-time information gathering using Ollama's web search
- **Research Note-Taking**: Structured note collection and synthesis across research iterations
- **Chart Generation**: Automatic Plotly chart creation for data visualization
- **Report Generation**: Professional reports in Markdown and PDF formats
- **Local Privacy**: Runs entirely with local models (except web search queries)
- **Dual Interface**: Streamlit web app and command-line interface

![Deep Research Agent UI](assets/app_screenshot.png)

## Architecture

The agent uses a hierarchical multi-agent system built with LangGraph:

```
DeepResearchAgent (Main Orchestrator)
├── Clarification Phase → Asks user clarifying questions
├── Research Brief → Generates structured research plan
├── SupervisorAgent → Coordinates research
│   └── ResearcherAgents (concurrent) → Conduct web searches, take notes
└── ResearchWriterAgent → Generates final report
    ├── Report Planner → Plans report structure
    ├── SectionWriterAgents → Write sections with charts
    │   └── ChartAgent → Creates Plotly visualizations
    └── Final Document Assembly
```

![Agent Architecture](assets/overall_agent.png)

## Hardware Requirements

- **Tested on**: Mac Studio M3 Ultra (96GB RAM, 28 CPU cores, 60 GPU cores)
- **Minimum Requirements**:
  - **RAM/VRAM**: 16GB+ for the default `gpt-oss` model (4-bit quantized, ~14GB)
  - **Storage**: 15-20GB free space for model weights
  - **Recommended**: Apple Silicon Mac or NVIDIA GPU with 16GB+ VRAM

## Prerequisites

- **Python**: >= 3.14
- **Ollama**: Installed and running ([Download Ollama](https://ollama.com))

## Installation

### Quick Start

```bash
# Clone the repository
git clone https://github.com/sgauravm/OpenDeepResearch.git
cd OpenDeepResearch

# Run the interactive setup script
python setup_interactive.py
```

The setup script will:
- Install `uv` package manager (if missing)
- Install Python dependencies
- Set up Ollama models
- Configure your Ollama API key
- Launch the app

### Manual Setup

1. **Install `uv`** (recommended package manager)
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **Install Dependencies**
   ```bash
   uv sync
   ```

3. **Pull the Required Model**
   ```bash
   ollama pull gpt-oss
   ```

4. **Configure Web Search API Key**

   Get your API key from [Ollama settings](https://ollama.com/settings/keys) and set it:
   ```bash
   # Mac/Linux - add to ~/.zshrc or ~/.bashrc
   export OLLAMA_API_KEY="your_api_key_here"

   # Windows PowerShell
   setx OLLAMA_API_KEY "your_api_key_here"
   ```

## Usage

### Streamlit Web App

```bash
# Activate virtual environment
source .venv/bin/activate  # Mac/Linux
# or
.venv\Scripts\activate     # Windows

# Run the app
streamlit run src/web_app/streamlit_deepresearch_chat_app.py

# Or using uv directly
uv run streamlit run src/web_app/streamlit_deepresearch_chat_app.py
```

### Command Line Interface

```bash
# Run with default settings
python -m src.scripts.run_deep_research_agent

# Run with high reasoning level
python -m src.scripts.run_deep_research_agent --reasoning high

# Run with custom configuration
python -m src.scripts.run_deep_research_agent \
  --max-web-search-calls 10 \
  --max-researcher-iterations 3

# Hide reasoning output
python -m src.scripts.run_deep_research_agent --no-reasoning
```

**CLI Options:**
| Option | Default | Description |
|--------|---------|-------------|
| `--reasoning` | medium | Reasoning level (low/medium/high) |
| `--interleaved-thinking` | True | Enable interleaved thinking |
| `--max-web-search-calls` | 5 | Max web searches per researcher |
| `--max-web-search-results` | 3 | Max results per search |
| `--max-researcher-iterations` | 2 | Max research iterations |
| `--no-reasoning` | False | Hide reasoning output |

## Configuration

Edit `src/config.py` to customize agent behavior:

```python
MODEL_CONFIG = {
    "model_name": "gpt-oss",
    "temperature": 0,
    "reasoning": "medium",
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
```

## Output

Research reports are saved to the `output/` directory:
- `research_report_YYYYMMDD_HHMMSS.md` - Markdown report
- `research_report_YYYYMMDD_HHMMSS.pdf` - PDF report
- `output/charts/` - Generated chart images (PNG, HTML, JSON)

## Project Structure

```
OpenDeepResearch/
├── src/
│   ├── config.py                    # Configuration settings
│   ├── types.py                     # Type definitions
│   ├── deep_research_agent/
│   │   ├── agents/
│   │   │   ├── final_deep_research_agent.py   # Main orchestrator
│   │   │   ├── supervisor_agent.py            # Research coordinator
│   │   │   ├── research_agent.py              # Web search researcher
│   │   │   ├── research_report_writer_agent.py # Report generator
│   │   │   ├── section_writer_agent.py        # Section writer with charts
│   │   │   └── chart_agent_code_with_tools.py # Chart generator
│   │   ├── tools/
│   │   │   ├── search_tool.py          # Web search tool
│   │   │   ├── create_chart_tool.py    # Chart creation tool
│   │   │   └── ...
│   │   ├── prompts/                    # Jinja2 prompt templates
│   │   └── state.py                    # Agent state definitions
│   ├── web_app/
│   │   └── streamlit_deepresearch_chat_app.py  # Streamlit UI
│   ├── scripts/
│   │   └── run_deep_research_agent.py  # CLI script
│   └── utils/
│       ├── models.py                   # Model utilities
│       ├── stream.py                   # Streaming utilities
│       └── helpers.py                  # Helper functions
├── output/                             # Generated reports and charts
├── assets/                             # Screenshots and images
└── pyproject.toml                      # Project dependencies
```

## How It Works

1. **User Input**: Enter a research query
2. **Clarification**: Agent asks clarifying questions to understand scope
3. **Research Brief**: Generates a structured research plan
4. **Research Phase**: Supervisor dispatches multiple researchers who:
   - Conduct web searches
   - Analyze and summarize findings
   - Take structured notes
5. **Report Generation**:
   - Planner creates report outline
   - Section writers draft each section
   - Charts are generated for data visualization
   - Final document is assembled
6. **Output**: Report saved as Markdown and PDF

## Dependencies

Key dependencies (see `pyproject.toml` for full list):
- `langgraph` - Agent orchestration
- `langchain-ollama` - Ollama integration
- `streamlit` - Web interface
- `plotly` - Chart generation
- `weasyprint` - PDF generation
- `markdown` - Markdown processing
