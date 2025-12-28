import asyncio
import random
import time
from collections import defaultdict
from typing import Dict, Generator, Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import ToolMessage, AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph

from src.types import ContentType, AgentType, ToolName


async def graph_stream_print(
    agent: CompiledStateGraph,
    state: dict,
    config: RunnableConfig | None = None,
    subgraphs=True,
):
    result = None
    async for subgraph, mode, chunk in agent.astream(
        input=state,
        stream_mode=["messages", "custom", "values"],
        config=config,
        subgraphs=subgraphs,
    ):
        # Print custom values
        if mode == "custom":
            msg = chunk.get("message", None)
            if msg:
                print(msg, "\n")

        # Print llm messages from selected nodes
        if mode == "messages":
            if (
                chunk[1]["langgraph_node"] in ["compress_research"]
                and chunk[0].content != ""
            ):
                print(chunk[0].content, end="", flush=True)

        # Save final response
        if mode == "values" and chunk:
            result = chunk

    return result


async def graph_stream_generator(
    agent: CompiledStateGraph, state: dict, config: RunnableConfig | None = None
):
    async for mode, chunk in agent.astream(
        input=state, stream_mode=["messages", "custom", "values"], config=config
    ):
        payload = None
        # Print custom values
        if mode == "custom":
            msg = chunk.get("message", None)
            if msg:
                payload = {"content": msg, "content_type": "updates"}

        # Print llm messages from selected nodes
        if mode == "messages":
            if chunk[0].content != "":

                payload = {
                    "content": chunk[0].content,
                    "content_type": "stream_messages",
                }

        # Save final response
        if mode == "values" and chunk:
            payload = {"content": chunk, "content_type": "response"}
        if payload:
            yield payload


# Streamed response emulator
async def response_generator_emulator(query: str, model: BaseChatModel | None = None):
    updates = [
        "Initializing system components...",
        "Loading configuration files...",
        "Establishing database connection...",
        "Running validation checks...",
        "Startup complete. System is ready.",
    ]

    response = random.choice(
        [
            "Hello there! How can I assist you today?",
            "Hi, human! Is there anything I can help you with?",
            "Do you need help?",
        ]
    )
    # Yield updates
    for i in range(len(updates)):
        update = random.choice(updates)
        payload = {"content": update, "content_type": "updates"}
        yield payload
        time.sleep(0.5)
    if model:
        async for chunk in model.astream(query):
            payload = {"content": chunk, "content_type": "stream_messages"}
            yield payload

    else:
        for word in response.split():
            payload = {"content": word + " ", "content_type": "stream_messages"}
            yield payload
            time.sleep(0.05)


def run_async_generator(async_gen):
    """Safely convert an async generator to sync for Streamlit."""
    try:
        # Check if there is already a running loop in this thread
        loop = asyncio.get_event_loop()
    except RuntimeError:
        # If no loop exists, create a new one
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    while True:
        try:
            # We use the loop to run the 'anext' task until it's finished
            yield loop.run_until_complete(async_gen.__anext__())
        except StopAsyncIteration:
            break
        except Exception as e:
            # Log specific errors here if needed
            raise e


class StreamEventProcessor:
    """
    Handles the raw events coming from LangGraph and converts them
    into clean Payloads for the UI.
    """

    def __init__(self):
        self.reason_streaming_dict = defaultdict(lambda: False)

    @staticmethod
    def _get_agent_type(agent_path: list) -> str:
        if not agent_path:
            return AgentType.MAIN
        elif "supervisor_subgraph" in agent_path[-1]:
            return AgentType.SUPERVISOR
        return AgentType.RESEARCHER

    def _get_payload_from_tool_call(
        self, tool_call: dict, agent_type: str
    ) -> Optional[Dict]:
        name = tool_call["name"]
        args = tool_call.get("args", {})
        content = None

        if name == ToolName.THINK:
            content = f"Agent Reflection: {args.get('reflection')}"
        elif name == ToolName.RESEARCH:
            content = f"Delegating to Researcher subagent to research on topic: {args.get('research_topic')}"
        elif name == ToolName.WEB_SEARCH:
            content = f"Conducting web search on topic: {args.get('query')}"
        elif name == ToolName.COMPLETE:
            content = "Generating final report"
        else:
            return None

        return {
            "content": content,
            "content_type": ContentType.TOOL_CALLED,
            "tool_name": name,
            "agent_type": agent_type,
        }

    def _extract_sub_agent_messages(self, chunk: dict, agent_type: str) -> list:
        """Helper to extract message lists from different dictionary structures."""
        # Determine the key based on agent type
        msg_key = (
            "supervisor_messages"
            if agent_type == AgentType.SUPERVISOR
            else "researcher_messages"
        )

        # Determine the node source (LLM vs Tool)
        if "llm_call" in chunk:
            return chunk["llm_call"].get(msg_key, [])
        elif "tool_node" in chunk:
            return chunk["tool_node"].get(msg_key, [])

        return []

    def _handle_main_updates(
        self, chunk: dict, agent_type: str
    ) -> Generator[Dict, None, None]:
        """Handles top-level updates (Clarification requests or Final Reports)."""
        clarify = chunk.get("clarify_with_user")
        final = chunk.get("final_report_generation")

        content = None
        if clarify and clarify.get("messages"):
            content = clarify["messages"][0].content
        elif final:
            content = final.get("final_report", "")

        if content:
            yield {
                "content": content,
                "content_type": ContentType.ASSISTANT_MESSAGE,
                "agent_type": agent_type,
            }

    def _handle_sub_agent_updates(
        self, chunk: dict, agent_type: str
    ) -> Generator[Dict, None, None]:
        """Handles sub-agent tool calls and completions."""
        messages = self._extract_sub_agent_messages(chunk, agent_type)

        for msg in messages:
            # 1. Handle Tool Calls (The request to do work)
            if isinstance(msg, AIMessage) and msg.tool_calls:
                for tc in msg.tool_calls:
                    payload = self._get_payload_from_tool_call(tc, agent_type)
                    if payload:
                        yield payload

            # 2. Handle Tool Output (The work is done)
            if isinstance(msg, ToolMessage):
                yield {
                    "content": "",
                    "content_type": ContentType.TOOL_CALL_COMPLETE,
                    "tool_name": msg.name,
                    "agent_type": agent_type,
                }

    def _handle_reasoning_stream(
        self, chunk, node_name: str, agent_type: str
    ) -> Optional[Dict]:
        """Detects if reasoning (Chain of Thought) is starting, ongoing, or stopping."""
        msg_chunk, metadata = chunk
        key = f"{node_name}_{agent_type}"

        if metadata.get("langgraph_node") != node_name:
            return None

        reasoning = msg_chunk.additional_kwargs.get("reasoning_content", "")

        # Start or Continue Reasoning
        if reasoning:
            if not self.reason_streaming_dict[key]:
                self.reason_streaming_dict[key] = True
                c_type = ContentType.START_STREAM_REASON
            else:
                c_type = ContentType.STREAMING_REASON

            return {
                "content": reasoning,
                "content_type": c_type,
                "agent_type": agent_type,
                "node_name": node_name,
            }

        # Stop Reasoning
        if (msg_chunk.content or msg_chunk.tool_calls) and self.reason_streaming_dict[
            key
        ]:
            self.reason_streaming_dict[key] = False
            return {
                "content": "",
                "content_type": "stop_stream_reason",
                "agent_type": agent_type,
                "node_name": node_name,
            }

        return None

    async def process_stream(
        self, agent_graph: CompiledStateGraph, state: dict, config: RunnableConfig
    ):
        """Main generator that yields clean payloads."""
        async for subagent, mode, chunk in agent_graph.astream(
            input=state,
            stream_mode=["messages", "updates", "values", "custom"],
            config=config,
            subgraphs=True,
        ):
            agent_type = self._get_agent_type(subagent)

            # --- Modularized Update Handling ---
            if mode == "updates":
                if agent_type == AgentType.MAIN:
                    for payload in self._handle_main_updates(chunk, agent_type):
                        yield payload
                else:
                    for payload in self._handle_sub_agent_updates(chunk, agent_type):
                        yield payload

            # --- Message Handling (Reasoning) ---
            if mode == "messages":
                payload = self._handle_reasoning_stream(chunk, "llm_call", agent_type)
                if payload:
                    yield payload

            if mode == "custom":
                if chunk.get("content_type") == ContentType.COMPRESSION_START:
                    yield {
                        "content": "Compressing research.",
                        "content_type": ContentType.TOOL_CALLED,
                        "tool_name": chunk["node_name"],
                        "agent_type": agent_type,
                    }
                elif chunk.get("content_type") == ContentType.COMPRESSION_STOP:
                    yield {
                        "content": "",
                        "content_type": ContentType.TOOL_CALL_COMPLETE,
                        "tool_name": chunk["node_name"],
                        "agent_type": agent_type,
                    }

            # --- Value Handling ---
            if mode == "values" and chunk:
                yield {"content": chunk, "content_type": "response"}
