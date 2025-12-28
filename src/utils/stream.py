import random
import time
from langgraph.graph.state import CompiledStateGraph
from langchain_core.runnables import RunnableConfig
from langchain_core.language_models.chat_models import BaseChatModel
import asyncio


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
