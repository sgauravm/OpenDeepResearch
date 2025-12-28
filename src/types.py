from typing import Any, TypedDict, Required, NotRequired
from enum import Enum


class ContentType(str, Enum):
    START_STREAM_REASON = "start_stream_reason"
    STREAMING_REASON = "streaming_reason"
    STOP_STREAM_REASON = "stop_stream_reason"
    TOOL_CALL_COMPLETE = "tool_call_complete"
    ASSISTANT_MESSAGE = "assistant_message"
    TOOL_CALLED = "tool_called"
    RESPONSE = "response"
    COMPRESSION_START = "compression_start"
    COMPRESSION_STOP = "compression_stop"


class Payload(TypedDict):
    content: Required[Any]
    content_type: Required[ContentType]
    agent_type: Required[str]
    node_name: NotRequired[str]
    tool_name: NotRequired[str]


class AgentType(str, Enum):
    MAIN = "main"
    SUPERVISOR = "supervisor"
    RESEARCHER = "researcher"


class ToolName(str, Enum):
    THINK = "think_tool"
    RESEARCH = "conduct_research"
    WEB_SEARCH = "web_search"
    COMPLETE = "research_complete"
