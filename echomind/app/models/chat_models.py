"""EchoMind 2.0 聊天数据模型 - ReAct 范式"""

from datetime import datetime, timezone
from typing import Optional
from pydantic import BaseModel, Field


class ToolCall(BaseModel):
    tool_name: str = Field(...)
    arguments: dict = Field(default_factory=dict)
    result: str = Field(default="")
    called_at: float = Field(default_factory=lambda: datetime.now(timezone.utc).timestamp())


class ThoughtStep(BaseModel):
    thought: str = Field(default="")
    action: Optional[str] = Field(default=None)
    observation: Optional[str] = Field(default=None)
    tool_call: Optional[ToolCall] = Field(default=None)
    step_index: int = Field(default=0)


class ChatMessage(BaseModel):
    role: str = Field(default="user")
    content: str = Field(...)
    timestamp: float = Field(default_factory=lambda: datetime.now(timezone.utc).timestamp())
    metadata: dict = Field(default_factory=dict)


class ChatRequest(BaseModel):
    user_id: str = Field(...)
    session_id: str = Field(default="default")
    message: str = Field(..., min_length=1)
    stream: bool = Field(default=True)
    enable_memory: bool = Field(default=True)
    enable_tools: bool = Field(default=True)


class ChatResponse(BaseModel):
    user_id: str = Field(...)
    session_id: str = Field(default="default")
    reply: str = Field(...)
    thought_chain: list[ThoughtStep] = Field(default_factory=list)
    retrieved_memories: list[dict] = Field(default_factory=list)
    tool_calls: list[ToolCall] = Field(default_factory=list)
    response_time_ms: float = Field(default=0.0)
    timestamp: float = Field(default_factory=lambda: datetime.now(timezone.utc).timestamp())