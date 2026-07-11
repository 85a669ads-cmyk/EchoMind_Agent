"""
EchoMind 2.0 数据模型层
定义系统中所有核心数据结构 (Pydantic Models)
"""

from .memory_models import (
    MemoryItem,
    WorkingMemoryItem,
    LongTermMemoryItem,
    MemorySearchResult,
    MemoryConsolidationResult,
    MemoryConflictResult,
    MemoryStats,
    MemoryPolarity,
)

from .chat_models import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    ThoughtStep,
    ToolCall,
)

from .user_models import (
    UserProfile,
    UserPreferences,
)

__all__ = [
    # Memory Models
    "MemoryItem",
    "WorkingMemoryItem",
    "LongTermMemoryItem",
    "MemorySearchResult",
    "MemoryConsolidationResult",
    "MemoryConflictResult",
    "MemoryStats",
    "MemoryPolarity",
    # Chat Models
    "ChatMessage",
    "ChatRequest",
    "ChatResponse",
    "ThoughtStep",
    "ToolCall",
    # User Models
    "UserProfile",
    "UserPreferences",
]