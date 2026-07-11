"""
EchoMind 2.0 外部工具层
Agent 可以调用的工具：搜索、时间、日历等。
"""

from .base_tools import BaseTool, ToolRegistry
from .search_tool import SearchWebTool
from .time_tool import GetCurrentTimeTool

__all__ = [
    "BaseTool",
    "ToolRegistry",
    "SearchWebTool",
    "GetCurrentTimeTool",
]