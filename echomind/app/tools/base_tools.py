"""
工具基类和注册中心

定义所有工具的通用接口，支持工具描述注入 System Prompt。
设计参考: 计划.md §3 模块二 - 工具调用
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional

from pydantic import BaseModel, Field


class ToolParameter(BaseModel):
    """工具参数定义"""
    name: str
    type: str = "string"
    description: str = ""
    required: bool = True


class ToolDefinition(BaseModel):
    """工具定义（用于注入 System Prompt）"""
    name: str
    description: str
    parameters: list[ToolParameter] = Field(default_factory=list)

    def to_prompt_text(self) -> str:
        """转为 LLM 可读的工具描述文本"""
        params_text = ""
        if self.parameters:
            params_text = "\n  参数：" + ", ".join(
                f"{p.name} ({p.type}){' [必填]' if p.required else ''}: {p.description}"
                for p in self.parameters
            )
        return f"**{self.name}**：{self.description}{params_text}"


class BaseTool(ABC):
    """
    工具基类

    所有工具必须实现:
    - name: 工具名称（用于 LLM 调用）
    - description: 工具描述（注入 System Prompt）
    - parameters: 参数列表
    - execute(): 执行方法
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """工具名称"""
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """工具功能描述"""
        ...

    @property
    def parameters(self) -> list[ToolParameter]:
        """工具参数列表"""
        return []

    def get_definition(self) -> ToolDefinition:
        """获取工具定义"""
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters=self.parameters,
        )

    @abstractmethod
    async def execute(self, **kwargs) -> str:
        """
        执行工具

        Args:
            **kwargs: 工具参数

        Returns:
            str: 执行结果文本
        """
        ...


class ToolRegistry:
    """
    工具注册中心

    管理所有可用工具，提供工具发现和调用功能。
    生成注入 System Prompt 的工具描述。
    """

    def __init__(self):
        self._tools: dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        """注册一个工具"""
        self._tools[tool.name] = tool
        print(f"[EchoMind] 工具已注册: {tool.name}")

    def unregister(self, tool_name: str) -> None:
        """取消注册工具"""
        if tool_name in self._tools:
            del self._tools[tool_name]

    def get_tool(self, tool_name: str) -> Optional[BaseTool]:
        """获取工具实例"""
        return self._tools.get(tool_name)

    def list_tools(self) -> list[str]:
        """列出所有已注册工具名称"""
        return list(self._tools.keys())

    def get_tool_definitions(self) -> list[ToolDefinition]:
        """获取所有工具定义列表"""
        return [tool.get_definition() for tool in self._tools.values()]

    def get_tools_prompt(self) -> str:
        """
        生成注入 System Prompt 的工具描述文本

        格式适配 ReAct 范式：
        ```
        可用工具：
        **search_web**：搜索互联网...
        **get_current_time**：获取当前时间...
        ```
        """
        if not self._tools:
            return "（当前无可用工具）"

        lines = ["可用工具："]
        for tool in self._tools.values():
            lines.append(tool.get_definition().to_prompt_text())
        return "\n".join(lines)

    async def execute_tool(self, tool_name: str, **kwargs) -> str:
        """
        执行指定工具

        Args:
            tool_name: 工具名称
            **kwargs: 工具参数

        Returns:
            str: 执行结果

        Raises:
            ValueError: 工具未注册
        """
        tool = self._tools.get(tool_name)
        if not tool:
            raise ValueError(f"工具 '{tool_name}' 未注册。可用工具: {self.list_tools()}")

        try:
            result = await tool.execute(**kwargs)
            return result
        except Exception as e:
            return f"工具执行错误: {str(e)}"