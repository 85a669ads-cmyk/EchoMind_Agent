"""
EchoMind 2.0 Agent 核心引擎

实现 ReAct (Reasoning + Acting) 范式的智能体。
核心流程:
1. 检索相关长期记忆
2. 构建包含记忆上下文的 System Prompt
3. Thought → Action → Observation 循环
4. 流式输出最终回复

设计参考: 计划.md §3 模块二 - ReAct 范式、流式输出
"""

from __future__ import annotations

import asyncio
import re
import time
from typing import AsyncGenerator, Optional

from app.core.config import EchoMindConfig
from app.core.llm_client import LLMClient
from app.models.chat_models import (
    ChatRequest,
    ChatResponse,
    ThoughtStep,
    ToolCall,
)
from app.models.memory_models import WorkingMemoryItem, MemoryPolarity
from app.memory.working_memory import WorkingMemory
from app.memory.long_term_memory import LongTermMemoryManager
from app.memory.consolidator import MemoryConsolidator
from app.memory.conflict_resolver import ConflictResolver
from app.tools.base_tools import ToolRegistry


class EchoMindAgent:
    """
    EchoMind 智能体 —— ReAct 范式

    核心特性:
    - 记忆感知对话: 每次对话前自动检索相关长期记忆
    - ReAct 推理: Thought → Action → Observation 循环
    - 工具调用: 在需要时调用外部工具
    - 流式输出: 支持 token 级别的流式响应
    - 记忆巩固: 对话自动转化为长期记忆
    """

    # ReAct System Prompt 模板
    SYSTEM_PROMPT_TEMPLATE = """你是一个名为 EchoMind 的智能记忆助手。你拥有持久记忆能力，能记住与用户的过往交互。

## 你的核心能力
1. **记忆检索**: 你能访问用户的长期记忆库，了解他们的偏好、习惯和历史对话
2. **工具调用**: 你可以使用以下工具获取实时信息
3. **深度思考**: 你会先思考再回答，展示清晰的推理过程

## 检索到的用户记忆
{memory_context}

## {tools_prompt}

## 回答格式 (ReAct 范式)
在回答用户问题前，请按以下格式输出思考过程：
```
Thought: [你对问题的分析和思考]
Action: [如果需要使用工具，在此处写出工具调用，格式: tool_name(param1=value1, ...)]
Observation: [工具返回的结果]
... (可以重复 Thought-Action-Observation 多次)
Final Answer: [你的最终回答]
```

## 重要规则
1. 只有当确实需要实时信息时才调用工具，不要滥用
2. 优先基于已检索的记忆回答问题
3. 如果记忆中有相关信息，主动引用（例如："根据我之前的记录，你..."）
4. 对于不确定的信息，请诚实说明
5. 使用中文与用户对话（除非用户使用其他语言）
6. 最终回答要友好、自然，不要包含 Thought/Action/Observation 格式
"""

    def __init__(
        self,
        config: EchoMindConfig,
        llm_client: LLMClient,
        working_memory: WorkingMemory,
        long_term_memory: LongTermMemoryManager,
        tool_registry: ToolRegistry,
        memory_consolidator: Optional[MemoryConsolidator] = None,
        conflict_resolver: Optional[ConflictResolver] = None,
    ):
        """
        Args:
            config: 系统配置
            llm_client: LLM 客户端
            working_memory: 短期工作记忆
            long_term_memory: 长期记忆管理器
            tool_registry: 工具注册中心
            memory_consolidator: 记忆巩固器（可选）
            conflict_resolver: 冲突解决器（可选）
        """
        self.config = config
        self.llm = llm_client
        self.working_memory = working_memory
        self.long_term_memory = long_term_memory
        self.tool_registry = tool_registry
        self.memory_consolidator = memory_consolidator
        self.conflict_resolver = conflict_resolver

    async def process(
        self, request: ChatRequest
    ) -> ChatResponse:
        """
        处理用户聊天请求（非流式）

        Args:
            request: 聊天请求

        Returns:
            ChatResponse: 包含回复、思考链、记忆检索结果的响应
        """
        start_time = time.time()
        thought_chain: list[ThoughtStep] = []
        tool_calls: list[ToolCall] = []

        # 1. 检索相关长期记忆
        retrieved_memories = []
        if request.enable_memory:
            retrieved_memories = await self._retrieve_memories(
                user_id=request.user_id,
                query=request.message,
            )

        # 2. 构建 System Prompt
        system_prompt = self._build_system_prompt(
            memories=retrieved_memories,
            include_tools=request.enable_tools,
        )

        # 3. 构建消息列表
        messages = [
            {"role": "system", "content": system_prompt},
        ]

        # 添加短期工作记忆作为上下文
        recent_working = self.working_memory.get_recent(
            user_id=request.user_id,
            session_id=request.session_id,
            n=10,
        )
        for wm in recent_working:
            role = "user" if wm.turn_index % 2 == 0 else "assistant"
            messages.append({"role": role, "content": wm.content})

        # 添加当前用户消息
        messages.append({"role": "user", "content": request.message})

        # 4. 调用 LLM (ReAct 主循环)
        reply = ""
        step_index = 0

        for _ in range(self.config.max_react_steps):
            step_index += 1

            raw_response = await self.llm.chat(
                messages=messages,
                model=self.config.llm_model_complex,
                temperature=self.config.llm_temperature,
                max_tokens=self.config.llm_max_tokens,
            )

            # 解析 ReAct 格式
            thought = self._extract_section(raw_response, "Thought")
            action = self._extract_section(raw_response, "Action")
            observation = self._extract_section(raw_response, "Observation")
            final_answer = self._extract_section(raw_response, "Final Answer")

            thought_step = ThoughtStep(
                thought=thought or "",
                action=action,
                observation=observation,
                step_index=step_index,
            )
            thought_chain.append(thought_step)

            # 如果有工具调用
            if action and request.enable_tools:
                tool_name, tool_args = self._parse_action(action)
                if tool_name and self.tool_registry.get_tool(tool_name):
                    tool_result = await self.tool_registry.execute_tool(
                        tool_name, **tool_args
                    )
                    tool_call = ToolCall(
                        tool_name=tool_name,
                        arguments=tool_args,
                        result=tool_result,
                    )
                    tool_calls.append(tool_call)
                    thought_step.tool_call = tool_call
                    thought_step.observation = tool_result

                    # 将工具结果添加到消息中，继续循环
                    messages.append({"role": "assistant", "content": raw_response})
                    messages.append({"role": "system", "content": f"工具结果: {tool_result}"})
                    continue

            # 有最终回答，结束循环
            if final_answer:
                reply = final_answer
                break

            # 没有明确 Final Answer，将整段作为回复
            reply = self._extract_final_answer(raw_response)
            break

        # 5. 保存短期工作记忆
        if request.enable_memory:
            self.working_memory.add(
                user_id=request.user_id,
                content=f"User: {request.message}",
                session_id=request.session_id,
                importance=0.6,
            )
            self.working_memory.add(
                user_id=request.user_id,
                content=f"EchoMind: {reply[:200]}",
                session_id=request.session_id,
                importance=0.6,
            )

        # 6. 计算响应时间
        response_time_ms = (time.time() - start_time) * 1000

        return ChatResponse(
            user_id=request.user_id,
            session_id=request.session_id,
            reply=reply,
            thought_chain=thought_chain,
            retrieved_memories=[
                {"content": m.memory.content, "score": m.similarity_score}
                for m in retrieved_memories
            ],
            tool_calls=tool_calls,
            response_time_ms=round(response_time_ms, 2),
        )

    async def process_stream(
        self, request: ChatRequest
    ) -> AsyncGenerator[str, None]:
        """
        流式处理用户聊天请求

        Args:
            request: 聊天请求

        Yields:
            str: 逐 token 输出文本
        """
        # 1. 检索记忆
        retrieved_memories = []
        memory_context_text = "（未检索到相关记忆）"

        if request.enable_memory:
            yield "🧠 正在检索长期记忆...\n"
            try:
                retrieved_memories = await self._retrieve_memories(
                    user_id=request.user_id,
                    query=request.message,
                )
                if retrieved_memories:
                    memory_context_text = "\n".join(
                        f"- [{m.similarity_score:.2f}] {m.memory.content}"
                        for m in retrieved_memories[:5]
                    )
                    yield f"✅ 找到 {len(retrieved_memories)} 条相关记忆\n"
                else:
                    yield "📭 未找到相关长期记忆\n"
            except Exception as e:
                yield f"⚠️ 记忆检索异常: {e}\n"

        # 2. 构建消息
        system_prompt = self._build_system_prompt(
            memories=retrieved_memories,
            include_tools=request.enable_tools,
        )

        messages = [
            {"role": "system", "content": system_prompt},
        ]

        recent_working = self.working_memory.get_recent(
            user_id=request.user_id,
            session_id=request.session_id,
            n=10,
        )
        for wm in recent_working:
            role = "user" if wm.turn_index % 2 == 0 else "assistant"
            messages.append({"role": role, "content": wm.content})

        messages.append({"role": "user", "content": request.message})

        # 3. 流式输出
        yield "💬 EchoMind 回复:\n"

        full_reply = ""
        async for token in self.llm.chat_stream(
            messages=messages,
            model=self.config.llm_model_complex,
            temperature=self.config.llm_temperature,
            max_tokens=self.config.llm_max_tokens,
        ):
            full_reply += token
            yield token

        # 4. 保存短期记忆
        if request.enable_memory:
            self.working_memory.add(
                user_id=request.user_id,
                content=f"User: {request.message}",
                session_id=request.session_id,
                importance=0.6,
            )
            self.working_memory.add(
                user_id=request.user_id,
                content=f"EchoMind: {full_reply[:200]}",
                session_id=request.session_id,
                importance=0.6,
            )

    # --------------------- 辅助方法 ---------------------

    async def _retrieve_memories(
        self, user_id: str, query: str, top_k: int = 10
    ) -> list:
        """检索长期记忆"""
        from app.models.memory_models import MemorySearchResult
        try:
            results = self.long_term_memory.search(
                query=query,
                user_id=user_id,
                top_k=top_k,
                similarity_threshold=0.3,
            )
            # 记录访问
            for r in results:
                r.memory.record_access()
            return results
        except Exception as e:
            print(f"[EchoMind] 记忆检索异常: {e}")
            return []

    def _build_system_prompt(
        self,
        memories: list,
        include_tools: bool = True,
    ) -> str:
        """构建包含记忆上下文的 System Prompt"""
        # 构建记忆上下文
        if memories:
            memory_lines = []
            for i, mem in enumerate(memories[:5], 1):
                memory_lines.append(
                    f"{i}. [相似度:{mem.similarity_score:.2f}] {mem.memory.content}"
                )
            memory_context = "\n".join(memory_lines)
        else:
            memory_context = "（暂无相关长期记忆）"

        # 工具描述
        tools_prompt = self.tool_registry.get_tools_prompt() if include_tools else ""

        return self.SYSTEM_PROMPT_TEMPLATE.format(
            memory_context=memory_context,
            tools_prompt=tools_prompt,
        )

    @staticmethod
    def _extract_section(text: str, section: str) -> Optional[str]:
        """从 ReAct 格式输出中提取指定段落"""
        pattern = rf'{section}:\s*(.+?)(?=\n[A-Z]|$)'
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return None

    @staticmethod
    def _extract_final_answer(text: str) -> str:
        """提取最终答案（兼容无 ReAct 格式的输出）"""
        # 尝试提取 Final Answer
        final = EchoMindAgent._extract_section(text, "Final Answer")
        if final:
            return final

        # 尝试移除 Thought/Action/Observation 行后的纯文本
        lines = text.split("\n")
        clean_lines = []
        for line in lines:
            if re.match(r'^(Thought|Action|Observation):', line, re.IGNORECASE):
                continue
            clean_lines.append(line)
        cleaned = "\n".join(clean_lines).strip()
        return cleaned if cleaned else text

    @staticmethod
    def _parse_action(action_text: str) -> tuple[Optional[str], dict]:
        """
        解析工具调用 Action

        支持格式: tool_name(param1=value1, param2=value2)

        Returns:
            tuple[Optional[str], dict]: (工具名称, 参数字典)
        """
        action_text = action_text.strip()

        # 格式: tool_name(key1=val1, key2=val2)
        match = re.match(r'(\w+)\((.+)\)', action_text)
        if match:
            tool_name = match.group(1)
            args_str = match.group(2)

            # 解析参数
            kwargs = {}
            arg_pairs = re.findall(r'(\w+)\s*=\s*["\']?([^"\',]+)["\']?', args_str)
            for key, value in arg_pairs:
                kwargs[key] = value.strip()

            # 如果只有一个位置参数
            if not kwargs and args_str.strip():
                kwargs["query"] = args_str.strip().strip('"\'')
                if not kwargs["query"]:
                    kwargs = {}

            return tool_name, kwargs

        # 格式: tool_name "query text"
        match = re.match(r'(\w+)\s+"?([^"]+)"?', action_text)
        if match:
            return match.group(1), {"query": match.group(2).strip()}

        return None, {}