"""
记忆巩固器 (Memory Consolidator)

当 WorkingMemory 达到阈值时，将零散的短期对话总结为结构化的长期记忆。
调用 Qwen-Plus 进行智能摘要，并自动判断重要性评分和情感极性。

设计参考: 计划.md §3 模块一 - 记忆巩固机制
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from typing import Optional

from app.models.memory_models import (
    WorkingMemoryItem,
    LongTermMemoryItem,
    MemoryConsolidationResult,
    MemoryPolarity,
)


class MemoryConsolidator:
    """
    记忆巩固器

    功能:
    1. 接收一批短期记忆项，调用 LLM 生成摘要
    2. 自动判断记忆类型（偏好/事实/事件/技能）和重要性
    3. 检测并合并与已有长期记忆的重复项
    4. 生成结构化长期记忆并存入 LongTermMemoryManager
    """

    # 巩固 Prompt 模板
    CONSOLIDATION_PROMPT = """你是一个记忆管理助手。请分析以下用户的短期对话记录，将其总结为一条结构化的长期记忆。

**对话记录**：
{conversation_text}

**要求**：
1. 提取出关于用户的长期有用信息（如偏好、事实、习惯、重要事件等）
2. 忽略闲聊、问候等无长期价值的内容
3. 判断记忆类别（preference/fact/event/skill）
4. 评估重要性（0-1，越重要越高）
5. 判断情感极性（positive/negative/neutral）
6. 生成1-3个标签

**请严格按以下 JSON 格式输出**（不要包含其他文字）：
```json
{{
  "content": "一句话总结的长期记忆内容",
  "category": "preference|fact|event|skill",
  "importance": 0.0,
  "polarity": "positive|negative|neutral",
  "tags": ["标签1", "标签2"],
  "summary": "更详细的总结，2-3句话"
}}
```

如果没有值得长期记忆的信息，content 设为空字符串。
"""

    MERGE_CHECK_PROMPT = """你是一个记忆管理助手。请判断以下两条记忆是否可以合并。

**已有记忆**：{existing_content}
**新记忆**：{new_content}

**要求**：判断两条记忆描述的是否是同一信息（同一偏好/事实的不同表述）。

**请严格按以下 JSON 格式输出**：
```json
{{
  "should_merge": true或false,
  "merged_content": "如果合并，输出合并后的内容（保留最新最准确的信息）"
}}
```
"""

    def __init__(
        self,
        llm_call_func: Optional[callable] = None,
        long_term_memory_manager=None,
        model_name: str = "qwen-plus",
    ):
        """
        Args:
            llm_call_func: LLM 调用函数，签名为 async (system_prompt, user_prompt) -> str
            long_term_memory_manager: LongTermMemoryManager 实例
            model_name: 使用的模型名称（qwen-plus 推荐用于记忆提取）
        """
        self.llm_call_func = llm_call_func
        self.long_term_memory = long_term_memory_manager
        self.model_name = model_name

        # 统计
        self.total_consolidations = 0
        self.total_merges = 0

    async def consolidate(
        self,
        user_id: str,
        working_memories: list[WorkingMemoryItem],
    ) -> Optional[MemoryConsolidationResult]:
        """
        将短期记忆巩固为长期记忆

        Args:
            user_id: 用户ID
            working_memories: 待巩固的短期记忆列表

        Returns:
            MemoryConsolidationResult 或 None（无值得记忆的内容）
        """
        if not working_memories:
            return None

        # 构建对话文本
        conversation_parts = []
        for item in working_memories:
            conversation_parts.append(
                f"[轮次{item.turn_index}] {item.content}"
            )
        conversation_text = "\n".join(conversation_parts)

        # 调用 LLM 进行巩固总结
        prompt = self.CONSOLIDATION_PROMPT.format(
            conversation_text=conversation_text
        )
        llm_response = await self._call_llm(
            system_prompt="你是专业的记忆管理助手，擅长从对话中提取结构化信息。",
            user_prompt=prompt,
        )

        if not llm_response:
            return None

        # 解析 JSON 响应
        parsed = self._parse_json_response(llm_response)
        if not parsed or not parsed.get("content"):
            return None  # 无值得长期记忆的内容

        # 创建长期记忆项
        long_term_memory = LongTermMemoryItem(
            user_id=user_id,
            content=parsed["content"],
            category=parsed.get("category", "general"),
            importance=float(parsed.get("importance", 0.5)),
            polarity=MemoryPolarity(parsed.get("polarity", "neutral")),
            tags=parsed.get("tags", []),
        )

        # 检查是否与已有记忆重复/可合并
        merged_count = 0
        if self.long_term_memory:
            merged_count = await self._check_and_merge(
                user_id=user_id,
                new_memory=long_term_memory,
            )

        # 存储长期记忆
        if self.long_term_memory and not long_term_memory.memory_id:
            self.long_term_memory.add_memory(long_term_memory)

        self.total_consolidations += 1
        if merged_count > 0:
            self.total_merges += merged_count

        return MemoryConsolidationResult(
            source_items=working_memories,
            consolidated_memory=long_term_memory,
            summary=parsed.get("summary", parsed["content"]),
            merged_count=merged_count,
        )

    async def _check_and_merge(
        self, user_id: str, new_memory: LongTermMemoryItem
    ) -> int:
        """
        检查新记忆是否与已有记忆冲突或可合并

        Args:
            user_id: 用户ID
            new_memory: 新记忆

        Returns:
            int: 合并的记忆数
        """
        # 搜索相似记忆
        similar = self.long_term_memory.search(
            query=new_memory.content,
            user_id=user_id,
            top_k=3,
            similarity_threshold=0.85,  # 高阈值，只检查高度相似的
        )

        merged_count = 0
        for result in similar:
            existing = result.memory

            # 调用 LLM 判断是否可合并
            check_prompt = self.MERGE_CHECK_PROMPT.format(
                existing_content=existing.content,
                new_content=new_memory.content,
            )
            check_response = await self._call_llm(
                system_prompt="你是记忆管理助手，判断记忆合并。",
                user_prompt=check_prompt,
            )

            if check_response:
                check_parsed = self._parse_json_response(check_response)
                if check_parsed and check_parsed.get("should_merge"):
                    # 合并：更新已有记忆
                    existing.content = check_parsed.get(
                        "merged_content", new_memory.content
                    )
                    existing.importance = max(
                        existing.importance, new_memory.importance
                    )
                    existing.tags = list(
                        set(existing.tags + new_memory.tags)
                    )
                    existing.record_access()
                    self.long_term_memory.update_memory(existing)
                    merged_count += 1
                    break  # 合并一次即可

        return merged_count

    async def _call_llm(
        self, system_prompt: str, user_prompt: str
    ) -> Optional[str]:
        """
        调用 LLM

        Args:
            system_prompt: 系统提示
            user_prompt: 用户提示

        Returns:
            str 或 None
        """
        if self.llm_call_func:
            try:
                return await self.llm_call_func(system_prompt, user_prompt)
            except Exception as e:
                print(f"[EchoMind] LLM 调用失败: {e}")
                return None

        # 回退：尝试直接使用 DashScope
        try:
            import dashscope
            from http import HTTPStatus

            api_key = os.getenv("DASHSCOPE_API_KEY", "")
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]

            response = dashscope.Generation.call(
                model=self.model_name if "qwen" in self.model_name
                else "qwen-plus",
                messages=messages,
                result_format="message",
                api_key=api_key,
            )

            if response.status_code == HTTPStatus.OK:
                return response.output.choices[0].message.content
            else:
                print(f"[EchoMind] DashScope 调用失败: {response.message}")
                return None
        except Exception as e:
            print(f"[EchoMind] LLM 调用异常: {e}")
            return None

    @staticmethod
    def _parse_json_response(response: str) -> Optional[dict]:
        """解析 LLM 返回的 JSON"""
        if not response:
            return None

        # 尝试直接解析
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            pass

        # 尝试提取 ```json ... ``` 代码块
        json_match = re.search(
            r'```(?:json)?\s*\n?(.*?)\n?```',
            response,
            re.DOTALL,
        )
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass

        # 尝试提取 { ... } 对象
        brace_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
        if brace_match:
            try:
                return json.loads(brace_match.group())
            except json.JSONDecodeError:
                pass

        print(f"[EchoMind] 无法解析 JSON 响应: {response[:200]}...")
        return None

    def get_stats(self) -> dict:
        """获取巩固统计"""
        return {
            "total_consolidations": self.total_consolidations,
            "total_merges": self.total_merges,
        }