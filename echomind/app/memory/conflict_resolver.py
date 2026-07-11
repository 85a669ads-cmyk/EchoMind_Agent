"""
记忆冲突解决器 (Conflict Resolver)

在存入新记忆时检测与已有记忆的冲突（相似度 > 0.85），
调用 LLM 判断新旧记忆哪个更准确，并决定保留、替换或合并。

设计参考: 计划.md §3 模块一 - 记忆冲突解决
"""

from __future__ import annotations

import json
import os
import re
from typing import Optional

from app.models.memory_models import (
    LongTermMemoryItem,
    MemoryConflictResult,
)


class ConflictResolver:
    """
    记忆冲突解决器

    功能:
    1. 在 save_memory 时检测高度相似的已有记忆
    2. 调用 LLM 判断冲突，决定替换/合并/保留
    3. 记录冲突解决历史
    """

    # 冲突解决 Prompt 模板
    CONFLICT_RESOLUTION_PROMPT = """你是一个记忆管理助手。检测到两条记忆可能存在冲突，请判断应该如何处理。

**已有记忆**：
- 内容: {existing_content}
- 类别: {existing_category}
- 重要性: {existing_importance}
- 情感极性: {existing_polarity}
- 创建时间: {existing_time}

**新记忆**：
- 内容: {new_content}
- 类别: {new_category}
- 重要性: {new_importance}
- 情感极性: {new_polarity}
- 创建时间: {new_time}

**判断标准**：
1. 如果新旧记忆直接矛盾（如"喜欢苹果" vs "讨厌苹果"），选择最新最明确的表述
2. 如果新记忆是对旧记忆的补充/细化，应该合并
3. 如果新旧记忆描述不同方面，保留旧记忆并添加新记忆
4. 如果新记忆明确纠正了旧记忆的错误，替换旧记忆

**请严格按以下 JSON 格式输出**（不要包含其他文字）：
```json
{{
  "resolution": "replace|merge|keep_existing|keep_both",
  "resolved_content": "解决后的最终记忆内容",
  "reason": "一句话说明理由"
}}
```

注意：
- "replace": 用新记忆替换旧记忆
- "merge": 合并新旧记忆为一条
- "keep_existing": 保留旧记忆，丢弃新记忆
- "keep_both": 两条记忆都保留（不冲突或描述不同方面）
"""

    def __init__(
        self,
        llm_call_func: Optional[callable] = None,
        long_term_memory_manager=None,
        similarity_threshold: float = 0.85,
    ):
        """
        Args:
            llm_call_func: LLM 调用函数
            long_term_memory_manager: LongTermMemoryManager 实例
            similarity_threshold: 触发冲突检测的相似度阈值
        """
        self.llm_call_func = llm_call_func
        self.long_term_memory = long_term_memory_manager
        self.similarity_threshold = similarity_threshold

        # 冲突解决历史
        self.conflict_history: list[MemoryConflictResult] = []

        # 统计
        self.total_conflicts_detected = 0
        self.total_resolved = 0
        self.resolution_stats = {
            "replace": 0,
            "merge": 0,
            "keep_existing": 0,
            "keep_both": 0,
        }

    async def resolve_conflict(
        self,
        user_id: str,
        new_memory: LongTermMemoryItem,
    ) -> MemoryConflictResult:
        """
        检测并解决记忆冲突

        Args:
            user_id: 用户ID
            new_memory: 待存入的新记忆

        Returns:
            MemoryConflictResult: 冲突解决结果（即使无冲突也返回 keep_both）
        """
        # 搜索高度相似的已有记忆
        if not self.long_term_memory:
            # 无长期记忆管理器，直接返回
            early_result = MemoryConflictResult(
                existing_memory=new_memory,
                new_memory=new_memory,
                resolution="keep_both",
                resolved_memory=new_memory,
                reason="无长期记忆管理器，跳过冲突检测",
            )
            return early_result

        similar = self.long_term_memory.search(
            query=new_memory.content,
            user_id=user_id,
            top_k=5,
            similarity_threshold=self.similarity_threshold,
        )

        if not similar:
            # 无冲突
            return MemoryConflictResult(
                existing_memory=new_memory,
                new_memory=new_memory,
                resolution="keep_both",
                resolved_memory=new_memory,
                reason="未检测到相似记忆，无冲突",
            )

        # 对每个高度相似的记忆进行冲突判断
        self.total_conflicts_detected += 1

        final_result: Optional[MemoryConflictResult] = None

        for search_result in similar:
            existing = search_result.memory

            # 调用 LLM 判断冲突
            resolution = await self._judge_conflict(existing, new_memory)
            final_result = resolution
            self.total_resolved += 1
            self.resolution_stats[resolution.resolution] = (
                self.resolution_stats.get(resolution.resolution, 0) + 1
            )
            self.conflict_history.append(resolution)

            # 根据解决方案执行操作
            if resolution.resolution == "replace":
                # 替换旧记忆
                new_memory.memory_id = existing.memory_id
                new_memory.created_at = existing.created_at  # 保留原始创建时间
                new_memory.access_count = existing.access_count + 1
                self.long_term_memory.update_memory(new_memory)
                break  # 一个冲突解决后停止

            elif resolution.resolution == "merge":
                # 合并
                existing.content = resolution.resolved_memory.content
                existing.importance = max(
                    existing.importance, new_memory.importance
                )
                existing.tags = list(
                    set(existing.tags + new_memory.tags)
                )
                existing.record_access()
                self.long_term_memory.update_memory(existing)
                break

            elif resolution.resolution == "keep_existing":
                # 保留旧记忆，丢弃新记忆
                break

            elif resolution.resolution == "keep_both":
                # 两条都保留：新记忆正常存入，旧记忆不变
                continue

        if final_result is None:
            final_result = MemoryConflictResult(
                existing_memory=similar[0].memory,
                new_memory=new_memory,
                resolution="keep_both",
                resolved_memory=new_memory,
                reason="所有相似记忆均判断为不冲突",
            )

        return final_result

    async def _judge_conflict(
        self,
        existing: LongTermMemoryItem,
        new_memory: LongTermMemoryItem,
    ) -> MemoryConflictResult:
        """
        调用 LLM 判断冲突并给出解决建议

        Args:
            existing: 已有记忆
            new_memory: 新记忆

        Returns:
            MemoryConflictResult
        """
        from datetime import datetime, timezone

        prompt = self.CONFLICT_RESOLUTION_PROMPT.format(
            existing_content=existing.content,
            existing_category=existing.category,
            existing_importance=existing.importance,
            existing_polarity=existing.polarity.value,
            existing_time=datetime.fromtimestamp(
                existing.created_at, tz=timezone.utc
            ).strftime("%Y-%m-%d %H:%M:%S"),
            new_content=new_memory.content,
            new_category=new_memory.category,
            new_importance=new_memory.importance,
            new_polarity=new_memory.polarity.value,
            new_time=datetime.fromtimestamp(
                new_memory.created_at, tz=timezone.utc
            ).strftime("%Y-%m-%d %H:%M:%S"),
        )

        llm_response = await self._call_llm(
            system_prompt="你是记忆管理助手，负责判断和解决记忆冲突。",
            user_prompt=prompt,
        )

        if llm_response:
            parsed = self._parse_json_response(llm_response)
            if parsed:
                resolution_type = parsed.get("resolution", "keep_both")
                resolved_content = parsed.get(
                    "resolved_content", new_memory.content
                )
                reason = parsed.get("reason", "LLM 判断结果")

                # 构建解决后的记忆
                if resolution_type == "replace":
                    resolved_memory = new_memory.model_copy()
                    resolved_memory.content = resolved_content
                elif resolution_type == "merge":
                    resolved_memory = existing.model_copy()
                    resolved_memory.content = resolved_content
                else:
                    resolved_memory = (
                        existing if resolution_type == "keep_existing"
                        else new_memory
                    )

                return MemoryConflictResult(
                    existing_memory=existing,
                    new_memory=new_memory,
                    resolution=resolution_type,
                    resolved_memory=resolved_memory,
                    reason=reason,
                )

        # LLM 调用失败时的默认行为：保留两条
        return MemoryConflictResult(
            existing_memory=existing,
            new_memory=new_memory,
            resolution="keep_both",
            resolved_memory=new_memory,
            reason="LLM 调用失败，默认保留两条",
        )

    async def _call_llm(
        self, system_prompt: str, user_prompt: str
    ) -> Optional[str]:
        """调用 LLM"""
        if self.llm_call_func:
            try:
                return await self.llm_call_func(system_prompt, user_prompt)
            except Exception as e:
                print(f"[EchoMind] LLM 调用失败: {e}")
                return None

        try:
            import dashscope
            from http import HTTPStatus

            api_key = os.getenv("DASHSCOPE_API_KEY", "")
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]

            response = dashscope.Generation.call(
                model="qwen-plus",
                messages=messages,
                result_format="message",
                api_key=api_key,
            )

            if response.status_code == HTTPStatus.OK:
                return response.output.choices[0].message.content
        except Exception as e:
            print(f"[EchoMind] LLM 调用异常: {e}")

        return None

    @staticmethod
    def _parse_json_response(response: str) -> Optional[dict]:
        """解析 LLM 返回的 JSON"""
        if not response:
            return None
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            pass
        json_match = re.search(
            r'```(?:json)?\s*\n?(.*?)\n?```', response, re.DOTALL
        )
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass
        brace_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
        if brace_match:
            try:
                return json.loads(brace_match.group())
            except json.JSONDecodeError:
                pass
        return None

    def get_stats(self) -> dict:
        """获取冲突解决统计"""
        return {
            "total_conflicts_detected": self.total_conflicts_detected,
            "total_resolved": self.total_resolved,
            "resolution_breakdown": self.resolution_stats,
            "recent_conflicts": [
                {
                    "existing": c.existing_memory.content[:50],
                    "new": c.new_memory.content[:50],
                    "resolution": c.resolution,
                }
                for c in self.conflict_history[-5:]
            ],
        }