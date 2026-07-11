"""
短期工作记忆管理器 (Working Memory)

使用内存字典存储最近 N 轮对话，支持 TTL 过期自动清理。
达到阈值后触发记忆巩固回调。

设计参考: 计划.md §3 模块一 - 记忆分层
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from typing import Callable, Optional

from app.models.memory_models import WorkingMemoryItem, MemoryPolarity


class WorkingMemory:
    """
    短期工作记忆管理器

    特性:
    - 每个用户维护一个对话队列，最多保留 max_items 条
    - 每条记忆有 TTL（默认30分钟），过期自动清理
    - 达到 consolidate_threshold 时触发巩固回调
    - 线程安全（使用 threading.Lock）
    """

    def __init__(
        self,
        max_items_per_user: int = 50,
        default_ttl_seconds: int = 1800,  # 30 分钟
        consolidate_threshold: int = 10,
        on_consolidate_callback: Optional[Callable[[str, list[WorkingMemoryItem]], None]] = None,
    ):
        """
        Args:
            max_items_per_user: 每个用户最多保留的短期记忆条数
            default_ttl_seconds: 默认生存时间（秒）
            consolidate_threshold: 触发巩固的阈值（条数）
            on_consolidate_callback: 巩固回调函数，签名为 (user_id, list[WorkingMemoryItem]) -> None
        """
        self.max_items_per_user = max_items_per_user
        self.default_ttl_seconds = default_ttl_seconds
        self.consolidate_threshold = consolidate_threshold
        self.on_consolidate_callback = on_consolidate_callback

        # 核心存储: { user_id: OrderedDict[session_id, list[WorkingMemoryItem]] }
        self._storage: dict[str, dict[str, list[WorkingMemoryItem]]] = {}
        self._lock = threading.Lock()

        # 已巩固计数（统计用）
        self.consolidation_count: dict[str, int] = {}

    def add(
        self,
        user_id: str,
        content: str,
        session_id: str = "default",
        importance: float = 0.5,
        polarity: MemoryPolarity = MemoryPolarity.NEUTRAL,
    ) -> WorkingMemoryItem:
        """
        添加一条短期记忆

        Args:
            user_id: 用户ID
            content: 记忆内容
            session_id: 会话ID
            importance: 重要性评分
            polarity: 情感极性

        Returns:
            WorkingMemoryItem: 创建的短期记忆项
        """
        item = WorkingMemoryItem(
            user_id=user_id,
            session_id=session_id,
            content=content,
            importance=importance,
            polarity=polarity,
            turn_index=0,
            ttl_seconds=self.default_ttl_seconds,
        )

        with self._lock:
            # 初始化用户存储
            if user_id not in self._storage:
                self._storage[user_id] = {}
            if session_id not in self._storage[user_id]:
                self._storage[user_id][session_id] = []

            session_memories = self._storage[user_id][session_id]
            item.turn_index = len(session_memories)
            session_memories.append(item)

            # 超过最大条数时移除最旧的
            if len(session_memories) > self.max_items_per_user:
                session_memories.pop(0)

            # 检查是否需要触发巩固
            if len(session_memories) >= self.consolidate_threshold:
                if self.on_consolidate_callback:
                    # 复制一份避免在回调中修改
                    items_to_consolidate = list(session_memories[-self.consolidate_threshold:])
                    self.on_consolidate_callback(user_id, items_to_consolidate)
                    self.consolidation_count[user_id] = (
                        self.consolidation_count.get(user_id, 0) + 1
                    )

        return item

    def get_recent(self, user_id: str, session_id: str = "default", n: int = 10) -> list[WorkingMemoryItem]:
        """
        获取最近的 N 条短期记忆（自动过滤过期项）

        Args:
            user_id: 用户ID
            session_id: 会话ID
            n: 返回条数

        Returns:
            list[WorkingMemoryItem]: 最近的短期记忆列表
        """
        with self._lock:
            if user_id not in self._storage:
                return []
            if session_id not in self._storage[user_id]:
                return []

            memories = self._storage[user_id][session_id]

            # 过滤过期记忆
            valid_memories = [m for m in memories if not m.is_expired()]

            # 移除过期项
            self._storage[user_id][session_id] = valid_memories

            return valid_memories[-n:]

    def get_all_for_user(self, user_id: str) -> list[WorkingMemoryItem]:
        """获取某用户所有会话的短期记忆（合并）"""
        with self._lock:
            if user_id not in self._storage:
                return []

            all_items: list[WorkingMemoryItem] = []
            for session_id, memories in self._storage[user_id].items():
                valid = [m for m in memories if not m.is_expired()]
                self._storage[user_id][session_id] = valid
                all_items.extend(valid)
            return all_items

    def clear_session(self, user_id: str, session_id: str = "default") -> None:
        """清除指定会话的短期记忆"""
        with self._lock:
            if user_id in self._storage and session_id in self._storage[user_id]:
                del self._storage[user_id][session_id]

    def clear_user(self, user_id: str) -> None:
        """清除某用户的所有短期记忆"""
        with self._lock:
            if user_id in self._storage:
                del self._storage[user_id]

    def count(self, user_id: str, session_id: str = "default") -> int:
        """获取某用户某会话的短期记忆条数"""
        with self._lock:
            if user_id not in self._storage:
                return 0
            if session_id not in self._storage[user_id]:
                return 0
            valid = [m for m in self._storage[user_id][session_id] if not m.is_expired()]
            self._storage[user_id][session_id] = valid
            return len(valid)

    def total_count(self) -> int:
        """获取所有用户的短期记忆总条数"""
        with self._lock:
            total = 0
            for user_id in self._storage:
                for session_id in self._storage[user_id]:
                    total += len(self._storage[user_id][session_id])
            return total

    def get_consolidation_count(self, user_id: str) -> int:
        """获取某用户的记忆巩固次数"""
        return self.consolidation_count.get(user_id, 0)

    def get_stats(self) -> dict:
        """获取统计信息"""
        with self._lock:
            active_users = len(self._storage)
            total_items = sum(
                len(items)
                for user_sessions in self._storage.values()
                for items in user_sessions.values()
            )
            return {
                "active_users": active_users,
                "total_working_memories": total_items,
                "total_consolidations": sum(self.consolidation_count.values()),
            }