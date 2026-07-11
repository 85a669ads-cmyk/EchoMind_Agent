"""
遗忘引擎 (Forgetting Engine)

基于艾宾浩斯遗忘曲线 + 访问频率，定期清理低价值记忆。
可配置定时任务，自动评估每条长期记忆的保留价值。

设计参考: 计划.md §3 模块一 - 优化遗忘曲线

公式: Score = Importance * (1 / (1 + k * time_diff_hours)) * log(1 + access_count)
"""

from __future__ import annotations

import asyncio
import threading
import time
from typing import Optional

from app.models.memory_models import LongTermMemoryItem


class ForgettingEngine:
    """
    遗忘引擎

    功能:
    1. 基于艾宾浩斯遗忘曲线评估每条记忆的保留分数
    2. 自动清理低于阈值的低价值记忆
    3. 支持定时任务和手动触发
    4. 记录遗忘统计
    """

    def __init__(
        self,
        long_term_memory_manager=None,
        forgetting_threshold: float = 0.1,
        decay_rate_k: float = 0.05,
        purge_interval_seconds: int = 3600,  # 每小时检查一次
        auto_purge: bool = True,
    ):
        """
        Args:
            long_term_memory_manager: LongTermMemoryManager 实例
            forgetting_threshold: 遗忘分数阈值（低于此值的记忆被清除）
            decay_rate_k: 遗忘速率常数（默认 0.05，模拟艾宾浩斯曲线）
            purge_interval_seconds: 定时清理间隔（秒）
            auto_purge: 是否启用自动定时清理
        """
        self.long_term_memory = long_term_memory_manager
        self.forgetting_threshold = forgetting_threshold
        self.decay_rate_k = decay_rate_k
        self.purge_interval_seconds = purge_interval_seconds
        self.auto_purge = auto_purge

        # 统计
        self.total_purged = 0
        self.total_evaluated = 0
        self.last_purge_time: Optional[float] = None
        self.purge_history: list[dict] = []

        # 后台清理任务
        self._stop_event = threading.Event()
        self._purge_thread: Optional[threading.Thread] = None

        if auto_purge:
            self.start_background_purge()

    def evaluate_memory(self, memory: LongTermMemoryItem) -> float:
        """
        评估单条记忆的保留价值

        Args:
            memory: 长期记忆项

        Returns:
            float: 0~1 之间的保留分数
        """
        # 使用数据模型中定义的遗忘曲线公式
        score = memory.forgetting_score(k=self.decay_rate_k)

        # 额外因素：访问频率加成
        if memory.access_count > 10:
            score *= 1.2  # 高频访问记忆更不易遗忘
        if memory.importance > 0.8:
            score *= 1.1  # 高重要性记忆有额外保护

        return min(1.0, max(0.0, score))

    def should_forget(self, memory: LongTermMemoryItem) -> tuple[bool, float]:
        """
        判断记忆是否应该被遗忘

        Args:
            memory: 长期记忆项

        Returns:
            tuple[bool, float]: (是否遗忘, 保留分数)
        """
        score = self.evaluate_memory(memory)
        return (score < self.forgetting_threshold, score)

    def purge(
        self, user_id: Optional[str] = None, dry_run: bool = False
    ) -> dict:
        """
        执行记忆清理

        Args:
            user_id: 限定用户（None 表示全部用户）
            dry_run: 仅评估不实际删除

        Returns:
            dict: 清理统计 { total_evaluated, total_purged, purged_memories }
        """
        if not self.long_term_memory:
            return {"error": "无长期记忆管理器", "total_evaluated": 0, "total_purged": 0}

        purged_items: list[dict] = []
        total_evaluated = 0

        # 获取待评估的记忆
        if user_id:
            memories = self.long_term_memory.get_user_memories(user_id, limit=1000)
        else:
            # 获取所有用户（通过本地回退扫描）
            memories = self._get_all_memories()

        total_evaluated = len(memories)
        to_delete: list[str] = []

        for memory in memories:
            should_del, score = self.should_forget(memory)
            if should_del:
                to_delete.append(memory.memory_id)
                purged_items.append({
                    "memory_id": memory.memory_id,
                    "content": memory.content[:80],
                    "score": round(score, 4),
                    "importance": memory.importance,
                    "access_count": memory.access_count,
                    "age_hours": round(
                        (time.time() - memory.created_at) / 3600, 1
                    ),
                })

        # 执行删除
        if not dry_run:
            for mem_id in to_delete:
                self.long_term_memory.delete_memory(mem_id)

        total_purged = len(to_delete)
        self.total_evaluated += total_evaluated
        self.total_purged += total_purged
        self.last_purge_time = time.time()

        result = {
            "total_evaluated": total_evaluated,
            "total_purged": total_purged,
            "purged_memories": purged_items,
            "timestamp": time.time(),
        }
        self.purge_history.append(result)

        # 只保留最近 100 条历史
        if len(self.purge_history) > 100:
            self.purge_history = self.purge_history[-100:]

        return result

    def _get_all_memories(self) -> list[LongTermMemoryItem]:
        """获取所有用户的长期记忆（本地回退实现）"""
        if self.long_term_memory and hasattr(
            self.long_term_memory, "_local_store"
        ):
            store = self.long_term_memory._local_store
            memories = []
            for mem_id, (vec, meta) in store.items():
                memories.append(LongTermMemoryItem.from_metadata(
                    memory_id=mem_id,
                    metadata=meta,
                    vector=vec,
                ))
            return memories
        return []

    async def async_purge(
        self, user_id: Optional[str] = None, dry_run: bool = False
    ) -> dict:
        """异步版本的清理（用于 FastAPI 后台任务）"""
        return await asyncio.to_thread(self.purge, user_id, dry_run)

    def start_background_purge(self) -> None:
        """启动后台定时清理任务"""
        if self._purge_thread and self._purge_thread.is_alive():
            return

        self._stop_event.clear()
        self._purge_thread = threading.Thread(
            target=self._background_purge_loop, daemon=True
        )
        self._purge_thread.start()
        print(
            f"[EchoMind] 遗忘引擎已启动，清理间隔: {self.purge_interval_seconds}s"
        )

    def stop_background_purge(self) -> None:
        """停止后台定时清理任务"""
        self._stop_event.set()
        if self._purge_thread:
            self._purge_thread.join(timeout=5)

    def _background_purge_loop(self) -> None:
        """后台清理循环"""
        while not self._stop_event.is_set():
            self._stop_event.wait(self.purge_interval_seconds)
            if not self._stop_event.is_set():
                try:
                    result = self.purge(dry_run=False)
                    if result.get("total_purged", 0) > 0:
                        print(
                            f"[EchoMind] 遗忘引擎清理了 {result['total_purged']} 条低价值记忆"
                        )
                except Exception as e:
                    print(f"[EchoMind] 遗忘引擎清理异常: {e}")

    def get_stats(self) -> dict:
        """获取遗忘引擎统计"""
        return {
            "total_evaluated": self.total_evaluated,
            "total_purged": self.total_purged,
            "forgetting_threshold": self.forgetting_threshold,
            "decay_rate_k": self.decay_rate_k,
            "purge_interval_seconds": self.purge_interval_seconds,
            "auto_purge": self.auto_purge,
            "last_purge_time": self.last_purge_time,
            "recent_purges": self.purge_history[-5:],
        }

    def get_forgetting_curve_stats(self, user_id: Optional[str] = None) -> dict[str, int]:
        """
        获取遗忘曲线分布统计

        Returns:
            dict: {
                'high_retention': 高保留 (>0.7),
                'moderate': 中等 (0.3-0.7),
                'low_retention': 低保留 (<0.3)
            }
        """
        stats = {"high_retention": 0, "moderate": 0, "low_retention": 0}

        if self.long_term_memory:
            memories = (
                self.long_term_memory.get_user_memories(user_id, limit=1000)
                if user_id else self._get_all_memories()
            )

            for memory in memories:
                score = self.evaluate_memory(memory)
                if score > 0.7:
                    stats["high_retention"] += 1
                elif score >= 0.3:
                    stats["moderate"] += 1
                else:
                    stats["low_retention"] += 1

        return stats