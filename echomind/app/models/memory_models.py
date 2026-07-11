"""
EchoMind 2.0 记忆数据模型

定义短期记忆、长期记忆、记忆检索、巩固、冲突解决的数据结构。
支持记忆的情感极性和艾宾浩斯遗忘曲线权重计算。
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator


class MemoryPolarity(str, Enum):
    """记忆情感极性"""
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


class MemoryItem(BaseModel):
    """记忆基类"""
    content: str = Field(..., description="记忆内容文本")
    importance: float = Field(default=0.5, ge=0.0, le=1.0, description="重要性评分 0~1")
    polarity: MemoryPolarity = Field(default=MemoryPolarity.NEUTRAL, description="情感极性")
    access_count: int = Field(default=0, ge=0, description="访问次数")
    created_at: float = Field(
        default_factory=lambda: datetime.now(timezone.utc).timestamp(),
        description="创建时间戳 (UTC)"
    )
    last_accessed_at: Optional[float] = Field(default=None, description="最后访问时间戳")

    def record_access(self) -> None:
        """记录一次访问"""
        self.access_count += 1
        self.last_accessed_at = time.time()

    def forgetting_score(self, k: float = 0.05) -> float:
        """
        基于艾宾浩斯遗忘曲线 + 访问频率计算记忆保留分数

        公式: Score = Importance * (1 / (1 + k * time_diff_hours)) * log(1 + access_count)

        Args:
            k: 遗忘速率常数（默认 0.05，模拟艾宾浩斯曲线）

        Returns:
            float: 0~1 之间的保留分数，越高越不易遗忘
        """
        now = time.time()
        hours_passed = (now - self.created_at) / 3600.0
        import math
        time_decay = 1.0 / (1.0 + k * hours_passed)
        access_bonus = math.log(2.0 + self.access_count)
        score = self.importance * time_decay * min(access_bonus, 2.0)
        return max(0.0, min(1.0, score))


class WorkingMemoryItem(MemoryItem):
    """
    短期工作记忆项

    存储在 Redis 或内存字典中，具有 TTL（默认30分钟），
    达到阈值后触发记忆巩固。
    """
    user_id: str = Field(..., description="用户ID")
    session_id: str = Field(..., description="会话ID")
    turn_index: int = Field(default=0, description="当前会话中的轮次索引")
    ttl_seconds: int = Field(default=1800, description="生存时间(秒)，默认30分钟")

    def is_expired(self) -> bool:
        """检查记忆是否已过期"""
        now = time.time()
        return (now - self.created_at) > self.ttl_seconds


class LongTermMemoryItem(MemoryItem):
    """
    长期记忆项

    存储在 DashVector（语义检索）+ SQLite/PostgreSQL（结构化元数据）中。
    经过巩固器总结后生成。
    """
    user_id: str = Field(..., description="用户ID")
    memory_id: str = Field(default="", description="记忆唯一标识（向量库返回的ID）")
    source_episode_ids: list[str] = Field(
        default_factory=list,
        description="源情景记忆ID列表（巩固来源）"
    )
    tags: list[str] = Field(default_factory=list, description="标签分类")
    category: str = Field(default="general", description="记忆类别：偏好/事实/事件/技能")
    vector: Optional[list[float]] = Field(default=None, description="文本向量（由嵌入模型生成）")

    def to_metadata(self) -> dict[str, Any]:
        """转为 DashVector 可存储的元数据字典"""
        return {
            "user_id": self.user_id,
            "content": self.content,
            "importance": self.importance,
            "polarity": self.polarity.value,
            "access_count": self.access_count,
            "created_at": self.created_at,
            "last_accessed_at": self.last_accessed_at or self.created_at,
            "tags": ",".join(self.tags),
            "category": self.category,
            "source_episode_ids": ",".join(self.source_episode_ids),
        }

    @classmethod
    def from_metadata(
        cls, memory_id: str, metadata: dict[str, Any], vector: Optional[list[float]] = None
    ) -> "LongTermMemoryItem":
        """从 DashVector 元数据还原记忆对象"""
        return cls(
            memory_id=memory_id,
            user_id=metadata.get("user_id", ""),
            content=metadata.get("content", ""),
            importance=float(metadata.get("importance", 0.5)),
            polarity=MemoryPolarity(metadata.get("polarity", "neutral")),
            access_count=int(metadata.get("access_count", 0)),
            created_at=float(metadata.get("created_at", time.time())),
            last_accessed_at=float(metadata.get("last_accessed_at", time.time())),
            tags=metadata.get("tags", "").split(",") if metadata.get("tags") else [],
            category=metadata.get("category", "general"),
            source_episode_ids=(
                metadata.get("source_episode_ids", "").split(",")
                if metadata.get("source_episode_ids")
                else []
            ),
            vector=vector,
        )


class MemorySearchResult(BaseModel):
    """记忆检索结果"""
    memory: LongTermMemoryItem = Field(..., description="检索到的记忆项")
    similarity_score: float = Field(..., ge=0.0, le=1.0, description="相似度评分")
    rank: int = Field(default=1, description="检索排名")


class MemoryConsolidationResult(BaseModel):
    """记忆巩固结果"""
    source_items: list[WorkingMemoryItem] = Field(..., description="被巩固的短期记忆项")
    consolidated_memory: LongTermMemoryItem = Field(..., description="生成的长期记忆项")
    summary: str = Field(default="", description="巩固总结文本")
    merged_count: int = Field(default=0, description="合并的相似记忆数")


class MemoryConflictResult(BaseModel):
    """记忆冲突解决结果"""
    existing_memory: LongTermMemoryItem = Field(..., description="已有记忆")
    new_memory: LongTermMemoryItem = Field(..., description="新记忆")
    resolution: str = Field(..., description="冲突解决结果: 'replace' | 'merge' | 'keep_existing'")
    resolved_memory: LongTermMemoryItem = Field(..., description="解决后的最终记忆")
    reason: str = Field(default="", description="解决理由")


class MemoryStats(BaseModel):
    """记忆统计信息（用于仪表盘展示）"""
    total_long_term_memories: int = Field(default=0, description="长期记忆总数")
    total_working_memories: int = Field(default=0, description="当前短期记忆数")
    recent_consolidations: int = Field(default=0, description="近期巩固次数")
    recent_conflicts_resolved: int = Field(default=0, description="近期冲突解决次数")
    active_users: int = Field(default=0, description="活跃用户数")
    average_importance: float = Field(default=0.0, description="平均记忆重要性")
    forgetting_curve_stats: dict[str, float] = Field(
        default_factory=dict,
        description="遗忘曲线统计 { 'high_retention': count, 'moderate': count, 'low_retention': count }"
    )