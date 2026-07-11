"""
EchoMind 2.0 用户数据模型

存储用户画像和偏好信息，供记忆系统个性化检索使用。
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, Field


class UserPreferences(BaseModel):
    """用户偏好设置"""
    language: str = Field(default="zh-CN", description="对话语言")
    response_style: str = Field(default="detailed", description="回复风格: 'concise' | 'detailed' | 'creative'")
    interests: list[str] = Field(default_factory=list, description="用户兴趣标签")
    dietary_restrictions: list[str] = Field(default_factory=list, description="饮食限制")
    timezone: str = Field(default="Asia/Shanghai", description="用户时区")
    custom_instructions: str = Field(default="", description="用户自定义指令")


class UserProfile(BaseModel):
    """用户画像"""
    user_id: str = Field(..., description="用户唯一ID")
    username: str = Field(default="anonymous", description="用户名")
    preferences: UserPreferences = Field(default_factory=UserPreferences, description="偏好设置")
    total_conversations: int = Field(default=0, ge=0, description="总对话轮次")
    total_memories: int = Field(default=0, ge=0, description="长期记忆总数")
    created_at: float = Field(
        default_factory=lambda: datetime.now(timezone.utc).timestamp(),
        description="用户创建时间"
    )
    last_active_at: float = Field(
        default_factory=lambda: datetime.now(timezone.utc).timestamp(),
        description="最后活跃时间"
    )

    def record_activity(self) -> None:
        """记录用户活跃"""
        self.total_conversations += 1
        self.last_active_at = datetime.now(timezone.utc).timestamp()