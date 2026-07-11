"""
EchoMind 2.0 配置管理

集中管理所有环境变量和系统配置。
支持从 .env 文件和系统环境变量加载。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class EchoMindConfig:
    """EchoMind 全局配置"""

    # ---- 阿里云 DashScope (LLM) ----
    dashscope_api_key: str = field(
        default_factory=lambda: os.getenv("DASHSCOPE_API_KEY", "")
    )
    llm_model_complex: str = "qwen-max"       # 复杂推理主模型
    llm_model_simple: str = "qwen-plus"        # 记忆提取/总结用

    # ---- DashVector (向量数据库) ----
    dashvector_api_key: str = field(
        default_factory=lambda: os.getenv("DASHVECTOR_API_KEY", "")
    )
    dashvector_endpoint: str = field(
        default_factory=lambda: os.getenv("DASHVECTOR_ENDPOINT", "")
    )
    dashvector_collection: str = "echomind_long_term_memory"

    # ---- 记忆系统 ----
    working_memory_max_items: int = 50
    working_memory_ttl_seconds: int = 1800        # 30分钟
    consolidate_threshold: int = 10                # 10条触发巩固
    memory_similarity_threshold: float = 0.85      # 冲突检测阈值
    memory_forgetting_threshold: float = 0.1       # 遗忘分数阈值
    memory_decay_rate_k: float = 0.05              # 艾宾浩斯曲线衰减率

    # ---- Agent 配置 ----
    max_react_steps: int = 5                       # ReAct 最大步数
    llm_temperature: float = 0.7
    llm_max_tokens: int = 2000
    stream_output: bool = True

    # ---- 服务配置 ----
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = field(
        default_factory=lambda: os.getenv("DEBUG", "false").lower() == "true"
    )
    use_local_memory_fallback: bool = field(
        default_factory=lambda: os.getenv("USE_LOCAL_FALLBACK", "true").lower() == "true"
    )

    # ---- 用户配置 ----
    default_user_id: str = "default_user"
    default_session_id: str = "default_session"

    @classmethod
    def from_env(cls) -> "EchoMindConfig":
        """从环境变量加载配置"""
        return cls()

    @classmethod
    def from_dotenv(cls, dotenv_path: Optional[str] = None) -> "EchoMindConfig":
        """从 .env 文件加载配置"""
        try:
            from dotenv import load_dotenv
            load_dotenv(dotenv_path=dotenv_path)
        except ImportError:
            pass
        return cls()