"""
EchoMind 2.0 应用入口

组装所有模块，初始化 Agent，启动 FastAPI 服务。
支持 Chainlit 前端集成（可选）。

设计参考: 计划.md §3 模块三/四
"""

from __future__ import annotations

import asyncio
import os
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router, set_agent
from app.core.config import EchoMindConfig
from app.core.llm_client import LLMClient
from app.core.agent_engine import EchoMindAgent
from app.memory.working_memory import WorkingMemory
from app.memory.long_term_memory import LongTermMemoryManager
from app.memory.consolidator import MemoryConsolidator
from app.memory.conflict_resolver import ConflictResolver
from app.memory.forgetting_engine import ForgettingEngine
from app.tools.base_tools import ToolRegistry
from app.tools.search_tool import SearchWebTool
from app.tools.time_tool import GetCurrentTimeTool


# ---------- 全局变量 ----------
config: EchoMindConfig = None
agent: EchoMindAgent = None
tool_registry: ToolRegistry = None
working_memory: WorkingMemory = None
long_term_memory: LongTermMemoryManager = None
memory_consolidator: MemoryConsolidator = None
conflict_resolver: ConflictResolver = None
forgetting_engine: ForgettingEngine = None


def init_components() -> None:
    """初始化所有系统组件"""
    global config, agent, tool_registry
    global working_memory, long_term_memory
    global memory_consolidator, conflict_resolver, forgetting_engine

    print("=" * 60)
    print("  EchoMind 2.0 - Bionic Memory Agent")
    print("=" * 60)

    # 1. 加载配置
    config = EchoMindConfig.from_dotenv()
    print(f"[配置] 主模型: {config.llm_model_complex}")
    print(f"[配置] 辅助模型: {config.llm_model_simple}")
    print(f"[配置] 记忆回退模式: {'本地' if config.use_local_memory_fallback else 'DashVector'}")

    # 2. 初始化 LLM 客户端
    llm_client = LLMClient(config)
    print("[LLM] DashScope 客户端已就绪")

    # 3. 初始化长期记忆管理器
    long_term_memory = LongTermMemoryManager(
        dashvector_api_key=config.dashvector_api_key,
        dashvector_cluster_endpoint=config.dashvector_endpoint,
        collection_name=config.dashvector_collection,
        use_local_fallback=config.use_local_memory_fallback,
        embedding_func=None,  # 使用内置 DashScope 嵌入
    )
    print(f"[记忆] 长期记忆管理器已就绪 (集合: {config.dashvector_collection})")

    # 4. 初始化记忆巩固器
    # 注入 LLM 简化调用函数
    async def llm_simple_call(system_prompt: str, user_prompt: str) -> str:
        return await llm_client.simple_chat(system_prompt, user_prompt)

    memory_consolidator = MemoryConsolidator(
        llm_call_func=llm_simple_call,
        long_term_memory_manager=long_term_memory,
        model_name=config.llm_model_simple,
    )
    print("[记忆] 巩固器已就绪")

    # 5. 初始化冲突解决器
    conflict_resolver = ConflictResolver(
        llm_call_func=llm_simple_call,
        long_term_memory_manager=long_term_memory,
        similarity_threshold=config.memory_similarity_threshold,
    )
    print("[记忆] 冲突解决器已就绪")

    # 6. 初始化遗忘引擎
    forgetting_engine = ForgettingEngine(
        long_term_memory_manager=long_term_memory,
        forgetting_threshold=config.memory_forgetting_threshold,
        decay_rate_k=config.memory_decay_rate_k,
        auto_purge=not config.debug,  # 调试模式不自动清理
    )
    print(f"[记忆] 遗忘引擎已就绪 (阈值: {config.memory_forgetting_threshold})")

    # 7. 初始化短期工作记忆（注入巩固回调）
    def on_consolidate(user_id: str, items: list):
        """当短期记忆达到阈值时，异步触发巩固"""
        asyncio.create_task(
            memory_consolidator.consolidate(user_id, items)
        )

    working_memory = WorkingMemory(
        max_items_per_user=config.working_memory_max_items,
        default_ttl_seconds=config.working_memory_ttl_seconds,
        consolidate_threshold=config.consolidate_threshold,
        on_consolidate_callback=on_consolidate,
    )
    print(f"[记忆] 短期工作记忆已就绪 (TTL: {config.working_memory_ttl_seconds}s, "
          f"巩固阈值: {config.consolidate_threshold}条)")

    # 8. 初始化工具注册中心
    tool_registry = ToolRegistry()
    tool_registry.register(SearchWebTool())
    tool_registry.register(GetCurrentTimeTool())
    print(f"[工具] 已注册 {len(tool_registry.list_tools())} 个工具: {tool_registry.list_tools()}")

    # 9. 初始化 Agent 引擎
    agent = EchoMindAgent(
        config=config,
        llm_client=llm_client,
        working_memory=working_memory,
        long_term_memory=long_term_memory,
        tool_registry=tool_registry,
        memory_consolidator=memory_consolidator,
        conflict_resolver=conflict_resolver,
    )
    print("[Agent] ReAct 引擎已就绪")

    # 10. 注入到 API 路由
    set_agent(agent)

    print("-" * 60)
    print("  EchoMind 2.0 初始化完成！")
    print(f"  API 文档: http://{config.host}:{config.port}/docs")
    print(f"  健康检查: http://{config.host}:{config.port}/api/v1/health")
    print("=" * 60)


def create_app() -> FastAPI:
    """创建 FastAPI 应用"""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # 启动时初始化
        init_components()
        yield
        # 关闭时清理
        if forgetting_engine:
            forgetting_engine.stop_background_purge()
        print("[EchoMind] 服务已关闭")

    app = FastAPI(
        title="EchoMind 2.0",
        description="Bionic Cognitive Memory Agent with ReAct reasoning",
        version="2.0.0",
        lifespan=lifespan,
    )

    # CORS 配置
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 注册路由
    app.include_router(router)

    # 根路径
    @app.get("/")
    async def root():
        return {
            "name": "EchoMind 2.0",
            "version": "2.0.0",
            "description": "Bionic Cognitive Memory Agent",
            "docs": "/docs",
            "health": "/api/v1/health",
        }

    return app


# ---------- 主入口 ----------

if __name__ == "__main__":
    import uvicorn

    app = create_app()
    uvicorn.run(
        app,
        host=config.host if config else "0.0.0.0",
        port=config.port if config else 8000,
        log_level="info",
    )