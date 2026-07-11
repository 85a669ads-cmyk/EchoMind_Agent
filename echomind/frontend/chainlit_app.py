"""
EchoMind 2.0 Chainlit 前端

专为 LLM 设计的交互界面，原生支持：
- 思考过程展示（ReAct 可视化）
- 记忆检索卡片
- 流式输出打字机效果
- 记忆仪表盘侧边栏

设计参考: 计划.md §3 模块三 - 使用 Chainlit 替代基础 Gradio

运行方式:
    chainlit run frontend/chainlit_app.py
"""

from __future__ import annotations

import asyncio
import json
import os

import chainlit as cl

# 导入后端模块（需要设置正确的路径）
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.core.config import EchoMindConfig
from app.core.llm_client import LLMClient
from app.core.agent_engine import EchoMindAgent
from app.memory.working_memory import WorkingMemory
from app.memory.long_term_memory import LongTermMemoryManager
from app.memory.consolidator import MemoryConsolidator
from app.tools.base_tools import ToolRegistry
from app.tools.search_tool import SearchWebTool
from app.tools.time_tool import GetCurrentTimeTool
from app.models.chat_models import ChatRequest


# ---------- 全局初始化 ----------
config: EchoMindConfig = None
agent: EchoMindAgent = None
llm_client: LLMClient = None
tool_registry: ToolRegistry = None


def init_frontend():
    """初始化前端所需的后端组件"""
    global config, agent, llm_client, tool_registry

    config = EchoMindConfig.from_dotenv()
    llm_client = LLMClient(config)

    # 长期记忆管理器
    long_term_memory = LongTermMemoryManager(
        dashvector_api_key=config.dashvector_api_key,
        dashvector_cluster_endpoint=config.dashvector_endpoint,
        collection_name=config.dashvector_collection,
        use_local_fallback=config.use_local_memory_fallback,
    )

    # LLM 调用函数
    async def llm_simple_call(system_prompt: str, user_prompt: str) -> str:
        return await llm_client.simple_chat(system_prompt, user_prompt)

    # 记忆巩固器
    memory_consolidator = MemoryConsolidator(
        llm_call_func=llm_simple_call,
        long_term_memory_manager=long_term_memory,
    )

    # 工具注册
    tool_registry = ToolRegistry()
    tool_registry.register(SearchWebTool())
    tool_registry.register(GetCurrentTimeTool())

    # 短期工作记忆
    def on_consolidate(user_id: str, items: list):
        asyncio.create_task(
            memory_consolidator.consolidate(user_id, items)
        )

    working_memory = WorkingMemory(
        max_items_per_user=config.working_memory_max_items,
        default_ttl_seconds=config.working_memory_ttl_seconds,
        consolidate_threshold=config.consolidate_threshold,
        on_consolidate_callback=on_consolidate,
    )

    # Agent 引擎
    agent = EchoMindAgent(
        config=config,
        llm_client=llm_client,
        working_memory=working_memory,
        long_term_memory=long_term_memory,
        tool_registry=tool_registry,
        memory_consolidator=memory_consolidator,
    )


# ---------- Chainlit 生命周期 ----------

@cl.on_chat_start
async def on_chat_start():
    """聊天开始时初始化"""
    # 初始化后端组件
    init_frontend()

    # 发送欢迎消息
    await cl.Message(
        content="# 🧠 EchoMind 2.0\n"
                "你好！我是具备**持久记忆**的智能助手。\n\n"
                "**核心能力**：\n"
                "- 📝 记住你的偏好、习惯和历史对话\n"
                "- 🧠 自动将短期对话总结为长期记忆\n"
                "- 🔧 在需要时调用工具（搜索、时间等）\n"
                "- 💬 展示思考过程和记忆检索结果\n\n"
                "开始对话吧！我会记得你说过的一切 🚀",
        author="EchoMind",
    ).send()

    # 存储用户会话
    cl.user_session.set("user_id", "chainlit_user")
    cl.user_session.set("session_id", "chainlit_session")


@cl.on_message
async def on_message(message: cl.Message):
    """处理用户消息"""
    user_id = cl.user_session.get("user_id", "default_user")
    session_id = cl.user_session.get("session_id", "default_session")

    # 1. 检索长期记忆
    retrieved_msg = cl.Message(
        content="",
        author="System",
        parent_id=message.id,
    )

    if agent:
        try:
            from app.models.memory_models import MemorySearchResult
            memories = agent.long_term_memory.search(
                query=message.content,
                user_id=user_id,
                top_k=5,
                similarity_threshold=0.3,
            )

            if memories:
                memory_content = "🧠 **检索到的长期记忆**：\n"
                for i, mem in enumerate(memories[:5], 1):
                    memory_content += (
                        f"{i}. 💾 *{mem.memory.content}* "
                        f"(相关度: {mem.similarity_score:.2f})\n"
                    )
            else:
                memory_content = "📭 未找到相关长期记忆"

            retrieved_msg.content = memory_content
            await retrieved_msg.send()
        except Exception as e:
            retrieved_msg.content = f"⚠️ 记忆检索异常: {e}"
            await retrieved_msg.send()

    # 2. 构建 ChatRequest 并获取回复
    chat_request = ChatRequest(
        user_id=user_id,
        session_id=session_id,
        message=message.content,
        stream=False,
        enable_memory=True,
        enable_tools=True,
    )

    if agent:
        # 发送"思考中"状态
        thinking_msg = cl.Message(
            content="💭 正在思考...",
            author="System",
            parent_id=retrieved_msg.id,
        )
        await thinking_msg.send()

        # 获取 Agent 回复
        response = await agent.process(chat_request)

        # 更新思考消息为 ReAct 步骤
        if response.thought_chain:
            react_content = "📋 **思考过程 (ReAct)**：\n"
            for step in response.thought_chain:
                react_content += f"\n**Step {step.step_index}**：\n"
                if step.thought:
                    react_content += f"💡 *Thought*: {step.thought[:150]}...\n"
                if step.action:
                    react_content += f"⚡ *Action*: {step.action}\n"
                if step.tool_call:
                    react_content += f"🔧 *工具结果*: {step.tool_call.result[:100]}...\n"
            thinking_msg.content = react_content
            await thinking_msg.update()

        # 发送最终回复
        reply_msg = cl.Message(
            content=response.reply,
            author="EchoMind",
            parent_id=thinking_msg.id,
        )
        await reply_msg.send()
    else:
        await cl.Message(
            content="⚠️ Agent 未初始化，请检查配置。",
            author="System",
        ).send()


# ---------- Chainlit 配置 ----------

# 以下配置也可以在 .chainlit/config.toml 中设置
cl.instrument_openai = False  # 不使用 OpenAI，使用阿里云 DashScope


if __name__ == "__main__":
    from chainlit.cli import run_chainlit
    run_chainlit(__file__)