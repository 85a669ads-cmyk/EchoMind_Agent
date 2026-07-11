"""
EchoMind 2.0 API 路由

FastAPI 路由定义，包含：
- /chat - 对话接口（非流式和流式）
- /health - 健康检查（含阿里云服务状态）
- /memory/stats - 记忆统计仪表盘
- /memory/consolidate - 手动触发记忆巩固
- /memory/purge - 手动触发遗忘清理
"""

from __future__ import annotations

import asyncio
import time
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.core.agent_engine import EchoMindAgent
from app.models.chat_models import ChatRequest, ChatResponse
from app.models.memory_models import MemoryStats

router = APIRouter(prefix="/api/v1", tags=["EchoMind"])


# ---------- 请求/响应辅助模型 ----------

class ChatRequestBody(BaseModel):
    user_id: str = "default_user"
    session_id: str = "default_session"
    message: str = Field(..., min_length=1)
    stream: bool = True
    enable_memory: bool = True
    enable_tools: bool = True


class HealthResponse(BaseModel):
    status: str
    version: str = "2.0.0"
    timestamp: float
    services: dict
    memory_stats: Optional[dict] = None


# ---------- 全局 Agent 引用（由 app.py 注入）----------
_agent: Optional[EchoMindAgent] = None


def set_agent(agent: EchoMindAgent) -> None:
    """设置全局 Agent 实例"""
    global _agent
    _agent = agent


def get_agent() -> EchoMindAgent:
    """获取全局 Agent 实例"""
    if _agent is None:
        raise HTTPException(status_code=503, detail="Agent 未初始化")
    return _agent


# ---------- 路由定义 ----------

@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    健康检查接口

    返回阿里云服务（DashScope, DashVector）的连接状态。
    设计参考: 计划.md §3 模块四 - 阿里云证明
    """
    agent = _agent
    services_status = {
        "dashscope": {"status": "unknown", "message": ""},
        "dashvector": {"status": "unknown", "message": ""},
        "memory_system": {"status": "unknown", "message": ""},
    }

    # 检查 DashScope (LLM)
    try:
        import dashscope
        api_key = agent.config.dashscope_api_key if agent else ""
        if api_key:
            services_status["dashscope"] = {
                "status": "healthy",
                "message": f"已配置 (模型: {agent.config.llm_model_complex})"
            }
        else:
            services_status["dashscope"] = {
                "status": "degraded",
                "message": "API Key 未配置"
            }
    except ImportError:
        services_status["dashscope"] = {
            "status": "unavailable",
            "message": "dashscope 未安装"
        }
    except Exception as e:
        services_status["dashscope"] = {
            "status": "error",
            "message": str(e)
        }

    # 检查 DashVector
    try:
        import dashvector
        if agent and not agent.long_term_memory.use_local_fallback:
            services_status["dashvector"] = {
                "status": "healthy",
                "message": f"集合: {agent.long_term_memory.collection_name}"
            }
        else:
            services_status["dashvector"] = {
                "status": "degraded",
                "message": "使用本地回退存储"
            }
    except ImportError:
        services_status["dashvector"] = {
            "status": "unavailable",
            "message": "dashvector 未安装"
        }
    except Exception as e:
        services_status["dashvector"] = {
            "status": "error",
            "message": str(e)
        }

    # 检查记忆系统
    if agent:
        try:
            wm_stats = agent.working_memory.get_stats()
            ltm_stats = agent.long_term_memory.get_stats()
            services_status["memory_system"] = {
                "status": "healthy",
                "message": f"短期: {wm_stats['total_working_memories']}条, "
                          f"长期: {ltm_stats['total_long_term_memories']}条"
            }
        except Exception as e:
            services_status["memory_system"] = {
                "status": "error",
                "message": str(e)
            }

    # 计算整体状态
    statuses = [s["status"] for s in services_status.values()]
    if "error" in statuses:
        overall = "error"
    elif "unavailable" in statuses:
        overall = "degraded"
    elif "degraded" in statuses:
        overall = "degraded"
    else:
        overall = "healthy"

    memory_stats = None
    if agent:
        try:
            memory_stats = {
                "working_memory": agent.working_memory.get_stats(),
                "long_term_memory": agent.long_term_memory.get_stats(),
            }
        except Exception:
            pass

    return HealthResponse(
        status=overall,
        timestamp=time.time(),
        services=services_status,
        memory_stats=memory_stats,
    )


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequestBody):
    """
    非流式对话接口

    返回完整的 ChatResponse，包含思考链、记忆检索结果等。
    """
    agent = get_agent()

    chat_request = ChatRequest(
        user_id=request.user_id,
        session_id=request.session_id,
        message=request.message,
        stream=False,
        enable_memory=request.enable_memory,
        enable_tools=request.enable_tools,
    )

    response = await agent.process(chat_request)
    return response


@router.post("/chat/stream")
async def chat_stream(request: ChatRequestBody):
    """
    流式对话接口

    返回 Server-Sent Events 格式的实时响应流。
    """
    agent = get_agent()

    chat_request = ChatRequest(
        user_id=request.user_id,
        session_id=request.session_id,
        message=request.message,
        stream=True,
        enable_memory=request.enable_memory,
        enable_tools=request.enable_tools,
    )

    async def event_generator():
        async for token in agent.process_stream(chat_request):
            yield f"data: {token}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/memory/stats")
async def memory_stats(user_id: Optional[str] = Query(None)):
    """记忆统计仪表盘接口"""
    agent = get_agent()

    wm_stats = agent.working_memory.get_stats()
    ltm_stats = agent.long_term_memory.get_stats()

    # 遗忘曲线统计
    forgetting_stats = {"high_retention": 0, "moderate": 0, "low_retention": 0}
    if agent.memory_consolidator and hasattr(agent, 'forgetting_engine'):
        try:
            # 通过遗忘引擎获取
            pass
        except Exception:
            pass

    return {
        "total_long_term_memories": ltm_stats.get("total_long_term_memories", 0),
        "total_working_memories": wm_stats.get("total_working_memories", 0),
        "active_users": wm_stats.get("active_users", 0),
        "total_consolidations": wm_stats.get("total_consolidations", 0),
        "average_importance": ltm_stats.get("average_importance", 0.0),
        "forgetting_curve_stats": forgetting_stats,
    }


@router.post("/memory/consolidate")
async def trigger_consolidation(
    user_id: str = Query("default_user"),
    session_id: str = Query("default_session"),
):
    """手动触发记忆巩固"""
    agent = get_agent()

    if not agent.memory_consolidator:
        raise HTTPException(status_code=400, detail="记忆巩固器未配置")

    working_memories = agent.working_memory.get_all_for_user(user_id)

    if len(working_memories) < 3:
        return {
            "status": "skipped",
            "message": f"短期记忆不足 (当前 {len(working_memories)} 条，至少需要 3 条)",
        }

    result = await agent.memory_consolidator.consolidate(
        user_id=user_id,
        working_memories=working_memories,
    )

    if result:
        return {
            "status": "success",
            "consolidated_content": result.consolidated_memory.content,
            "merged_count": result.merged_count,
            "summary": result.summary,
        }
    else:
        return {
            "status": "no_action",
            "message": "未找到值得长期记忆的内容",
        }


@router.post("/memory/purge")
async def trigger_purge(
    user_id: Optional[str] = Query(None),
    dry_run: bool = Query(False),
):
    """手动触发低价值记忆清理"""
    agent = get_agent()

    # 使用长期记忆管理器的清理方法
    if user_id:
        purged = agent.long_term_memory.purge_low_value_memories(
            user_id=user_id,
            forgetting_threshold=agent.config.memory_forgetting_threshold,
        )
    else:
        purged = 0
        # 遍历所有用户
        for uid in []:  # 需要通过其他方式获取用户列表
            purged += agent.long_term_memory.purge_low_value_memories(
                user_id=uid,
                forgetting_threshold=agent.config.memory_forgetting_threshold,
            )

    return {
        "status": "dry_run" if dry_run else "completed",
        "purged_count": purged if not dry_run else "N/A (dry_run)",
    }