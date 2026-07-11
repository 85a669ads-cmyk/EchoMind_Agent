"""
LLM 客户端 (LLM Client)

封装 DashScope LLM 调用，支持流式和非流式输出。
提供统一的异步接口用于 ReAct Agent 引擎。

设计参考: 计划.md §3 模块二 - 流式输出
"""

from __future__ import annotations

import os
from typing import AsyncGenerator, Optional

from app.core.config import EchoMindConfig


class LLMClient:
    """
    阿里云 DashScope LLM 客户端

    支持:
    - Qwen-Max (复杂推理)
    - Qwen-Plus (记忆提取/总结)
    - 流式输出（SSE）
    - 非流式输出
    """

    def __init__(self, config: EchoMindConfig):
        self.config = config
        self.api_key = config.dashscope_api_key

        # 默认使用 Qwen-Max 进行复杂推理
        self.default_model = config.llm_model_complex
        self.simple_model = config.llm_model_simple

    async def chat(
        self,
        messages: list[dict],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: bool = False,
    ) -> str:
        """
        非流式对话接口

        Args:
            messages: 消息列表 [{"role": "user", "content": "..."}]
            model: 模型名称（默认 qwen-max）
            temperature: 温度参数
            max_tokens: 最大 token 数
            stream: 是否流式

        Returns:
            str: 模型回复文本
        """
        model = model or self.default_model
        temperature = temperature or self.config.llm_temperature
        max_tokens = max_tokens or self.config.llm_max_tokens

        try:
            import dashscope
            from http import HTTPStatus

            response = dashscope.Generation.call(
                model=model,
                messages=messages,
                result_format="message",
                temperature=temperature,
                max_tokens=max_tokens,
                stream=stream,
                api_key=self.api_key,
            )

            if response.status_code == HTTPStatus.OK:
                return response.output.choices[0].message.content
            else:
                error_msg = f"LLM 调用失败: {response.code} - {response.message}"
                print(f"[EchoMind] {error_msg}")
                return f"[错误] {error_msg}"

        except ImportError:
            return "[错误] dashscope 未安装，请运行: pip install dashscope"
        except Exception as e:
            print(f"[EchoMind] LLM 调用异常: {e}")
            return f"[错误] LLM 服务不可用: {str(e)}"

    async def chat_stream(
        self,
        messages: list[dict],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> AsyncGenerator[str, None]:
        """
        流式对话接口

        Args:
            messages: 消息列表
            model: 模型名称
            temperature: 温度参数
            max_tokens: 最大 token 数

        Yields:
            str: 逐 token 输出文本
        """
        model = model or self.default_model
        temperature = temperature or self.config.llm_temperature
        max_tokens = max_tokens or self.config.llm_max_tokens

        try:
            import dashscope
            from http import HTTPStatus

            responses = dashscope.Generation.call(
                model=model,
                messages=messages,
                result_format="message",
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
                incremental_output=True,
                api_key=self.api_key,
            )

            for response in responses:
                if response.status_code == HTTPStatus.OK:
                    content = response.output.choices[0].message.content
                    if content:
                        yield content
                else:
                    yield f"\n[错误] {response.code}: {response.message}"
                    break

        except ImportError:
            yield "[错误] dashscope 未安装，请运行: pip install dashscope"
        except Exception as e:
            yield f"[错误] LLM 流式输出异常: {str(e)}"

    async def simple_chat(
        self,
        system_prompt: str,
        user_prompt: str,
        model: Optional[str] = None,
    ) -> str:
        """
        简化的对话接口（用于记忆巩固和冲突解决）

        Args:
            system_prompt: 系统提示
            user_prompt: 用户提示
            model: 模型名称（默认 qwen-plus）

        Returns:
            str: 模型回复
        """
        return await self.chat(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            model=model or self.simple_model,
            temperature=0.3,  # 记忆提取用较低温度，更精确
            max_tokens=1000,
            stream=False,
        )