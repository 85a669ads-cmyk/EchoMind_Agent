"""
网络搜索工具 (Search Web Tool)

模拟搜索功能 - 在真实部署中可替换为 SerpAPI、Bing Search API 等。
"""

from __future__ import annotations

from datetime import datetime, timezone

from app.tools.base_tools import BaseTool, ToolParameter


class SearchWebTool(BaseTool):
    """互联网搜索工具（模拟实现）"""

    @property
    def name(self) -> str:
        return "search_web"

    @property
    def description(self) -> str:
        return "搜索互联网获取最新信息。用于回答需要实时数据的问题。"

    @property
    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="query",
                type="string",
                description="搜索查询关键词",
                required=True,
            ),
            ToolParameter(
                name="num_results",
                type="integer",
                description="返回结果数量，默认5",
                required=False,
            ),
        ]

    async def execute(self, query: str = "", num_results: int = 5, **kwargs) -> str:
        """
        执行搜索（当前为模拟实现）

        在生产环境中应替换为:
        - SerpAPI
        - Bing Search API
        - 阿里云搜索服务

        Args:
            query: 搜索关键词
            num_results: 结果数量

        Returns:
            str: 模拟的搜索结果
        """
        if not query.strip():
            return "错误：搜索关键词不能为空。"

        # 模拟搜索结果
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

        simulated_results = [
            {
                "title": f"搜索结果 1 - {query}",
                "snippet": f"关于'{query}'的最新信息。这是一条模拟搜索结果，"
                          f"在实际部署中，此处将显示真实的搜索引擎返回内容。",
                "url": f"https://example.com/search?q={query.replace(' ', '+')}"
            },
            {
                "title": f"搜索结果 2 - {query}更多内容",
                "snippet": f"关于'{query}'的更多详细信息和深度分析。",
                "url": f"https://example.com/search?q={query.replace(' ', '+')}&page=2"
            },
        ]

        results_text = f"[搜索时间: {now}]\n"
        results_text += f"搜索关键词: \"{query}\"\n\n"
        for i, result in enumerate(simulated_results[:num_results], 1):
            results_text += (
                f"{i}. **{result['title']}**\n"
                f"   {result['snippet']}\n"
                f"   URL: {result['url']}\n\n"
            )

        results_text += "（注：当前为模拟搜索结果，实际部署请配置搜索 API）"
        return results_text