"""
当前时间工具 (Get Current Time Tool)

获取准确的当前时间，支持时区指定。
"""

from __future__ import annotations

from datetime import datetime, timezone, timedelta

from app.tools.base_tools import BaseTool, ToolParameter


class GetCurrentTimeTool(BaseTool):
    """获取当前时间的工具"""

    @property
    def name(self) -> str:
        return "get_current_time"

    @property
    def description(self) -> str:
        return "获取当前日期和时间。用于需要知道当前时间的场景。"

    @property
    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="timezone",
                type="string",
                description="时区，如 'Asia/Shanghai', 'UTC', 'America/New_York'，默认 'UTC'",
                required=False,
            ),
        ]

    # 常用时区偏移映射
    _TIMEZONE_OFFSETS = {
        "Asia/Shanghai": 8,
        "Asia/Tokyo": 9,
        "Asia/Seoul": 9,
        "Asia/Singapore": 8,
        "Asia/Kolkata": 5.5,
        "Europe/London": 1,
        "Europe/Paris": 2,
        "Europe/Berlin": 2,
        "America/New_York": -4,
        "America/Chicago": -5,
        "America/Los_Angeles": -7,
        "Pacific/Auckland": 12,
    }

    async def execute(self, timezone: str = "UTC", **kwargs) -> str:
        """
        获取当前时间

        Args:
            timezone: 时区名称

        Returns:
            str: 格式化的当前时间
        """
        utc_now = datetime.now(timezone.utc)

        # 计算时区偏移
        offset_hours = self._TIMEZONE_OFFSETS.get(timezone, 0)
        local_now = utc_now + timedelta(hours=offset_hours)

        # 格式化输出
        date_str = local_now.strftime("%Y年%m月%d日")
        time_str = local_now.strftime("%H:%M:%S")
        weekday_str = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"][
            local_now.weekday()
        ]

        return (
            f"当前时间（{timezone}）:\n"
            f"{date_str} {time_str} {weekday_str}\n"
            f"UTC时间: {utc_now.strftime('%Y-%m-%d %H:%M:%S')} UTC"
        )