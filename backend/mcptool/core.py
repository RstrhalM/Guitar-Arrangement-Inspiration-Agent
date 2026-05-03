import os
import asyncio
from typing import Dict, Any
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from dotenv import load_dotenv

load_dotenv(override=True)

async def call_mcp_tool(tool_name: str, args: Dict[str, Any]) -> str:
    """按需创建 MCP 会话，严格限定在同一 async 作用域内，杜绝跨任务 cancel scope 冲突"""
    cookie = os.getenv("METING_NETEASE_COOKIE")
    if not cookie:
        raise ValueError("⚠️ 请在 .env 中配置 METING_NETEASE_COOKIE")

    params = StdioServerParameters(
        command="npx",
        args=["-y", "@eldment/meting-agent@latest"],
        env={"METING_NETEASE_COOKIE": cookie, **os.environ}
    )

    # 严格同步进入/退出，不跨越 LangGraph 的 Task 边界
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(tool_name, args)
            return result.content[0].text if result.content else "{}"