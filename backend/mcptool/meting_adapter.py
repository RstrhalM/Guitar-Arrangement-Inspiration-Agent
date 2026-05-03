from langchain_core.tools import tool
from .core import call_mcp_tool

# Meting 搜索类型码映射（源自 metowolf/Meting API）
SEARCH_TYPE_MAP = {
    "song": 1, "single": 1,
    "album": 10,
    "artist": 100, "singer": 100,
    "playlist": 1000, "list": 1000
}
VALID_PLATFORMS = {"netease", "tencent", "kugou", "kuwo"}


@tool
async def meting_search(
        keyword: str,  # ← 参数名必须是 keyword (单数)
        search_type: str = "song",
        platform: str = "netease",  # ← 参数名必须是 platform
        page: int = 1,
        limit: int = 20
) -> str:
    """搜索音乐资源"""
    if platform not in VALID_PLATFORMS:
        raise ValueError(f"Invalid platform: {platform}")

    type_code = SEARCH_TYPE_MAP.get(search_type.lower(), 1)

    # 🔑 MCP 底层调用：严格按 README Schema 传参
    return await call_mcp_tool("search", {
        "platform": platform.lower(),  # ← 不是 server
        "keyword": keyword.strip(),  # ← 不是 keywords / id
        "type": type_code,
        "page": page,
        "limit": min(limit, 100)
    })

@tool
async def meting_get_song(song_id: str, platform: str = "netease") -> str:
    """按歌曲 ID 获取详情"""
    if platform not in VALID_PLATFORMS:
        raise ValueError(f"Invalid platform: {platform}")
    return await call_mcp_tool("song", {
        "platform": platform.lower(),
        "id": str(song_id)
    })


@tool
async def meting_get_url(song_id: str, platform: str = "netease") -> str:
    """按歌曲 ID 获取播放链接"""
    if platform not in VALID_PLATFORMS:
        raise ValueError(f"Invalid platform: {platform}")
    return await call_mcp_tool("url", {
        "platform": platform.lower(),
        "id": str(song_id)
    })


@tool
async def meting_get_pic(
        resource_id: str,  # ← 参数名统一为 resource_id
        platform: str = "netease",
        resource_type: str = "song"
) -> str:
    """按资源 ID 获取封面。resource_type: song/album/artist"""
    if platform not in VALID_PLATFORMS:
        raise ValueError(f"Invalid platform: {platform}")

    return await call_mcp_tool("pic", {
        "platform": platform.lower(),
        "id": str(resource_id),  # MCP 底层用 id 字段
        "type": resource_type.lower()
    })

@tool
async def meting_get_lyric(song_id: str, platform: str = "netease") -> str:
    """按歌曲 ID 获取歌词"""
    if platform not in VALID_PLATFORMS:
        raise ValueError(f"Invalid platform: {platform}")
    return await call_mcp_tool("lyric", {
        "platform": platform.lower(),
        "id": str(song_id)
    })