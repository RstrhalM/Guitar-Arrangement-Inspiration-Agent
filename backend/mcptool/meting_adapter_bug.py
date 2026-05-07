from langchain_core.tools import tool
from .core import call_mcp_tool

# Meting 标准搜索类型码映射（源自 README 与 metowolf/Meting API）
SEARCH_TYPE_MAP = {
    "song": 1, "single": 1,
    "album": 10,
    "artist": 100, "singer": 100,
    "playlist": 1000, "list": 1000
}
VALID_PLATFORMS = {"netease", "tencent", "kugou", "kuwo"}

@tool
async def meting_search(keyword: str, search_type: str = "song", platform: str = "netease", page: int = 1, limit: int = 20) -> str:
    """搜索音乐资源。platform: netease/tencent/kugou/kuwo; type: song/album/artist/playlist"""
    if platform not in VALID_PLATFORMS:
        raise ValueError(f"Invalid platform: {platform}")
    type_code = SEARCH_TYPE_MAP.get(search_type.lower(), 1)
    return await call_mcp_tool("search", {
        "platform": platform.lower(),
        "keyword": keyword.strip(),
        "type": type_code,
        "page": page,
        "limit": min(limit, 100)
    })

@tool
async def meting_get_playlist(playlist_id: str, platform: str = "netease") -> str:
    """按歌单 ID 获取详情（包含完整曲目列表，官方支持工具）"""
    if platform not in VALID_PLATFORMS:
        raise ValueError(f"Invalid platform: {platform}")
    return await call_mcp_tool("playlist", {"platform": platform.lower(), "id": str(playlist_id)})

@tool
async def meting_get_song(song_id: str, platform: str = "netease") -> str:
    """按歌曲 ID 获取详情"""
    if platform not in VALID_PLATFORMS:
        raise ValueError(f"Invalid platform: {platform}")
    return await call_mcp_tool("song", {"platform": platform.lower(), "id": str(song_id)})

@tool
async def meting_get_url(song_id: str, platform: str = "netease") -> str:
    """按歌曲 ID 获取播放链接"""
    if platform not in VALID_PLATFORMS:
        raise ValueError(f"Invalid platform: {platform}")
    return await call_mcp_tool("url", {"platform": platform.lower(), "id": str(song_id)})

@tool
async def meting_get_pic(resource_id: str, platform: str = "netease", resource_type: str = "song") -> str:
    """按资源 ID 获取封面。resource_type: song/album/artist"""
    if platform not in VALID_PLATFORMS:
        raise ValueError(f"Invalid platform: {platform}")
    return await call_mcp_tool("pic", {
        "platform": platform.lower(),
        "id": str(resource_id),
        "type": resource_type.lower()
    })

@tool
async def meting_get_lyric(song_id: str, platform: str = "netease") -> str:
    """按歌曲 ID 获取歌词"""
    if platform not in VALID_PLATFORMS:
        raise ValueError(f"Invalid platform: {platform}")
    return await call_mcp_tool("lyric", {"platform": platform.lower(), "id": str(song_id)})