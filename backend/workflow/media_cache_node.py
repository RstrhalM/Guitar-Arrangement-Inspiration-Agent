import asyncio
from workflow.state import SongRecommendation
from services.media_processor import process_song_media


async def media_cache_node(state: dict) -> dict:
    recs = state.get("recommendations", [])

    # 🔑 关键修复：LangGraph 禁止返回 {}，必须返回至少一个状态键
    if not recs:
        return {"recommendations": recs}

    async def _cache_one(rec: SongRecommendation) -> SongRecommendation:
        # 若无远程链接则跳过下载，直接返回原对象
        if not getattr(rec, "preview_url", None):
            return rec

        try:
            media = await process_song_media(
                song_id=getattr(rec, "song_id", "unknown"),
                remote_audio_url=rec.preview_url,
                remote_cover_url=getattr(rec, "cover_url", ""),
                title=rec.title
            )

            # 覆盖为本地缓存路径
            if media.get("audio_path"):
                rec.preview_url = media["audio_path"]
            if media.get("cover_path"):
                rec.cover_url = media["cover_path"]
        except Exception as e:
            print(f"⚠️ 媒体缓存节点异常 ({rec.title}): {e}")

        return rec

    tasks = [_cache_one(r) for r in recs]
    cached = await asyncio.gather(*tasks, return_exceptions=True)

    # 过滤异常结果，保留有效 SongRecommendation 对象
    valid_recs = [r for r in cached if isinstance(r, SongRecommendation)]

    # ✅ 始终返回 recommendations 字段，确保 LangGraph 状态机合法更新
    return {"recommendations": valid_recs}