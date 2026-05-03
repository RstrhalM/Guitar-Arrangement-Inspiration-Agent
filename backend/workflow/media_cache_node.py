import asyncio
from workflow.state import SongRecommendation
from services.media_processor import process_song_media


async def media_cache_node(state: dict) -> dict:
    recs = state.get("recommendations", [])
    if not recs: return {}

    async def _cache_one(rec: SongRecommendation) -> SongRecommendation:
        if not getattr(rec, "preview_url", None): return rec

        media = await process_song_media(
            song_id=getattr(rec, "song_id", "0"),
            remote_audio_url=rec.preview_url,
            remote_cover_url=getattr(rec, "cover_url", ""),
            title=rec.title
        )

        # 覆盖为本地流路由路径 (供 FastAPI 直接访问)
        if media.get("audio_path"):
            rec.preview_url = media["audio_path"]
        if media.get("cover_path"):
            rec.cover_url = media["cover_path"]
        return rec

    tasks = [_cache_one(r) for r in recs]
    cached = await asyncio.gather(*tasks, return_exceptions=True)
    return {"recommendations": [r for r in cached if isinstance(r, SongRecommendation)]}