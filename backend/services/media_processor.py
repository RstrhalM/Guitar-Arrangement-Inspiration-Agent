import os
import asyncio
import aiohttp
import logging
from pathlib import Path
from typing import Optional, Dict
from ncmdump import NeteaseCloudMusicFile

CACHE_DIR = Path(__file__).parent.parent / "cache/media"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
logger = logging.getLogger(__name__)

NETEASE_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Referer": "https://music.163.com/"
}


async def _download_bytes(session: aiohttp.ClientSession, url: str, is_cover: bool = False) -> Optional[bytes]:
    if not url or not url.startswith("http"):
        return None
    try:
        headers = NETEASE_HEADERS if is_cover else None
        async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=10)) as resp:
            if resp.status == 200:
                return await resp.read()
            logger.warning(f"⚠️ HTTP {resp.status} 下载失败: {url[:60]}...")
    except Exception as e:
        logger.warning(f"⚠️ 网络请求异常: {e}")
    return None


def _decrypt_ncm_sync(ncm_path: Path) -> Optional[Path]:
    try:
        ncm = NeteaseCloudMusicFile(str(ncm_path))
        ncm.decrypt()
        out_path = ncm_path.with_suffix(".mp3")
        ncm.dump_music(str(out_path))
        return out_path
    except Exception as e:
        logger.warning(f"⚠️ ncmdump 解密失败: {e}")
        return None


async def process_song_media(
        song_id: str,
        remote_audio_url: str,
        remote_cover_url: str,
        title: str = "unknown"
) -> Dict[str, Optional[str]]:
    result = {"audio_path": None, "cover_path": None}

    async with aiohttp.ClientSession() as session:
        audio_bytes, cover_bytes = await asyncio.gather(
            _download_bytes(session, remote_audio_url, is_cover=False),
            _download_bytes(session, remote_cover_url, is_cover=True) if remote_cover_url else asyncio.sleep(0,
                                                                                                             result=None)
        )

        # 1️⃣ 处理音频
        if audio_bytes:
            is_ncm = audio_bytes[:4] == b"CTEN"
            ext = ".ncm" if is_ncm else ".mp3"
            raw_path = CACHE_DIR / f"{song_id}_raw{ext}"
            raw_path.write_bytes(audio_bytes)

            if is_ncm:
                dec_path = await asyncio.to_thread(_decrypt_ncm_sync, raw_path)
                if dec_path and dec_path.exists():
                    result["audio_path"] = dec_path.name
                    raw_path.unlink(missing_ok=True)
                else:
                    result["audio_path"] = raw_path.name
            else:
                result["audio_path"] = raw_path.name
            logger.info(f"✅ 音频处理完成: {result['audio_path']}")

        # 2️⃣ 处理封面（安全降级）
        if cover_bytes:
            cover_path = CACHE_DIR / f"{song_id}.jpg"
            cover_path.write_bytes(cover_bytes)
            result["cover_path"] = cover_path.name
            logger.info(f"✅ 封面下载成功: {cover_path.name}")
        else:
            result["cover_path"] = None
            logger.warning(f"⚠️ 封面获取失败，已降级为 None")

    return result