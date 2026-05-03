import asyncio
import os
import sys
import json
import logging
from pathlib import Path
from dotenv import load_dotenv

# 1. 路径与环境初始化
sys.path.insert(0, str(Path(__file__).parent.resolve()))
load_dotenv(override=True)
logging.basicConfig(level=logging.INFO, format="%(levelname)-7s | %(message)s")

from workflow.graph import app
from workflow.state import safe_dump

CACHE_DIR = Path(__file__).parent / "cache/media"


async def verify_cache(recommendations: list) -> dict:
    """验证推荐歌曲是否已成功下载至本地缓存"""
    stats = {"total": len(recommendations), "cached": 0, "failed": 0, "details": []}
    for rec in recommendations:
        audio_path = rec.get("preview_url")
        cover_path = rec.get("cover_url")

        audio_ok = bool(audio_path and (CACHE_DIR / audio_path).exists())
        cover_ok = bool(cover_path and (CACHE_DIR / cover_path).exists())

        if audio_ok:
            stats["cached"] += 1
        else:
            stats["failed"] += 1

        stats["details"].append({
            "title": rec.get("title"),
            "artist": rec.get("artist"),
            "audio_downloaded": audio_ok,
            "cover_downloaded": cover_ok,
            "local_audio": str(CACHE_DIR / audio_path) if audio_ok else None
        })
    return stats


async def run_full_pipeline(audio_path: str):
    logging.info("=" * 70)
    logging.info("🚀 启动：音频分析 → 情感意图 → MCP推荐 → 本地下载 全链路测试")
    logging.info("=" * 70)

    # 0️⃣ 环境校验
    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError("❌ 未找到 OPENAI_API_KEY，请在 .env 中配置")
    cookie = os.getenv("METING_NETEASE_COOKIE") or os.getenv("METING_COOKIE")
    if not cookie:
        logging.warning("⚠️ 未配置网易云 Cookie，MCP 搜索/下载可能受限")

    audio_file = Path(audio_path).absolute()
    if not audio_file.exists():
        raise FileNotFoundError(f"❌ 找不到音频文件: {audio_file}")

    # 1️⃣ 构建初始状态
    initial_state = {
        "messages": [],
        "input_type": "audio",
        "audio_path": str(audio_file),
        "analysis_json": None,
        "emotion_intent": None,
        "nlp_inspiration": None,
        "recommendations": [],
        "final_output": None
    }

    try:
        # 2️⃣ 执行 LangGraph 工作流
        logging.info("⏳ 正在执行工作流 (音频转MIDI/节拍/和声 → 情感解析 → MCP搜索 → 并发下载)...")
        result = await app.ainvoke(initial_state)
        final = result.get("final_output")

        if not final:
            logging.error("❌ 工作流执行完毕但未生成 final_output")
            return

        # 3️⃣ 安全序列化 & 提取推荐列表
        recs = safe_dump(final.get("recommendations", []))
        if not isinstance(recs, list):
            recs = []

        # 4️⃣ 验证本地缓存
        logging.info("📦 正在验证本地缓存目录...")
        cache_stats = await verify_cache(recs)

        # 5️⃣ 打印结构化结果
        print("\n" + "=" * 70)
        print("✅ 全链路执行成功！输出 JSON 如下：")
        print("=" * 70)
        print(json.dumps({
            "path": final.get("path"),
            "emotion": safe_dump(final.get("emotion")),
            "analysis_summary": safe_dump(final.get("analysis")).get("summary"),
            "recommendations_count": len(recs),
            "cache_status": cache_stats,
            "recommendations": recs
        }, indent=2, ensure_ascii=False))

        print("\n" + "-" * 70)
        print("📊 缓存验证报告:")
        for i, d in enumerate(cache_stats["details"], 1):
            status = "✅ 已下载" if d["audio_downloaded"] else "❌ 下载失败/无链接"
            print(f"  {i}. {d['title']} - {d['artist']} | {status}")
            if d["local_audio"]:
                print(f"     📁 本地路径: {d['local_audio']}")
        print("-" * 70)

        return final

    except Exception as e:
        logging.error(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    # 🔧 替换为你的测试音频路径
    TEST_AUDIO = "小小奇迹.mp3"
    asyncio.run(run_full_pipeline(TEST_AUDIO))
