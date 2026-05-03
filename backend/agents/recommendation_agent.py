import os
import json
import re
import asyncio
import unicodedata
from typing import List, Optional, Dict
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from workflow.state import SongRecommendation
from prompts.recommendation_prompt import RECOMMENDATION_SYSTEM
from mcptool.meting_adapter import meting_search, meting_get_url, meting_get_pic

load_dotenv(override=True)

llm = ChatOpenAI(
    model="qwen-plus",
    temperature=0.2,
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_API_BASE", "https://dashscope.aliyuncs.com/compatible-mode/v1")
)


# ==========================================
# 🔧 内部辅助函数 (MCP 响应解析 & 智能匹配)
# ==========================================

def _parse_mcp_response(raw: str) -> dict | list:
    """解析 Meting-Agent 标准响应: {ok: bool,  list/dict}"""
    if not raw or not raw.strip(): return {}
    try:
        parsed = json.loads(raw.strip())
        if isinstance(parsed, dict) and parsed.get("ok") is True:
            return parsed.get("data", {})
        return parsed
    except json.JSONDecodeError:
        return {}


def _safe_extract(val, default: str = "") -> str:
    """安全提取字符串 (处理 list[dict]/list[str]/str 混合格式)"""
    if isinstance(val, str): return val
    if isinstance(val, list):
        if not val: return default
        first = val[0]
        return first.get("name", str(first)) if isinstance(first, dict) else str(first)
    return str(val) if val else default


def _normalize(s: str) -> str:
    """移除音标/变音符号 + 转小写，用于跨字符集匹配"""
    if not s: return ""
    return ''.join(c for c in unicodedata.normalize('NFKD', s) if not unicodedata.combining(c)).lower().strip()


def _extract_url(mcp_res: str) -> Optional[str]:
    """从 MCP 响应中递归提取 http 链接，兼容嵌套 JSON 与纯文本"""
    if not mcp_res: return None
    try:
        data = json.loads(mcp_res)
    except json.JSONDecodeError:
        data = None

    def find_in_obj(obj):
        if isinstance(obj, str) and obj.startswith("http"): return obj
        if isinstance(obj, dict):
            if "url" in obj and isinstance(obj["url"], str): return obj["url"]
            for v in obj.values():
                res = find_in_obj(v)
                if res: return res
        if isinstance(obj, list):
            for i in obj:
                res = find_in_obj(i)
                if res: return res
        return None

    if data:
        return find_in_obj(data)
    return mcp_res.strip() if mcp_res.strip().startswith("http") else None

# ==========================================
# 🤖 推荐节点核心逻辑
# ==========================================

async def recommendation_node(state: dict) -> dict:
    intent = state.get("emotion_intent")
    analysis = state.get("analysis_json")

    if not intent or not analysis:
        return {"recommendations": []}

    # 安全转为 dict
    intent_dict = intent.model_dump() if hasattr(intent, "model_dump") else intent
    analysis_dict = analysis.model_dump() if hasattr(analysis, "model_dump") else analysis

    # 1️⃣ LLM 生成候选曲目
    prompt = HumanMessage(content=f"""
    【情感意图】
    {json.dumps(intent_dict, ensure_ascii=False)}
    【原曲特征】
    {analysis_dict.get('summary', '')}
    请推荐 3 首真实存在的吉他/乐队参考曲目，严格按 JSON 数组输出，不要任何额外文本。
    """)

    try:
        res = await llm.ainvoke([SystemMessage(content=RECOMMENDATION_SYSTEM), prompt])
        content = re.sub(r"^```(?:json)?\s*|\s*```$", "", res.content.strip(), flags=re.MULTILINE)
        raw_recs = json.loads(content)
    except Exception as e:
        print(f"❌ LLM 生成推荐失败: {e}")
        return {"recommendations": []}

    # 2️⃣ 并发增强 MCP 数据
    async def _enrich_one(rec: dict) -> Optional[SongRecommendation]:
        title = rec.get("title", "").strip()
        artist = rec.get("artist", "").strip()
        if not title: return None

        search_kw = f"{title} {artist}".strip()
        try:
            # 🔑 严格对齐 README Schema
            search_res = await meting_search.ainvoke({
                "platform": "netease",
                "keyword": search_kw,
                "type": 1,  # 1=单曲
                "page": 1,
                "limit": 5
            })
            songs = _parse_mcp_response(search_res)
            if not isinstance(songs, list): songs = []

            # 🧠 评分匹配逻辑
            t_title, t_artist = _normalize(title), _normalize(artist)
            best_match, best_score = None, -1

            for s in songs:
                s_title = _normalize(_safe_extract(s.get("name")))
                s_artist = _normalize(_safe_extract(s.get("artist")))
                score = 0
                if t_title == s_title:
                    score += 20
                elif t_title in s_title:
                    score += 5
                if t_artist == s_artist:
                    score += 20
                elif t_artist in s_artist:
                    score += 5
                if score > best_score: best_score, best_match = score, s

            song_id = preview_url = cover_url = None
            if best_score >= 20 and best_match:
                song_id = str(best_match.get("id"))
                url_res, pic_res = await asyncio.gather(
                    meting_get_url.ainvoke({"platform": "netease", "song_id": song_id}),
                    meting_get_pic.ainvoke({"platform": "netease", "resource_id": song_id, "resource_type": "song"}),
                    return_exceptions=True
                )
                preview_url = _extract_url(url_res) if not isinstance(url_res, Exception) else None
                cover_url = _extract_url(pic_res) if not isinstance(pic_res, Exception) else None

            return SongRecommendation(
                title=title, artist=artist,
                match_reason=rec.get("match_reason", ""),
                chord_style_sim=rec.get("chord_style_sim", ""),
                arrangement_tip=rec.get("arrangement_tip", ""),
                song_id=song_id, preview_url=preview_url, cover_url=cover_url
            )
        except Exception as e:
            print(f"⚠️ 歌曲 {title} MCP 增强失败: {e}")
            return SongRecommendation(title=title, artist=artist, match_reason=rec.get("match_reason", ""),
                                      chord_style_sim=rec.get("chord_style_sim", ""),
                                      arrangement_tip=rec.get("arrangement_tip", ""))

    tasks = [_enrich_one(r) for r in raw_recs if isinstance(r, dict)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    valid = [r for r in results if isinstance(r, SongRecommendation)]
    return {"recommendations": valid}