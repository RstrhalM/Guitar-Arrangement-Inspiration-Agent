import os
import json
import re
import asyncio
import unicodedata
from typing import List, Optional, Dict, Any
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

from workflow.state import SongRecommendation, safe_dump
from prompts.recommendation_prompt import RECOMMENDATION_SYSTEM, ARTIST_GENERATION_SYSTEM
from mcptool.meting_adapter import meting_search, meting_get_url, meting_get_pic

load_dotenv(override=True)

llm = ChatOpenAI(
    model="qwen-plus",
    temperature=0.2,
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_API_BASE", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
    max_tokens=1500
)


# ==========================================
# 🔧 辅助函数 (保持不变)
# ==========================================
def _parse_mcp_response(raw: str) -> dict | list:
    if not raw or not raw.strip(): return {}
    try:
        parsed = json.loads(raw.strip())
        return parsed.get("data", parsed) if isinstance(parsed, dict) and parsed.get("ok") else parsed
    except json.JSONDecodeError:
        return {}


def _safe_extract(val, default: str = "") -> str:
    if isinstance(val, str): return val
    if isinstance(val, list):
        if not val: return default
        first = val[0]
        return first.get("name", str(first)) if isinstance(first, dict) else str(first)
    return str(val) if val else default


def _normalize(s: str) -> str:
    if not s: return ""
    return ''.join(c for c in unicodedata.normalize('NFKD', s) if not unicodedata.combining(c)).lower().strip()


def _extract_url(mcp_res: Any) -> Optional[str]:
    if not mcp_res or isinstance(mcp_res, Exception): return None
    raw = str(mcp_res).strip()
    if raw.startswith("http"): return raw
    try:
        data = json.loads(raw)

        def _find(obj):
            if isinstance(obj, str) and obj.startswith("http"): return obj
            if isinstance(obj, dict):
                for v in obj.values():
                    r = _find(v)
                    if r: return r
            if isinstance(obj, list):
                for i in obj:
                    r = _find(i)
                    if r: return r
            return None

        return _find(data)
    except:
        return raw if raw.startswith("http") else None


# ==========================================
# 🤖 推荐节点 (零冲突调用版)
# ==========================================
async def recommendation_node(state: dict) -> dict:
    intent = state.get("emotion_intent")
    analysis = state.get("analysis_json")
    style_guide = state.get("user_style_guide", "").strip()

    if not intent or not analysis:
        return {"recommendations": []}

    intent_dict = safe_dump(intent)
    analysis_dict = safe_dump(analysis)

    # 🔑 阶段1：生成艺人池 (使用独立 System)
    artist_context = f"【情感意图】\n{json.dumps(intent_dict, ensure_ascii=False)}\n【用户风格】{style_guide or '无特定限制'}"
    try:
        res = await llm.ainvoke([
            SystemMessage(content=ARTIST_GENERATION_SYSTEM),
            HumanMessage(content=artist_context)
        ])
        content = re.sub(r"^```(?:json)?\s*|\s*```$", "", res.content.strip(), flags=re.MULTILINE)
        artists = json.loads(content)
        if not isinstance(artists, list): artists = []
        artists = [a for a in artists if a.get("name")][:3]
    except Exception as e:
        print(f"⚠️ 艺人生成失败，降级: {e}")
        artists = [{"name": style_guide or "吉他编曲", "sub_genre": "generic", "reason": "fallback"}]

    # 🔑 阶段2：并行拉取真实曲池 (MCP)
    real_candidates = []

    async def _fetch_artist_pool(art: dict):
        try:
            song_res = await meting_search.ainvoke({
                "platform": "netease", "keyword": art["name"], "type": 1, "limit": 3
            })
            songs = _parse_mcp_response(song_res)
            if not isinstance(songs, list): return []
            return [{"name": _safe_extract(s.get("name")), "artist": _safe_extract(s.get("artist")),
                     "id": str(s.get("id")), "sub_genre": art.get("sub_genre", "")} for s in songs[:2]]
        except:
            return []

    tasks = [_fetch_artist_pool(a) for a in artists]
    fetched = await asyncio.gather(*tasks, return_exceptions=True)
    for res in fetched:
        if isinstance(res, list): real_candidates.extend(res)

    # 去重 & 截断
    seen, unique_pool = set(), []
    for c in real_candidates:
        key = f"{_normalize(c['name'])}|{_normalize(c['artist'])}"
        if key not in seen and len(unique_pool) < 8:
            seen.add(key)
            unique_pool.append(c)

    # 降级兜底
    if len(unique_pool) < 4:
        try:
            fallback_kw = style_guide or " ".join(intent_dict.get("reference_tags", [])[:3])
            song_res = await meting_search.ainvoke(
                {"platform": "netease", "keyword": fallback_kw, "type": 1, "limit": 6})
            songs = _parse_mcp_response(song_res)
            if isinstance(songs, list):
                for s in songs[:4]:
                    c = {"name": _safe_extract(s.get("name")), "artist": _safe_extract(s.get("artist")),
                         "id": str(s.get("id")), "sub_genre": "fallback"}
                    key = f"{_normalize(c['name'])}|{_normalize(c['artist'])}"
                    if key not in seen:
                        seen.add(key)
                        unique_pool.append(c)
        except:
            pass

    # 🔑 阶段3：LLM 筛选 (使用主 System + 纯数据 Human)
    candidate_text = "\n".join(
        [f"- {i + 1}. {c['name']} - {c['artist']} (ID: {c['id']}, 风格: {c['sub_genre']})" for i, c in
         enumerate(unique_pool)])

    filter_context = f"""【任务指令】
    请严格从下方【真实曲池】中挑选 2~3 首最佳参考曲目，并按权重量化评分。
    【原曲段落与情感】{json.dumps(intent_dict, ensure_ascii=False)}
    【用户风格】{style_guide or "无"}
    【真实曲池】(已按 2~3 位艺人各提供 1~2 首构建，严禁挑选列表外的歌曲)
    {candidate_text}

    【评分与筛选原则】
    1. "style_fit_score 必须严格按权重计算：风格/曲风匹配度 (50%) + 节拍/速度/段落呼吸感 (30%) + 和声/调性/和弦色彩 (20%)。"
    2. 优先保证艺人多样性，避免连续推荐同质化曲目。
    3. match_reason 必须结合原曲的"节拍/和声情感标签"说明匹配逻辑。
    4. 输出格式与字段要求严格遵守 System Prompt 规范。"""


    try:
        response = await llm.ainvoke([
            SystemMessage(content=RECOMMENDATION_SYSTEM),
            HumanMessage(content=filter_context)
        ])

        # 🔑 增强清洗：移除所有可能包裹的非 JSON 内容
        content = response.content.strip()
        # 1. 移除 Markdown 代码块
        content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content, flags=re.MULTILINE).strip()
        # 2. 若仍含非 JSON 前缀（如"好的，以下是..."），尝试提取首个 [ 开始的内容
        if not content.startswith("["):
            start_idx = content.find("[")
            if start_idx != -1:
                content = content[start_idx:]

        raw_recs = json.loads(content)

        # 🔑 新增：校验并修复缺失字段（防 LLM 漏输出）
        for rec in raw_recs:
            if not isinstance(rec, dict): continue
            # 必填字段兜底
            rec.setdefault("match_reason", "风格与情感高度契合")
            rec.setdefault("chord_style_sim", "和声/节奏特征匹配")
            rec.setdefault("arrangement_tip", "可参考原曲编曲技法")
            # style_fit_score 缺失时按权重估算
            if "style_fit_score" not in rec:
                rec["style_fit_score"] = 0.75  # 默认中高匹配

    except json.JSONDecodeError as e:
        print(f"❌ JSON 解析失败: {e}")
        print(f"📄 原始响应预览: {response.content[:300]}...")
        # 🔑 降级：尝试用 MCP 曲池直接返回（保底有结果）
        fallback_recs = []
        for c in unique_pool[:3]:
            fallback_recs.append({
                "title": c["name"], "artist": c["artist"],
                "match_reason": f"实时曲库匹配：{c['sub_genre']} 风格参考",
                "chord_style_sim": "基于艺人热门曲目推荐",
                "arrangement_tip": "建议参考原曲编曲结构与音色处理",
                "style_fit_score": 0.7
            })
        raw_recs = fallback_recs
    except Exception as e:
        print(f"❌ 筛选生成异常: {e}")
        return {"recommendations": []}

    # 🔑 阶段4：后处理校验 + 并发获取试听/封面
    async def _enrich_final(rec: dict) -> Optional[SongRecommendation]:
        title = rec.get("title", "").strip()
        artist = rec.get("artist", "").strip()
        if not title: return None

        matched = next((c for c in unique_pool if
                        _normalize(c["name"]) == _normalize(title) and _normalize(c["artist"]) == _normalize(artist)),
                       None)
        song_id = matched["id"] if matched else None

        preview_url = cover_url = None
        if song_id:
            url_res, pic_res = await asyncio.gather(
                meting_get_url.ainvoke({"platform": "netease", "song_id": song_id}),
                meting_get_pic.ainvoke({"platform": "netease", "resource_id": song_id, "resource_type": "song"}),
                return_exceptions=True
            )
            preview_url = _extract_url(url_res)
            cover_url = _extract_url(pic_res)

        return SongRecommendation(
            title=title, artist=artist,
            match_reason=rec.get("match_reason", ""),
            chord_style_sim=rec.get("chord_style_sim", ""),
            arrangement_tip=rec.get("arrangement_tip", ""),
            style_fit_score=rec.get("style_fit_score"),
            song_id=song_id, preview_url=preview_url, cover_url=cover_url
        )

    tasks = [_enrich_final(r) for r in raw_recs if isinstance(r, dict)]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    valid = [r for r in results if isinstance(r, SongRecommendation) and r.title]

    return {"recommendations": valid}