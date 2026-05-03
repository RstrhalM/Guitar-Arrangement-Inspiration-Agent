from typing import Dict, List, Literal, Optional
from pydantic import BaseModel, Field
from langgraph.graph import MessagesState
#pydantic规范化输出
class ArrangementJSON(BaseModel):
    metadata: Dict
    tempo_structure: Dict
    harmonic_analysis: Dict
    summary: str

class EmotionIntent(BaseModel):
    core_emotion: str          # 如 "压抑释放型", "明朗叙事型"
    arrangement_purpose: str   # 如 "适合作为副歌前的情绪铺垫段落"
    tension_level: float       # 0.0 ~ 1.0 和声/节奏张力值
    reference_tags: List[str]  # 如 ["indie_pop", "post_rock", "guitar_driven"]

class SongRecommendation(BaseModel):
    title: str
    artist: str
    platform: str = "netease"
    match_reason: str
    chord_style_sim: str
    arrangement_tip: str
    preview_url: Optional[str] = None  # 🔑 新增：MCP 抓取的试听链接
    song_id: Optional[str] = None      # 🔑 新增：用于后续歌词/封面扩展

class MusicAgentState(MessagesState):
    input_type: Literal["audio", "text"] = "text"
    audio_path: Optional[str] = None
    analysis_json: Optional[ArrangementJSON] = None
    emotion_intent: Optional[EmotionIntent] = None
    recommendations: List[SongRecommendation] = []
    nlp_inspiration: Optional[str] = None
    final_output: Optional[dict] = None  # 👈 关键修复：声明为状态键

# 🔑 安全序列化函数（替代直接调用 .model_dump()）
def safe_dump(obj: any) -> any:
    """递归安全序列化：兼容 Pydantic 模型 / 列表 / 字典 / 基础类型"""
    if obj is None:
        return None
    if hasattr(obj, "model_dump"):  # Pydantic v2 模型
        return obj.model_dump()
    if isinstance(obj, list):
        return [safe_dump(item) for item in obj]
    if isinstance(obj, dict):
        return {k: safe_dump(v) for k, v in obj.items()}
    # 基础类型 (str/int/float/bool) 直接返回
    return obj