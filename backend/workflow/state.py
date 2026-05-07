from typing import Dict, List, Literal, Optional, Any
from pydantic import BaseModel, Field
from langgraph.graph import MessagesState

class RhythmSection(BaseModel):
    measure_range: str = Field(description="小节范围，如 '1-16'")
    change_desc: str = Field(description="节拍/速度大概变化描述")
    emotion_tag: str = Field(description="节拍情感标签，如 '稳步推进/呼吸留白/急促驱动'")

class HarmonicSection(BaseModel):
    measure_range: str = Field(description="小节范围")
    harmony_desc: str = Field(description="该段落主要和声进行与色彩特征")
    has_tension_harmony: bool = Field(description="是否包含张力和声（如减和弦、挂留、延伸音）")
    emotion_tag: str = Field(description="和声情感标签，如 '明朗开阔/悬置不安/暗黑压迫/温暖治愈'")

class EmotionIntent(BaseModel):
    core_emotion: str
    arrangement_purpose: str
    tension_level: float
    reference_tags: List[str] = Field(default_factory=list)
    # 🔑 新增段落级分析字段
    rhythm_sections: List[RhythmSection] = Field(default_factory=list)
    rhythm_overall_desc: str = ""
    harmonic_sections: List[HarmonicSection] = Field(default_factory=list)
    harmonic_overall_desc: str = ""

class SongRecommendation(BaseModel):
    title: str
    artist: str
    platform: str = "netease"
    match_reason: str
    chord_style_sim: str
    arrangement_tip: str
    style_fit_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    preview_url: Optional[str] = None
    cover_url: Optional[str] = None
    song_id: Optional[str] = None

class ArrangementJSON(BaseModel):
    metadata: Dict
    tempo_structure: Dict
    harmonic_analysis: Dict
    summary: str

class MusicAgentState(MessagesState):
    input_type: Literal["audio", "text"] = "text"
    audio_path: Optional[str] = None
    analysis_json: Optional[ArrangementJSON] = None
    emotion_intent: Optional[EmotionIntent] = None
    user_style_guide: Optional[str] = None
    recommendations: List[SongRecommendation] = []
    nlp_inspiration: Optional[str] = None
    final_output: Optional[dict] = None

def safe_dump(obj: Any) -> Any:
    if obj is None: return None
    if hasattr(obj, "model_dump"): return obj.model_dump()
    if isinstance(obj, list): return [safe_dump(i) for i in obj]
    if isinstance(obj, dict): return {k: safe_dump(v) for k, v in obj.items()}
    return obj