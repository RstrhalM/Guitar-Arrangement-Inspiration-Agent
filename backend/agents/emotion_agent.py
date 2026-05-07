import json
import asyncio
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from prompts.emotion_prompt import EMOTION_ANALYSIS_SYSTEM
from workflow.state import EmotionIntent
import os
from dotenv import load_dotenv

# ⚠️ 必须放在所有 LLM 实例化之前，且加 override=True
load_dotenv(override=True)

llm = ChatOpenAI(
    model="qwen-plus",
    temperature=0.2,
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_API_BASE", "https://dashscope.aliyuncs.com/compatible-mode/v1")
)


async def emotion_analysis_node(state: dict) -> dict:
    """情感分析节点：注入用户风格引导 → LLM 推理 → 结构化输出"""
    data = state.get("analysis_json")
    style_guide = state.get("user_style_guide", "").strip()

    if not data:
        return {"emotion_intent": None}

    # 兼容 Pydantic 实例与 dict
    json_payload = data.model_dump() if hasattr(data, "model_dump") else data

    # 🔑 核心改动：将用户风格注入 Prompt 上下文，激活 emotion_prompt.py 中的风格分析维度
    style_context = f"\n【用户风格引导】{style_guide}" if style_guide else "\n【用户风格引导】无特定限制，请纯基于音频特征分析"
    prompt_content = f"【音频分析数据】\n{json.dumps(json_payload, ensure_ascii=False)}{style_context}"

    prompt = HumanMessage(content=prompt_content)
    system = SystemMessage(content=EMOTION_ANALYSIS_SYSTEM)

    try:
        # 🔑 改为异步调用，匹配 LangGraph 全局 async 流水线
        response = await llm.ainvoke([system, prompt])
        intent_dict = json.loads(response.content)

        # 严格 Pydantic 校验
        return {"emotion_intent": EmotionIntent(**intent_dict)}

    except Exception as e:
        print(f"️ 情感分析节点失败，使用降级默认值: {e}")
        # 🔑 降级对象需补齐新增字段，防止 Pydantic 校验拦截
        return {"emotion_intent": EmotionIntent(
            core_emotion="动态推进型",
            arrangement_purpose="通用参考",
            tension_level=0.5,
            reference_tags=["indie_guitar"],
            rhythm_sections=[],
            rhythm_overall_desc="无明显节奏变化",
            harmonic_sections=[],
            harmonic_overall_desc="和声色彩稳定延续"
        )}