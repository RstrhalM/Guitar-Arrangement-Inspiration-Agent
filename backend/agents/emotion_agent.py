import json
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from prompts.emotion_prompt import EMOTION_ANALYSIS_SYSTEM
from workflow.state import EmotionIntent
import os
from dotenv import load_dotenv
# ⚠️ 必须放在所有 LLM 实例化之前，且加 override=True
load_dotenv(override=True)

# 调试验证（运行后看控制台是否打印出 Key 前缀）
print(f"🔑 OPENAI_API_KEY 状态: {'✅已加载' if os.getenv('OPENAI_API_KEY') else '❌未找到'}")
llm = ChatOpenAI(
    model="qwen-plus", temperature=0.2,
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_API_BASE", "https://dashscope.aliyuncs.com/compatible-mode/v1")
)


def emotion_analysis_node(state: dict) -> dict:
    data = state["analysis_json"]
    # 兼容 Pydantic 实例与 dict
    json_payload = data.model_dump() if hasattr(data, "model_dump") else data

    prompt = HumanMessage(content=json.dumps(json_payload, ensure_ascii=False))
    system = SystemMessage(content=EMOTION_ANALYSIS_SYSTEM)

    try:
        response = llm.invoke([system, prompt])
        intent_dict = json.loads(response.content)
        from backend.workflow.state import EmotionIntent
        return {"emotion_intent": EmotionIntent(**intent_dict)}
    except Exception:
        from backend.workflow.state import EmotionIntent
        return {"emotion_intent": EmotionIntent(
            core_emotion="动态推进型",
            arrangement_purpose="通用参考",
            tension_level=0.5,
            reference_tags=["indie_guitar"]
        )}