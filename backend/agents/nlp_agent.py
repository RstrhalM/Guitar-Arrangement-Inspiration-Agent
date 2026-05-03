import json
import re
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from prompts.nlp_prompt import NLP_INSPIRATION_SYSTEM

# 推荐使用低温度保证 JSON 结构稳定
llm = ChatOpenAI(model="qwen-plus", temperature=0.2)


def nlp_inspiration_node(state: dict) -> dict:
    """解析自然语言输入，输出结构化编曲灵感"""
    # 提取最新用户消息
    messages = state.get("messages", [])
    user_prompt = messages[-1].content if messages else ""

    if not user_prompt or len(user_prompt.strip()) < 5:
        return {"nlp_inspiration": json.dumps({
            "core_emotion": "未提供有效描述",
            "inspiration_summary": "请补充具体需求，例如：'想要一段适合过渡段的指弹前奏，带点爵士色彩，不要太复杂'"
        }, ensure_ascii=False)}

    system_msg = SystemMessage(content=NLP_INSPIRATION_SYSTEM)
    human_msg = HumanMessage(content=user_prompt)

    response = llm.invoke([system_msg, human_msg])

    # 鲁棒性 JSON 提取（兼容 LLM 偶尔包裹 ```json 的情况）
    raw_content = response.content.strip()
    if raw_content.startswith("```"):
        raw_content = re.sub(r"^```(?:json)?\n?", "", raw_content).rstrip("```").strip()

    try:
        parsed = json.loads(raw_content)
        return {"nlp_inspiration": json.dumps(parsed, ensure_ascii=False)}
    except json.JSONDecodeError:
        # 降级兜底：返回原始文本+安全提示
        return {"nlp_inspiration": json.dumps({
            "core_emotion": "解析异常",
            "inspiration_summary": raw_content[:300] + "...(格式解析降级)"
        }, ensure_ascii=False)}