from langgraph.graph import StateGraph, END
from workflow.state import MusicAgentState, safe_dump,ArrangementJSON
from agents.emotion_agent import emotion_analysis_node
from agents.recommendation_agent import recommendation_node
from agents.nlp_agent import nlp_inspiration_node
from tools.guitar_tool import GuitarAnalyzerTool
import json
from workflow.media_cache_node import media_cache_node
guitar_tool = GuitarAnalyzerTool()

def route_input(state: MusicAgentState) -> str:
    return "audio_analysis" if state.get("input_type") == "audio" else "nlp_inspiration_handler"

def audio_analysis_node(state: MusicAgentState) -> dict:
    raw_json = guitar_tool.run({"audio_path": state["audio_path"]})
    parsed_dict = json.loads(raw_json)
    # 🔑 核心修复：将 dict 转为 Pydantic 模型，支持 .tempo_structure 等点号访问
    return {"analysis_json": ArrangementJSON.model_validate(parsed_dict)}

def merge_output(state: MusicAgentState) -> dict:
    return {"final_output": {
        "path": state.get("input_type"),
        "analysis": safe_dump(state.get("analysis_json")),
        "emotion": safe_dump(state.get("emotion_intent")),
        "nlp": state.get("nlp_inspiration"),
        "recommendations": safe_dump(state.get("recommendations"))
    }}

workflow = StateGraph(MusicAgentState)
workflow.add_node("audio_analysis", audio_analysis_node)
workflow.add_node("emotion_analysis", emotion_analysis_node)
workflow.add_node("nlp_inspiration_handler", nlp_inspiration_node)
workflow.add_node("recommendation", recommendation_node)
workflow.add_node("media_cache", media_cache_node)
workflow.add_node("merge", merge_output)

workflow.set_conditional_entry_point(route_input)
workflow.add_edge("audio_analysis", "emotion_analysis")
workflow.add_edge("emotion_analysis", "recommendation")
workflow.add_edge("recommendation", "media_cache")
workflow.add_edge("media_cache", "merge")
workflow.add_edge("nlp_inspiration_handler", "merge")
workflow.add_edge("merge", END)

app = workflow.compile()