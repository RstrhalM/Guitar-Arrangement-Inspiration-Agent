# backend/main.py 头部
import sys
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from typing import Optional
# 🔑 自动将项目根目录加入 Python 搜索路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import os
import uuid
import asyncio
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from workflow.graph import app as langgraph_app
from workflow.state import safe_dump

# 目录初始化
UPLOAD_DIR = Path("./uploads/temp")
CACHE_DIR = Path("./cache/media")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

api = FastAPI(title="Guitar Arrangement Agent", version="1.0.0")

# CORS (开发环境允许全放，生产请限制域名)
api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 🔑 媒体流式路由
@api.get("/api/media/stream/{filename}")
async def stream_media(filename: str):
    file_path = CACHE_DIR / filename
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(404, "File not found")
    mime_map = {"mp3": "audio/mpeg", "flac": "audio/flac", "wav": "audio/wav",
                "jpg": "image/jpeg", "png": "image/png"}
    mime = mime_map.get(file_path.suffix.lstrip("."), "application/octet-stream")
    return FileResponse(path=str(file_path), media_type=mime, headers={"Accept-Ranges": "bytes"})


# 🔑 异步清理临时文件
async def _cleanup(file_path: Path):
    await asyncio.sleep(5)  # 延迟清理确保工作流读取完毕
    if file_path.exists(): file_path.unlink(missing_ok=True)


# 🔑 音频分析入口
@api.post("/api/analyze/upload")
async def analyze_audio(
    file: UploadFile = File(...),
    style: Optional[str] = Form(default=None, description="风格引导（如：后摇氛围/指弹分解/失真扫弦/布鲁斯推弦）")
):
    if not file.filename.lower().endswith(('.mp3', '.wav', '.flac', '.ogg', '.aac')):
        raise HTTPException(400, "Unsupported format. Use .mp3/.wav/.flac/.ogg/.aac")

    temp_id = f"upload_{uuid.uuid4().hex[:8]}"
    ext = Path(file.filename).suffix
    temp_path = UPLOAD_DIR / f"{temp_id}{ext}"

    try:
        content = await file.read()
        temp_path.write_bytes(content)

        state = {
            "messages": [], "input_type": "audio",
            "audio_path": str(temp_path.resolve()),
            "analysis_json": None, "emotion_intent": None,
            "nlp_inspiration": None, "recommendations": [],
            "user_style_guide": style,  # 🔑 新增：透传至工作流
            "final_output": None
        }

        result = await langgraph_app.ainvoke(state)
        final = result.get("final_output")
        if not final: raise HTTPException(500, "Workflow produced no output")

        return JSONResponse(content={"status": "success", "data": safe_dump(final)})

    except Exception as e:
        raise HTTPException(500, f"Processing failed: {str(e)}")
    finally:
        asyncio.create_task(_cleanup(temp_path))


# 🔑 NLP 预留入口 (未来接入)
@api.post("/api/analyze/text")
async def analyze_text(prompt: str):
    return JSONResponse(content={
        "status": "success",
        "data": {
            "path": "text",
            "nlp_inspiration": f"[NLP预留] 已接收需求: {prompt}",
            "recommendations": [], "emotion": None
        }
    })


@api.get("/health")
async def health(): return {"status": "ok",
                            "cache_size_mb": sum(f.stat().st_size for f in CACHE_DIR.rglob('*')) / 1024 / 1024}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(api, host="0.0.0.0", port=8000, reload=True)