import os
import uuid
import json
import base64
from pathlib import Path
from typing import Type, Optional
from pydantic import BaseModel, Field, field_validator
from langchain.tools import BaseTool

class GuitarAnalysisInput(BaseModel):
    audio_path: Optional[str] = Field(None, description="本地绝对路径（调试/后端直传）")
    upload_file_id: Optional[str] = Field(None, description="前端上传任务ID（预留）")
    upload_base64: Optional[str] = Field(None, description="Base64 音频数据（前端直传备用）")
    target_dir: str = Field("./uploads/temp", description="音频缓存目录")
    original_name: Optional[str] = Field(None, description="前端原始文件名")

    @field_validator("audio_path", "upload_file_id", "upload_base64")
    @classmethod
    def validate_source(cls, v, info):
        if info.field_name == "audio_path" and v:
            if not Path(v).exists():
                raise ValueError(f"文件不存在: {v}")
            if not v.lower().endswith(('.wav', '.mp3', '.flac', '.ogg', '.aac')):
                raise ValueError("仅支持 .wav/.mp3/.flac/.ogg/.aac 格式")
        return v

class GuitarAnalyzerTool(BaseTool):
    # ⚠️ 关键修复：所有覆盖父类的字段必须带类型注解
    name: str = "guitar_arrangement_analyzer"
    description: str = "分析吉他音频并返回结构化编曲JSON。支持本地路径、Base64或上传ID。"
    args_schema: Type[BaseModel] = GuitarAnalysisInput  # ← 此行修复 Pydantic 报错

    def _resolve_audio_path(self, data: GuitarAnalysisInput) -> str:
        os.makedirs(data.target_dir, exist_ok=True)
        if data.audio_path:
            return str(Path(data.audio_path).resolve())

        ext = ".wav"
        if data.original_name:
            ext = Path(data.original_name).suffix.lower() or ".wav"
        safe_name = f"upload_{uuid.uuid4().hex[:8]}{ext}"
        dest = Path(data.target_dir) / safe_name

        if data.upload_base64:
            b64_data = data.upload_base64.split("base64,")[1] if "base64," in data.upload_base64 else data.upload_base64
            with open(dest, "wb") as f:
                f.write(base64.b64decode(b64_data))
        elif data.upload_file_id:
            raise NotImplementedError("请实现 upload_file_id 的文件桥接逻辑")
        else:
            raise ValueError("未提供有效音频源")
        return str(dest.resolve())

    def _run(self, **kwargs) -> str:
        input_data = GuitarAnalysisInput(**kwargs)
        audio_path = self._resolve_audio_path(input_data)

        # 替换为你的实际分析器导入路径
        from backend.tools.guitar_analyzer import GuitarArrangementAnalyzer
        analyzer = GuitarArrangementAnalyzer()
        result = analyzer.run(audio_path)
        return json.dumps(result, ensure_ascii=False, indent=2)