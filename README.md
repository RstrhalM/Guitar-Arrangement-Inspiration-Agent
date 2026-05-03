
---

# 🎸 Guitar Arrangement Inspiration Agent

> 基于 **LangGraph 多智能体工作流** 与 **MCP 音乐检索协议** 的吉他编曲灵感推荐系统。  
> 只需上传一段吉他音频，即可自动解析节拍/和声结构、推断编曲情感意图，并实时匹配参考曲目，附带专业级编曲技法建议与本地无损试听。

[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg)](https://fastapi.tiangolo.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2+-ff6f00.svg)](https://langchain-ai.github.io/langgraph/)

---

## ✨ 核心特性

| 模块 | 能力说明 |
|:---|:---|
| 🔊 **音频深度解析** | 🎵 基于 [Spotify Basic Pitch](https://github.com/spotify/basic-pitch) 转 MIDI + `librosa` 节拍/和声分析 + 失真扫弦清洗 |
| 🧠 **情感意图推断** | LLM 驱动的和声张力计算 × 节奏密度映射 → 输出结构化情感标签 |
| 🤖 **异步 Agent 路由** | LangGraph 状态机编排，音频/文本双通道隔离，支持热插拔节点与降级容错 |
| 🌐 **MCP 实时检索** | 🎧 基于 [ELDment/Meting-Agent](https://github.com/ELDment/Meting-Agent) 协议，支持多平台搜索、播放链接、歌词与封面 |
| 📦 **本地媒体缓存** | `aiohttp` 并发下载 + `ncmdump` 自动解密 `.ncm` → FastAPI `Accept-Ranges` 流式直出 |
| 🖥️ **开箱即用前端** | 拖拽上传、自适应播放器、封面懒加载、全链路降级渲染 |

---

## 🏗 项目结构

```
Music-Agent/
├── backend/                 # 后端服务（FastAPI + LangGraph + Agent）
│   ├── main.py              # FastAPI 入口 & 路由
│   ├── .env                 # 环境变量配置（Key / Cookie）
│   ├── workflow/            # LangGraph 状态定义与图编排
│   ├── agents/              # 情感分析 / 推荐 / NLP预留节点
│   ├── services/            # MCP客户端 / 媒体下载 / NCM解密
│   ├── uploads/             # 临时上传目录（自动清理）
│   └── cache/               # 本地媒体缓存（mp3/jpg）
├── frontend/                # 前端界面（与 backend 同级）
│   └── index.html           # 单文件现代 UI（Tailwind + Vanilla JS）
└── requirements.txt         # Python 依赖清单
```

---

## 🚀 快速开始

### 1️⃣ 环境准备
```bash
# 创建虚拟环境（推荐）
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt
```

### 2️⃣ 配置环境变量
在项目根目录或 `backend/` 下创建 `.env` 文件：
```env
# 🤖 LLM 配置（支持 OpenAI / 通义 DashScope 等兼容接口）
OPENAI_API_KEY=sk-xxxx
OPENAI_API_BASE=https://dashscope.aliyuncs.com/compatible-mode/v1

# 🎵 Meting-Agent Cookie（必填，至少配置一个平台）
METING_NETEASE_COOKIE="__csrf=xxx; MUSIC_U=xxx; NMTID=xxx;"
```
> 🔑 **获取方式**：浏览器登录对应音乐网页版 → `F12` → `Application` → `Cookies` → 复制对应键值拼接。

### 3️⃣ 启动服务
```bash
# ⚠️ 必须在项目根目录执行（包含 backend/ 和 frontend/ 的父级）
uvicorn backend.main:api --reload --host 0.0.0.0 --port 8000
```

### 4️⃣ 访问前端
- **方式 A**：直接双击打开 `frontend/index.html`
- **方式 B**：使用 Live Server 或 Python 静态服务器
  ```bash
  cd frontend
  python -m http.server 3000  # 访问 http://localhost:3000
  ```

---

## 🎵 MCP 音乐检索服务说明

本项目通过 `npx` 动态拉起 MCP Server，**无需额外安装 Python 包**。底层基于 [metowolf/Meting](https://github.com/metowolf/Meting) API 封装，由 [ELDment/Meting-Agent](https://github.com/ELDment/Meting-Agent) 提供标准化 MCP 协议支持。

### 🔹 支持平台与能力
| 平台 | 代号 | 提供能力 |
|:---|:---|:---|
| 网易云音乐 | `netease` | 搜索 / 歌曲详情 / 专辑 / 歌手 / 歌单 / 播放链接 / 歌词 / 封面 |
| 腾讯音乐 | `tencent` | 同上 |
| 酷狗音乐 | `kugou` | 同上 |
| 酷我音乐 | `kuwo` | 同上 |

### 🔹 可用 MCP 工具
| 工具名 | 功能说明 |
|:---|:---|
| `platforms` | 列出支持的平台和平台代号 |
| `search` | 按关键字搜索歌曲、专辑、歌手或平台特定资源 |
| `song` / `album` / `artist` / `playlist` | 按 ID 获取对应资源详情 |
| `url` | 按歌曲 ID 获取播放地址 |
| `lyric` | 按歌曲 ID 获取歌词 |
| `pic` | 按资源 ID 获取封面地址 |

### 🔹 接入配置示例
**Claude Desktop 配置**
```json
{
  "mcpServers": {
    "meting": {
      "command": "npx",
      "args": ["-y", "@eldment/meting-agent@latest"],
      "env": {
        "METING_NETEASE_COOKIE": "__csrf=...; MUSIC_U=...; NMTID=...;",
        "METING_TENCENT_COOKIE": "uin=...; qm_keyst=...;",
        "METING_KUGOU_COOKIE": "KugooID=...; t=...; dfid=...; mid=...;",
        "METING_KUWO_COOKIE": "HMACCOUNT=...; sid=...;"
      },
      "timeout": 60000
    }
  }
}
```

**Codex / 通用 stdio 配置**
```toml
[mcp_servers.meting]
type = "stdio"
command = "npx"
args = ["-y", "@eldment/meting-agent@latest"]
env = {
    METING_NETEASE_COOKIE = "__csrf=...; MUSIC_U=...; NMTID=...;",
    METING_TENCENT_COOKIE = "uin=...; qm_keyst=...;",
    METING_KUGOU_COOKIE = "KugooID=...; t=...; dfid=...; mid=...;",
    METING_KUWO_COOKIE = "HMACCOUNT=...; sid=...;"
}
tool_timeout_sec = 60
```

---

## 🔑 Cookie 鉴权规则

运行时按以下**优先级**自动获取鉴权凭据：
1. `METING_<PLATFORM>_COOKIE`（如 `METING_NETEASE_COOKIE`）
2. `METING_COOKIE`（通用兜底变量）
3. MCP 工具参数中传入的 `cookie` 字段

> 💡 **最佳实践**：仅使用单个平台时优先配置对应平台变量；需多平台轮询或统一鉴权时，仅设置 `METING_COOKIE` 即可。

---

## 🌟 核心开源依赖致谢

| 项目 | 说明 | 仓库 |
|:---|:---|:---|
| **Meting-Agent** | 基于 Meting API 构建的标准化 MCP 音乐检索服务 | [ELDment/Meting-Agent](https://github.com/ELDment/Meting-Agent) |
| **Basic Pitch** | Spotify 开源的轻量级音频转 MIDI 模型 | [spotify/basic-pitch](https://github.com/spotify/basic-pitch) |
| **LangGraph** | 基于状态机的多智能体编排框架 | [langchain-ai/langgraph](https://github.com/langchain-ai/langgraph) |
| **FastAPI** | 现代高性能异步 Web 框架 | [tiangolo/fastapi](https://github.com/tiangolo/fastapi) |

---

## 💡 常见问题排查

| 现象 | 可能原因 | 解决方案 |
|:---|:---|:---|
| `No module named 'backend'` | 未在根目录启动或 `sys.path` 未配置 | 确保执行 `uvicorn backend.main:api` 且位于项目根目录 |
| 封面/音频返回 `404` | 网易云直链防盗链或 Cookie 过期 | 检查 `.env` 中 `MUSIC_U` 是否有效；服务已内置 `Referer` 绕过 |
| `.ncm` 未转 `.mp3` | 未安装 `ncmdump` 或源文件非加密格式 | `pip install ncmdump`；直链 `mp3` 无需解密 |
| 推荐结果为空 | LLM 生成失败或 MCP 搜索限流 | 检查 API Key 余额；Cookie 含 `__csrf` 可提升请求成功率 |
| 端口被占用 | `8000` 被其他进程占用 | 修改启动参数 `--port 8001` |

---

## 📜 License

本项目仅供个人学习与技术研究使用。MCP 检索依赖第三方平台公开接口，请遵守各音乐平台使用协议，勿用于商业分发或大规模抓取。

---
🌟 **Star & Fork** 欢迎提交 Issue 或 PR 共同完善编曲灵感引擎！
