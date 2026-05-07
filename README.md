# 🎸 吉他编曲灵感推荐 Agent (Guitar Arrangement Inspiration Agent)

基于 **LangGraph + LLM + MCP 音乐数据协议** 构建的 AI 编曲辅助系统。面向吉他手与独立音乐制作人，通过音频特征解析、情感意图推断与风格引导检索，提供精准、可落地的参考曲目、段落演变分析与编曲实操建议。

---

## 🌟 核心特性

| 模块 | 能力 | 技术实现 |
|:---|:---|:---|
|  **音频解析** | 自动提取 MIDI/节拍/和声结构，清洗不合理和弦 | `librosa` + `basic_pitch` + 规则过滤 |
|  **情感与段落分析** | 推断编曲意图、张力值、节奏/和声演变轨迹 | LLM + 用户风格引导融合 Prompt |
| 🎯 **防幻觉推荐** | 严格基于真实艺人/曲目池筛选，100% 杜绝编造 | `Artist Generation → MCP Pool → LLM Filter` RAG 流水线 |
|  **量化契合度** | 风格(50%) + 节奏(30%) + 和声(20%) 加权评分 | Prompt 约束 + Pydantic 校验 |
|  **本地试听** | 自动下载音频/封面，解密 `.ncm` 格式，支持断点续播 | `aiohttp` + `ncmdump` + FastAPI `Range` 流式路由 |
| ⚡ **异步工作流** | 全链路非阻塞，支持并发请求与优雅降级 | LangGraph + `asyncio.gather` + 容错状态机 |

---

## 🏗️ 技术架构

```
前端 (Tailwind HTML)
  ↓ (上传音频 + 风格词)
FastAPI 路由 (/api/analyze/upload)
  ↓
LangGraph 状态机 (Async)
  ├─ 🔊 音频分析节点 (MIDI/Tempo/Harmony)
  ├─  情感意图节点 (风格融合 Prompt)
  ├─ 🎯 推荐节点 (4阶段 RAG: 艺人锚定 → MCP拉取 → LLM筛选 → 媒体增强)
  └─  缓存节点 (异步下载 + .ncm 解密)
  ↓
MCP Client (@eldment/meting-agent)
  ↓ (网易云/腾讯/酷狗/酷我)
本地缓存 (/cache/media/)
```

---

## 📂 项目结构

```
Music-Agent/
├── backend/
│   ├── main.py                  # FastAPI 入口 & 路由
│   ├── requirements.txt         # Python 依赖清单
│   ├── .env                     # 环境变量 (API Key / Cookie)
│   ├── workflow/
│   │   ├── graph.py             # LangGraph 状态机编排
│   │   └── state.py             # Pydantic 数据模型 & safe_dump
│   ├── agents/
│   │   ├── emotion_analysis_agent.py  # 情感意图节点
│   │   ├── recommendation_agent.py    # 推荐节点 (RAG防幻觉)
│   │   ├── media_cache_node.py        # 媒体缓存节点
│   │   └── prompts/                   # 独立 Prompt 文件
│   ├── services/
│   │   └── media_processor.py   # 异步下载 & ncmdump 解密
│   └── mcp/
│       └── meting_adapter.py    # MCP 工具封装 (search/url/pic/playlist)
── frontend/
│   └── index.html               # 单页应用 (上传/风格配置/结果渲染)
├── cache/media/                 # 运行时自动生成 (音频/封面缓存)
└── README.md                    # 本文件
```

---

## 🚀 快速开始

### 1. 环境准备
- Python `3.10+`
- Node.js `18+` (用于运行 MCP 服务)
- 推荐在虚拟环境中安装依赖

### 2. 安装依赖
```bash
cd backend
pip install -r requirements.txt
```

### 3. 配置环境变量
在 `backend/` 目录下创建 `.env` 文件：
```env
# LLM 配置 (通义千问 DashScope)
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxx
OPENAI_API_BASE=https://dashscope.aliyuncs.com/compatible-mode/v1

# MCP 音乐平台 Cookie (至少配置一个平台)
METING_NETEASE_COOKIE="__csrf=xxx; MUSIC_U=xxx; NMTID=xxx;"
# METING_TENCENT_COOKIE="uin=xxx; qm_keyst=xxx;"
```
>  **Cookie 获取**：浏览器登录对应音乐平台 → F12 → Application → Cookies → 拼接所需字段。

### 4. 启动服务
```bash
# 后端 (监听 8000 端口)
cd backend
uvicorn main:api --reload --host 0.0.0.0 --port 8000

# 前端
# 方式 A: 直接双击 frontend/index.html
# 方式 B: 使用 Live Server 或 Python 静态服务
cd frontend
python -m http.server 3000  # 访问 http://localhost:3000
```

---

## 🔄 核心工作流解析

### 🔹 情感意图节点 (`emotion_analysis_agent.py`)
- 接收音频分析 JSON + 用户风格引导词
- Prompt 动态注入风格偏好，输出核心情绪、张力值、`reference_tags`
- 按段落划分节奏演变 (`rhythm_sections`) 与和声色彩 (`harmonic_sections`)

### 🔹 推荐节点 (`recommendation_agent.py`) - 防幻觉四阶段
1. **艺人生成**：LLM 输出 2~3 位真实存在的吉他/器乐艺人
2. **MCP 拉取**：并行搜索艺人热门曲，构建 ≤8 首的真实曲池
3. **LLM 筛选**：基于曲池 + 用户风格 + 权重规则，挑选 2~3 首并生成专业分析
4. **媒体增强**：异步获取试听链接/封面，`.ncm` 自动解密转存本地

### 🔹 缓存节点 (`media_cache_node.py`)
- 拦截远程链接，调用 `aiohttp` 下载
- CPU 密集型解密操作移交 `asyncio.to_thread`，不阻塞事件循环
- 始终返回合法 State 更新，保障 LangGraph 状态机稳定

---



## ⚠️ 注意事项 & 常见问题

| 现象 | 原因 | 解决方案                                    |
|:---|:---|:----------------------------------------|
| `MCP error -32602` | 参数名/类型不匹配 Schema | 确保使用 `platform`/`keyword`/`type=1`      |
| 封面 404 / 音频下载慢 | 网易云 CDN 防盗链或网络波动 | 代码已内置 `Referer` 头与降级占位图，不影响核心流程         |
| LLM 输出 JSON 解析失败 | Prompt 约束过严或模型幻觉 | 已内置 `try-except` 清洗与字段兜底，自动触发降级推荐       |
| `.ncm` 解密失败 | 未安装 `ncmdump` 或版本不兼容 | `pip install ncmdump==1.1.0`，或检查系统编码环境  |
| 工作流卡在某个节点 | 异步调用阻塞或超时 | 所有 LLM/MCP 调用已改为 `ainvoke`，检查 API 配额与网络 |

---

##  许可证 & 致谢

- **开源协议**：MIT License
- **核心依赖**：
  - [LangGraph](https://github.com/langchain-ai/langgraph) - 异步状态机编排
  - [Meting-Agent](https://github.com/ELDment/Meting-Agent) - 跨平台音乐数据 MCP 服务
  - [Qwen / DashScope](https://dashscope.aliyun.com/) - 音乐领域 LLM 推理
  - [ncmdump](https://github.com/yt-dlp/ncmdump) - 网易云格式解密
- **适用场景**：个人学习、编曲灵感辅助、音乐教育演示。商业使用请遵守对应音乐平台版权协议。

> 🎵 *“让 AI 成为你的编曲副驾驶，而非黑盒生成器。”*  
> 如有架构优化建议或 Prompt 调优经验，欢迎提交 Issue 或 PR。