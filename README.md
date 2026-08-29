# 校园事务智能助手 🎓

面向本科生的**多模态校园信息处理**工具：把课程通知、活动海报、作业要求、报修材料（图片 / PDF / 文本）丢进来，系统自动完成 **通知识别 → 关键信息抽取 → 待办生成 → 任务追踪**，帮你不再错过截止时间。

> 开箱即用：**无需任何 API Key**。默认用内置规则（Mock VLM + 本地哈希向量 + Stub OCR）即可跑通完整链路；接入阿里云百炼 `qwen3-vl-plus` / `qwen3.5-vl-plus` 等视觉大模型后抽取与向量质量进一步提升。

---

## ✨ 功能特性

- **多模态导入**：图片（海报/截图）、PDF（课程通知/作业要求）、纯文本，拖拽或粘贴即可。
- **通知识别 + 关键信息抽取**：自动判定类别（课程通知 / 活动海报 / 作业要求 / 报修材料），抽取标题、发布方、地点、课程、截止时间、活动时间、联系人、标签。
- **中文时间语义**：相对日期（"3天后"）、星期（"周五"）、改期 vs 原定（"调整至周五…原定周三…"）精准区分；截止默认补到 23:59。
- **待办生成**：按规则自动拆出"开始动手 / 提交"等任务，带优先级与提醒时间。
- **任务追踪**：看板（待办 / 进行中 / 已完成 / 已归档）状态流转，逾期自动标红，状态变更写入审计时间线。
- **语义检索**：基于 FAISS 的向量检索，用自然语言在历史通知里找东西（"下周的作业啥时候交？"）。
- **去重**：内容向量相似度 ≥ 0.90 判定为重复通知，自动跳过入库。
- **人工复核**：低置信度结果进"待复核"，可在前端修正，保存后自动重算待办。

技术细节：**LangGraph** 编排识别→抽取→校验→去重→待办 的流水线（未装 langgraph 时自动降级为顺序执行）；**SQLAlchemy 2.0** 双兼容 SQLite / PostgreSQL；**PaddleOCR / RapidOCR** 可插拔，缺失时优雅降级。

---

## 🧱 技术栈

| 层 | 选型 |
|---|---|
| 前端 | React 18 + TypeScript + Vite |
| 后端 | Python + FastAPI + SQLAlchemy 2.0 + Pydantic v2 |
| 流程编排 | LangGraph（可选，自动降级） |
| 视觉大模型 | Mock / 阿里云百炼 OpenAI 兼容接口（qwen3-vl-plus 等） |
| OCR | PaddleOCR / RapidOCR / VLM / Stub（自动探测） |
| 向量检索 | FAISS（缺失时 NumPy 余弦回退） |
| 数据库 | SQLite（开发）/ PostgreSQL（生产，可选 pgvector） |

---

## 📁 目录结构

```
campus-assistant/
├── backend/                 # FastAPI 后端
│   ├── app/
│   │   ├── api/             # REST 路由（documents / notices / tasks / search）
│   │   ├── graph/           # LangGraph 流水线（classify→extract→validate→dedup→todo）
│   │   ├── parsers/         # 多模态解析（text / image / pdf → 统一 ParsedDoc）
│   │   ├── providers/       # VLM / OCR / Embedding 抽象与多实现
│   │   ├── services/        # 时间归一化、规则抽取、向量库
│   │   ├── config.py        # 全部可通过环境变量切换
│   │   ├── models.py        # ORM 模型与常量
│   │   ├── schemas.py       # Pydantic 契约
│   │   ├── db.py            # 引擎 / Session
│   │   ├── main.py          # 应用入口 + CORS + /health
│   │   └── seed.py          # 5 条演示数据（含 1 条重复）
│   ├── tests/               # 32 个 pytest（时间/流水线/API）
│   ├── requirements.txt
│   ├── Dockerfile
│   └── .env.example
├── frontend/                # React + Vite + TS
│   └── src/
│       ├── api.ts           # 类型化 API 客户端
│       ├── types.ts         # 与后端 schema 对齐的 TS 类型
│       ├── common.tsx       # 标签/颜色/格式化
│       ├── App.tsx          # 侧边栏 + 视图切换
│       └── components/      # Dashboard / Upload / Notices / Tasks / Search
└── docker-compose.yml       # 可选：PostgreSQL + 后端
```

---

## 🚀 快速开始

### 1) 后端

需要 Python 3.11+。建议用虚拟环境：

```bash
cd campus-assistant/backend

# 创建并激活虚拟环境（任选其一）
python -m venv .venv && source .venv/bin/activate        # macOS/Linux
# .venv\Scripts\activate                                     # Windows

# 安装依赖
pip install -r requirements.txt

# （可选）复制并修改配置
cp .env.example .env

# 启动（默认 http://127.0.0.1:8000）
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

启动后访问：
- API 文档：`http://localhost:8000/docs`
- 健康检查：`http://localhost:8000/health`

> 若 8000 被占用，换端口（如 `--port 8001`），并同步修改 `frontend/vite.config.ts` 里的代理 `target`。

#### 灌入演示数据（可选）

```bash
python -m app.seed
```

会写入 5 条样例通知（4 类 + 1 条重复），可在前端「概览 / 通知复核」直接看到效果。

### 2) 前端

需要 Node 18+。

```bash
cd campus-assistant/frontend
npm install
npm run dev          # 开发服务器 http://localhost:5173
```

前端开发服务器已将 `/api` 与 `/health` 反向代理到 `http://localhost:8000`，直接打开 `http://localhost:5173` 即可使用。

生产构建：

```bash
npm run build        # 产物在 dist/
npm run preview      # 本地预览构建产物
```

> 若后端不在 8000，修改 `frontend/vite.config.ts` 的 `proxy.target`，或构建后用任意静态服务器 + 反向代理把 `/api` 指到后端。

---

## ⚙️ 配置（`.env`）

全部可经环境变量覆盖，关键项：

| 变量 | 默认 | 说明 |
|---|---|---|
| `DATABASE_URL` | SQLite | 生产改 `postgresql+psycopg://...` |
| `VLM_PROVIDER` | `mock` | `dashscope` 走阿里云百炼 |
| `DASHSCOPE_API_KEY` | 空 | 填了才启用真实向量/VLM |
| `VLM_MODEL` | `qwen3-vl-plus` | 可改 `qwen3.5-vl-plus` 等 |
| `OCR_PROVIDER` | `auto` | 自动探测 Paddle/Rapid/VLM/Stub |
| `EMBEDDING_PROVIDER` | `auto` | 有 Key 用 dashscope，否则本地哈希 |
| `DEDUP_THRESHOLD` | `0.90` | 向量相似度阈值 |
| `REVIEW_CONFIDENCE_THRESHOLD` | `0.55` | 低于此置信度进"待复核" |

---

## 🔌 API 一览

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/api/documents/upload` | 上传图片/PDF/文本，跑完整链路 |
| POST | `/api/documents/text` | 直接粘贴文本处理 |
| GET | `/api/notices` | 通知列表（可按类别/待复核过滤） |
| PATCH | `/api/notices/{id}` | 人工复核修正（自动重算待办） |
| GET | `/api/tasks` | 任务列表（看板数据源） |
| POST | `/api/tasks` | 手动新建待办 |
| PATCH | `/api/tasks/{id}` | 更新/流转状态（写审计） |
| POST | `/api/search` | 语义检索历史通知 |
| GET | `/api/tasks/stats` | 仪表盘统计 |
| GET | `/health` | 健康检查与运行时能力 |

完整字段见 `backend/app/schemas.py` 与 `/docs`。

---

## ✅ 测试

```bash
cd campus-assistant/backend
pip install pytest
pytest -q
```

覆盖：中文时间解析（相对/星期/改期/截止）、流水线（分类/抽取/校验/去重/待办）、全链路 API（导入/去重/状态流转/复核/语义检索/校验错误）。当前 **32 passed**。

---

## 🐳 生产部署（可选）

```bash
cd campus-assistant
docker compose up --build
```

会启动 PostgreSQL(pgvector) + 后端（8000）。前端构建后可用 Nginx 等静态服务器托管，并把 `/api` 反向代理到后端。

---

## 📌 已知边界

- 当前为**单机单用户、无鉴权**的 MVP。
- Mock VLM 为规则实现，对排版极规整的正式通知效果最好；复杂海报建议接入真实视觉大模型。
- 生产 PostgreSQL 尚未内置 Alembic 迁移，目前用 `create_all`；规模化前建议补迁移。
