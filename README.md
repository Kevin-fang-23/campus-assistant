# 校园事务智能助手 🎓

[![CI](https://github.com/Kevin-fang-23/campus-assistant/actions/workflows/ci.yml/badge.svg)](https://github.com/Kevin-fang-23/campus-assistant/actions/workflows/ci.yml)

面向本科生的**多模态校园信息处理**工具：把课程通知、活动海报、作业要求、报修材料（图片 / PDF / 文本）丢进来，系统自动完成 **通知识别 → 关键信息抽取 → 待办生成 → 任务追踪**，并用 **BM25 + 向量混合检索** 支撑自然语言问答与溯源，帮你不再错过截止时间。

> **开箱即用：无需任何 API Key。** 默认用内置规则（Mock VLM + 本地哈希向量 + Stub OCR）即可跑通完整链路；接入阿里云百炼 `qwen3-vl-plus` 后抽取与向量质量进一步提升。

---

## 🎯 核心亮点

| 能力 | 量化结果 | 复现方式 |
|---|---|---|
| **BM25 + 向量混合检索** | 混合 MRR@5 **0.983** > 仅 BM25 0.977 > 仅向量 0.949（nDCG@5 0.985） | `python -m eval.run_retrieval_eval --embedding dashscope` |
| **句级片段选择** | 引用片段直接支撑答案；修复"答案句落在 400 字后被硬截断、LLM 只能答未找到" | `pytest tests/test_snippet.py` |
| **LLM 连接复用** | 20 次问答的 TCP 建连数 **21 → 1**（降约 95%） | `outputs/_verify_connection_reuse.py` |
| **三层请求限流** | 分钟级 / 每 IP 日 / 全局日，防止公网演示烧干额度 | `.env` 的 `RATE_LIMIT_*` |
| **测试与评测** | 后端 **210 passed**；抽取评测 40 案例微平均 F1 **0.98** | `pytest -q`、`python -m eval.run_eval` |
| **CI** | 每次推送自动跑后端测试 + 前端类型检查与构建（无需任何密钥） | 见上方 CI 徽章、`.github/workflows/ci.yml` |

---

## ✨ 功能特性

### 信息处理链路
- **多模态导入**：图片（海报/截图）、PDF（课程通知/作业要求）、纯文本，拖拽或粘贴即可。
- **通知识别 + 关键信息抽取**：自动判定类别（课程通知 / 活动海报 / 作业要求 / 报修材料），抽取标题、发布方、地点、课程、截止时间、活动时间、联系人、标签。
- **中文时间语义**：相对日期（"3天后"）、星期（"周五"）、改期 vs 原定（"调整至周五…原定周三…"）精准区分；截止默认补到 23:59。
- **待办生成**：按规则自动拆出"开始动手 / 提交"等任务，带优先级与提醒时间。
- **任务追踪**：看板（待办 / 进行中 / 已完成 / 已归档）状态流转，逾期自动标红，状态变更写入审计时间线。
- **去重**：内容向量相似度 ≥ 0.90 判定为重复通知，自动跳过入库。
- **人工复核**：低置信度结果进"待复核"，可在前端修正，保存后自动重算待办。

### 检索与问答
- **混合检索**：向量（语义）与 BM25（字面精确）双路召回后 RRF 融合。校园场景充斥课程名、房间号、电话、缩写等精确串，向量路对它们不可靠，BM25 一次命中。
- **句级片段选择**：引用与 LLM 上下文按**句子**择优选取，而非按字符硬截断 —— 引用不会是残句，长通知的关键信息也不会被截掉。
- **RAG 问答**：`/api/qa` 基于检索结果生成答案，答案内用 `[1] [2]` 标注来源，并附可溯源的引用片段。
- **降级可控**：无 Key / 上游失败时自动切换为抽取式回答（`degraded=true`），前端展示逻辑不变。

### 工程特性
- **连接复用**：LLM / Embedding 共用进程级 HTTP 客户端，复用连接池，避免每请求一次 TLS 握手。
- **请求限流**：三层策略保护成本端点（`/api/qa` 与 `/api/documents/*`）与只读接口。
- **优雅降级**：向量后端缺失 → NumPy 余弦回退；OCR 缺失 → Stub；LangGraph 缺失 → 顺序执行。
- **单端口部署**：构建前端后由后端直接托管 `dist`，公网只需映射一个端口，且页面与 API 同源。

> 技术细节：**LangGraph** 编排识别→抽取→校验→去重→待办 的流水线；**SQLAlchemy 2.0** 双兼容 SQLite / PostgreSQL；**PaddleOCR / RapidOCR** 可插拔。

---

## 🧱 技术栈

| 层 | 选型 |
|---|---|
| 前端 | React 18 + TypeScript + Vite 5 |
| 后端 | Python + FastAPI + SQLAlchemy 2.0 + Pydantic v2 |
| 流程编排 | LangGraph（可选，自动降级） |
| 视觉大模型 | Mock / 阿里云百炼 OpenAI 兼容接口（qwen3-vl-plus 等） |
| 文本问答模型 | 百炼 `qwen-plus`（无需视觉能力，比 VL 便宜） |
| OCR | PaddleOCR / RapidOCR / VLM / Stub（自动探测） |
| 向量检索 | FAISS（缺失时 NumPy 余弦回退）+ 自实现 BM25（零新依赖） |
| 融合排序 | RRF（Reciprocal Rank Fusion），对两路分数尺度不敏感 |
| 数据库 | SQLite（开发）/ PostgreSQL（生产，可选 pgvector） |

---

## 📁 目录结构

```
campus-assistant/
├── backend/                      # FastAPI 后端
│   ├── app/
│   │   ├── api/                  # REST 路由
│   │   │   ├── documents.py      #   导入（上传 / 粘贴文本）
│   │   │   ├── notices.py        #   通知列表 / 详情 / 人工复核
│   │   │   ├── tasks.py          #   任务看板 / 流转 / 统计
│   │   │   ├── search.py         #   混合检索
│   │   │   └── qa.py             #   RAG 问答
│   │   ├── graph/                # LangGraph 流水线（classify→extract→validate→dedup→todo）
│   │   ├── parsers/              # 多模态解析（text / image / pdf → 统一 ParsedDoc）
│   │   ├── providers/            # 抽象与多实现
│   │   │   ├── vlm.py            #   视觉模型（mock / dashscope）
│   │   │   ├── ocr.py            #   OCR（paddle / rapid / vlm / stub）
│   │   │   ├── embedding.py      #   向量（dashscope / local_hash）
│   │   │   └── llm_client.py     #   OpenAI 兼容客户端（共享连接池 + 重试退避）
│   │   ├── services/
│   │   │   ├── bm25.py           #   BM25 索引（字符 bigram 分词）
│   │   │   ├── hybrid.py         #   双路召回 + RRF 融合 + 索引写入入口
│   │   │   ├── snippet.py        #   句级片段选择（纯函数）
│   │   │   ├── notice_text.py    #   通知文本来源与片段（问答/检索共用）
│   │   │   ├── rate_limit.py     #   三层限流
│   │   │   ├── vector_store.py   #   FAISS / NumPy 向量库
│   │   │   ├── rule_extract.py   #   规则抽取兜底
│   │   │   ├── datetime_utils.py #   中文时间归一化
│   │   │   └── ingest.py         #   入库编排
│   │   ├── middleware.py         # 限流中间件
│   │   ├── config.py             # 全部可通过环境变量覆盖
│   │   ├── models.py             # ORM 模型与常量
│   │   ├── schemas.py            # Pydantic 契约
│   │   ├── db.py                 # 引擎 / Session
│   │   ├── main.py               # 应用入口 + CORS + 限流 + /health + 静态托管
│   │   └── seed.py               # 5 条演示数据（4 类 + 1 条近似重复）
│   ├── tests/                    # 10 个测试模块 / 210 条用例
│   ├── eval/                     # 离线评测（抽取质量 + 检索质量 + 基线）
│   ├── requirements.txt          # 全部依赖（含可选 OCR/DB 引擎）
│   ├── requirements-ci.txt       # CI 依赖（核心 + 生产路径，不含 OCR 栈）
│   ├── Dockerfile
│   └── .env.example
├── frontend/                     # React + Vite + TS
│   └── src/
│       ├── api.ts                # 类型化 API 客户端
│       ├── types.ts              # 与后端 schema 对齐的 TS 类型
│       ├── common.tsx            # 标签 / 颜色 / 格式化
│       ├── App.tsx               # 侧边栏 + 视图切换
│       └── components/           # Dashboard / Upload / Notices / Tasks / Search
├── .github/workflows/ci.yml       # CI：后端测试 + 前端类型检查与构建
├── .env.example                   # 环境变量模板（配置从项目根 .env 读取）
├── docker-compose.yml            # 可选：PostgreSQL(pgvector) + 后端
├── start.sh / start.bat          # 一键启动脚本
└── stop.bat                      # 按端口杀进程，停止服务
```

---

## 🚀 快速开始

### 1) 后端

需要 Python 3.11+（本项目在 3.13 上开发与验证）。建议用虚拟环境：

```bash
cd campus-assistant/backend

# 创建并激活虚拟环境（任选其一）
python -m venv .venv && source .venv/bin/activate        # macOS/Linux
# .venv\Scripts\activate                                     # Windows

# 安装依赖
pip install -r requirements.txt

# （可选）复制配置模板并修改
# 注意：配置从「项目根目录」的 .env 读取（同时也兼容 backend/.env）
cp ../.env.example ../.env

# 启动（默认 http://127.0.0.1:8000）
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

启动后访问：
- API 文档：`http://localhost:8000/docs`
- 健康检查：`http://localhost:8000/health`（返回流水线引擎、VLM/OCR/向量后端与是否降级）

> 若 8000 被占用，换端口（如 `--port 8001`），并同步修改 `frontend/vite.config.ts` 里的代理 `target`。

#### 灌入演示数据（可选）

```bash
python -m app.seed
```

会写入 5 条样例通知（4 类 + 1 条近似重复），可在前端「概览 / 通知复核」直接看到效果。

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

> 构建出 `frontend/dist` 后，后端启动时会**自动挂载**它到 `/`，此时单端口（8000）即可访问完整应用，无需再开前端服务。

---

## 🔍 检索与问答实现

这一部分是项目的技术核心，三条链路串联如下：

```
用户提问
   │
   ├─ 向量检索 ── Embedding ── FAISS  ──┐
   │                                    ├─ RRF 融合 ── 排序结果
   └─ BM25 检索 ─ 字符 bigram 索引 ─────┘        │
                                                 ▼
                                    句级片段选择（select_snippet / select_context）
                                                 │
                                    ┌────────────┴────────────┐
                                    ▼                         ▼
                              引用溯源片段              拼成 context 交给 LLM
```

### 混合检索

- **BM25 实现**（`services/bm25.py`）：中文无空格，为避免引入分词器依赖，采用**字符 bigram** 作 term（`"线性代数"` → `线性/性代/代数`；`"027-87659999"` 保留数字串的区分度）。标准 BM25 公式，`k1=1.5`、`b=0.75`。
- **融合策略**（`services/hybrid.py`）：用 **RRF** 而非加权求和 —— BM25 分数与余弦相似度量纲不同，RRF 只看名次，无需做归一化标定。融合结果再映射到 `[0,1]` 供前端展示百分比。
- **索引一致性**：向量与 BM25 由同一入口 `index_notice()` 写入，避免"向量更新了而 BM25 没更新"的静默不一致。
- **BM25 索引正文**：BM25 的价值就是精确串匹配，因此索引完整正文（含电话、房间号、邮箱），而非仅结构化字段。

**权重与依据**：

```bash
# 扫描 0.0~1.0 的权重组合，输出 MRR@5 / Recall / nDCG
python -m eval.run_retrieval_eval --embedding dashscope --sweep
```

默认 `HYBRID_WEIGHT_VECTOR=0.2` / `HYBRID_WEIGHT_BM25=0.8`，由 29 条查询实测确定（详见 `backend/eval/RETRIEVAL_BASELINE.md`）：

| 配置 | Recall@1 | MRR@5 | nDCG@5 |
|---|---|---|---|
| 仅向量 | 0.931 | 0.949 | 0.962 |
| 仅 BM25 | 0.966 | 0.977 | 0.978 |
| **混合 0.2:0.8** | **0.966** | **0.983** | **0.985** |

⚠️ 语料仅 16 条，权重曲线在小样本上呈 W 形（极差 3.4%）。**换语料后请重跑 `--sweep` 重新定标**，不要沿用当前值。

### 句级片段选择

原先两处都用字符硬截断：`text[:120]`（引用片段）与 `text[:400]`（LLM 上下文）。这带来两个问题：

1. 引用可能从句子中间切断 → 残句；
2. **更严重**：海报体通知的关键信息（截止时间/地点/联系方式）常落在 400 字之后，被截掉后 LLM 看不到，只会回答"未找到"，而库里其实有答案。

现在改为按句择优（`services/snippet.py`，纯函数、无 I/O）：

- **打分**：`(2.0×bigram覆盖率 + 0.6×unigram覆盖率)×10 + 意图奖励`；
- 用**覆盖率而非绝对计数**，否则长句靠堆词获胜；
- 意图奖励区分**具体值**（`23:00`、`图书馆`、电话）与**泛指词**（"延长开放时间"）两档；
- 输出取原文**字符区间**，保留海报体的换行排版。

### 连接复用

LLM 与 Embedding 共用进程级客户端（`providers/llm_client.py`），provider 按需解析客户端引用而非长期持有。实测 20 次 `/api/qa` 的 TCP 建连数 **21 → 1**（含 query embedding 在内的 40 次 HTTP 请求复用同一条连接）。

---

## ⚙️ 配置（`.env`）

全部可经环境变量覆盖，关键项：

### 基础

| 变量 | 默认 | 说明 |
|---|---|---|
| `DATABASE_URL` | SQLite | 生产改 `postgresql+psycopg://...` |
| `STORAGE_DIR` | `backend/storage` | 上传文件落盘目录 |
| `CORS_ORIGINS` | `localhost:5173` 等 | 允许的前端来源 |

### 模型与向量

| 变量 | 默认 | 说明 |
|---|---|---|
| `VLM_PROVIDER` | `mock` | `dashscope` 走阿里云百炼 |
| `DASHSCOPE_API_KEY` | 空 | 填了才启用真实 VLM / 向量 / 问答 |
| `VLM_MODEL` | `qwen3-vl-plus` | 可改 `qwen3.5-vl-plus` 等 |
| `QA_PROVIDER` | `auto` | `mock` 强制走抽取式降级路径 |
| `QA_MODEL` | `qwen-plus` | 纯文本问答模型 |
| `OCR_PROVIDER` | `auto` | 自动探测 Paddle / Rapid / VLM / Stub |
| `EMBEDDING_PROVIDER` | `auto` | 有 Key 用 dashscope，否则本地哈希 |
| `DEDUP_THRESHOLD` | `0.90` | 向量相似度阈值 |
| `REVIEW_CONFIDENCE_THRESHOLD` | `0.55` | 低于此置信度进"待复核" |

### 混合检索

| 变量 | 默认 | 说明 |
|---|---|---|
| `HYBRID_ENABLED` | `true` | 关掉即回退纯向量检索 |
| `HYBRID_WEIGHT_VECTOR` | `0.2` | 向量路权重（见上文定标依据） |
| `HYBRID_WEIGHT_BM25` | `0.8` | BM25 路权重 |
| `HYBRID_RRF_K` | `60` | RRF 平滑常数，取自原论文 |
| `HYBRID_FETCH_K` | `20` | 每路候选池大小（须 > `top_k`，否则失去融合意义） |
| `BM25_K1` / `BM25_B` | `1.5` / `0.75` | BM25 词频饱和与长度归一化参数 |

### 请求限流

成本端点每次请求都会消耗额度，公网演示时需护栏。三层任一超限即返回 429：L1 每 IP 每分钟（令牌桶，拦瞬时洪峰）、L2 每 IP 每日（拦单源慢刷）、L3 全局每日（总支出上限）。数值为 `0` 表示该层不启用。

| 变量 | 默认 | 说明 |
|---|---|---|
| `RATE_LIMIT_ENABLED` | `true` | 总开关 |
| `RATE_LIMIT_TRUST_PROXY` | `false` | **置 ngrok / Nginx 之后必须设为 `true`**，否则所有访客被视为同一来源；直连公网时必须保持 `false`，否则可伪造 `X-Forwarded-For` 绕过 |
| `RATE_LIMIT_QA_PER_MIN` / `_PER_IP_DAY` / `_PER_DAY` | `6` / `60` / `300` | `/api/qa`（= 1 次 LLM + 1 次 embedding） |
| `RATE_LIMIT_INGEST_PER_MIN` / `_PER_IP_DAY` / `_PER_DAY` | `10` / `40` / `200` | `/api/documents/*`（= VLM + OCR 解析） |
| `RATE_LIMIT_DEFAULT_PER_MIN` | `120` | 其余只读接口 |

> 限流为**单进程内存**实现。多 worker 或多实例部署时计数不共享，需换 Redis 计数或改在网关层限流。

---

## 🔌 API 一览

### 导入与通知

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/api/documents/upload` | 上传图片/PDF/文本，跑完整链路 |
| POST | `/api/documents/text` | 直接粘贴文本处理 |
| GET | `/api/documents` | 文档列表 |
| GET | `/api/documents/{doc_id}` | 文档详情（含原文） |
| GET | `/api/notices` | 通知列表（可按类别/待复核过滤） |
| GET | `/api/notices/{notice_id}` | 通知详情 |
| PATCH | `/api/notices/{notice_id}` | 人工复核修正（自动重算待办） |
| POST | `/api/notices/{notice_id}/regenerate-tasks` | 重算待办 |

### 任务

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/api/tasks` | 任务列表（看板数据源） |
| GET | `/api/tasks/upcoming` | 未来 N 天待办（提醒用） |
| GET | `/api/tasks/stats` | 仪表盘统计 |
| GET | `/api/tasks/{task_id}` | 任务详情（含状态流转时间线） |
| POST | `/api/tasks` | 手动新建待办 |
| PATCH | `/api/tasks/{task_id}` | 更新 / 流转状态（写审计） |
| DELETE | `/api/tasks/{task_id}` | 删除任务 |

### 检索与问答

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/api/search` | 混合检索；返回 `snippet`（句级片段）、`match`（命中来源 both/vector/bm25）与两路分数 |
| POST | `/api/qa` | RAG 问答；返回答案（含 `[n]` 引用标记）、`citations`、`degraded` |

### 元信息

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/health` | 健康检查与运行时能力（引擎 / 模型 / 向量后端 / 是否降级） |
| GET | `/docs` | 交互式 API 文档（OpenAPI） |

完整字段见 `backend/app/schemas.py` 与 `/docs`。

---

## ✅ 测试与评测

### 单元与集成测试

```bash
cd campus-assistant/backend
pytest -q
```

当前 **210 passed**（10 个测试模块 + `conftest.py`）。

> **测试完全不需要 API Key，也不访问外网**：`conftest.py` 已把 VLM / QA 固定为 mock、
> OCR 固定为 stub，embedding 在无 Key 时自动回退本地哈希。因此可直接在 CI 中运行。

覆盖范围：

| 测试文件 | 关注点 |
|---|---|
| `test_datetime_utils.py` | 中文时间解析（相对 / 星期 / 改期 / 截止） |
| `test_pipeline.py` | 流水线（分类 / 抽取 / 校验 / 去重 / 待办） |
| `test_api.py` | 全链路 API（导入 / 去重 / 流转 / 复核 / 检索 / 校验错误） |
| `test_snippet.py` | 句级选片：中文分句边界、打分、`/api/qa` 与 `/api/search` 端到端 |
| `test_hybrid.py` | BM25 索引与融合：tokenization、RRF 边界、权重归一化、正文精确串召回 |
| `test_llm_client.py` | LLM 客户端：重试退避、错误分级、响应格式校验 |
| `test_llm_client_reuse.py` | 连接复用（统计真实 TCP 连接数）、provider 引用自愈、**无 Key 时不阻断启动** |
| `test_rate_limit.py` | 三层限流的边界与并发行为 |
| `test_qa_degradation.py` | 问答降级路径 |
| `test_embedding_backend_truth.py` | 配置后端 vs 实际生效后端的一致性 |

### 离线评测

```bash
# 抽取质量（40 条案例：分类准确率 + 6 字段 P/R/F1）
python -m eval.run_eval

# 检索质量（16 条语料 / 29 条查询：Recall@k / MRR@5 / nDCG@5，--sweep 做权重扫描）
python -m eval.run_retrieval_eval --embedding dashscope --sweep
```

基线快照与结论见 `backend/eval/` 下的 `BASELINE_*.md` 与 `RETRIEVAL_BASELINE.md`。

### 前端类型检查

```bash
cd campus-assistant/frontend
npx tsc --noEmit
```

### 持续集成（CI）

`.github/workflows/ci.yml` 在每次 push 到 `main` 与每个 PR 上跑两个 job：

| Job | 内容 |
|---|---|
| `backend` | Python 3.13 + `requirements-ci.txt` → `pytest -q` |
| `frontend` | Node 20 + `npm ci` → `npx tsc --noEmit` → `npm run build` |

设计取舍：

- **不注入任何密钥**。测试本就应能在无密钥、无外网的环境下跑通（`conftest.py` 已固定 mock/stub provider）。
  若为了让 CI 通过而注入密钥，等于承认测试依赖外部服务，是工程上的退步。
- **`requirements-ci.txt` 只排除 OCR 引擎栈**（paddleocr / paddlepaddle / rapidocr）。
  它们在 Linux 上体积大、下载慢，而 `conftest.py` 已把 OCR 固定为 stub，测试不会触达。
  反之 **faiss / langgraph 保留**：它们是生产配置下真正生效的路径，只测兜底路径会漏缺陷。
- 未 pin 版本（沿用 `requirements.txt` 的 `>=` 风格）。上游发布不兼容大版本时 CI 可能转红，
  届时再按需收紧约束。

---

## 🐳 生产部署（可选）

### Docker Compose

```bash
cd campus-assistant
docker compose up --build
```

会启动 PostgreSQL(pgvector) + 后端（8000）。前端构建后可选择：

1. **由后端托管**（推荐，单端口）：把 `frontend/dist` 放到约定位置，后端启动时自动挂载到 `/`；
2. **Nginx 托管**：静态文件走 Nginx，并把 `/api` 反向代理到后端。

### 限流与反向代理

若部署在 ngrok / Nginx 之后，请设置 `RATE_LIMIT_TRUST_PROXY=true`，否则限流会把所有访客当成同一来源；反之，直连公网时**必须**保持 `false`。

---

## 📌 已知边界

- 当前为**单机单用户、无鉴权**的 MVP；公网演示依赖限流作为唯一护栏，不建议存放敏感数据。
- **限流为单进程内存实现**，多 worker / 多实例下计数不共享。
- 检索评测语料仅 **16 条**，权重结论属小样本；语料扩大后需重跑 `--sweep` 重新定标。
- Mock VLM 为规则实现，对排版规整的正式通知效果最好；复杂海报建议接入真实视觉大模型。
- 生产 PostgreSQL 尚未内置 Alembic 迁移，目前用 `create_all`；规模化前建议补迁移。
- 前端暂无自动化测试，渲染改动依赖人工验证 + 端到端截图。
