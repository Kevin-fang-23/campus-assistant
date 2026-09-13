# 系统架构

> 面向招聘官的 5 分钟导览：本文说明**各层职责、关键数据流与设计取舍**。
> 图以 `docs/images/architecture.png` 为准（矢量源见 `docs/architecture.svg`）。

![系统架构](./images/architecture.png)

---

## 一、分层与职责

系统按「**接入 → 编排 → 能力 → 存储**」四段式分层，外加贯穿全局的**降级护栏**。
分层的目的不是好看，而是让每层只依赖下一层的抽象（例如业务层不认识 dashscope，
只认识 `providers/` 的接口），从而任何外部依赖坏了都能局部降级。

| 层 | 模块 | 职责 | 关键文件 |
|---|---|---|---|
| ① 前端 | React SPA（5 视图） | 交互与呈现；**句级片段**、**引用溯源高亮**、**命中来源**（语义/关键词/混合）可视化 | `frontend/src/components/*.tsx` |
| ② 接入层 | FastAPI 路由 | 请求校验、路由分发、**三层限流护栏**、统一错误语义（429 带跨域头） | `app/api/*.py`、`app/middleware.py` |
| ③ 编排层 | LangGraph 流水线 | 把「原始文件」变成「结构化通知 + 待办」：`parse → extract → normalize → dedup → persist → tasks` | `app/graph/{pipeline,nodes,state}.py`、`app/parsers/*` |
| ④ 检索层 | 混合检索 | 双路召回 → 加权 RRF → 相关性过滤 → 句级选片 | `app/services/{hybrid,bm25,vector_store,snippet}.py` |
| ⑤ 能力层 | Providers | 外部能力统一封装（VLM / OCR / Embedding / LLM），**每项都可降级** | `app/providers/*.py` |
| ⑥ 存储层 | SQLAlchemy 2.0 | 业务数据、向量、限流计数、原文件 | `app/models.py`、`app/db.py` |
| ⑦ 横切 | 降级与护栏 | 任何外部依赖失效都不返回 500；降级态可被 `/health` 观测 | 见下文「三、降级矩阵」 |

### 关键设计取舍（面试常问）

1. **为什么能力层要单独抽一层？**
   因为「无 Key 也能跑通全链路」是硬需求（招聘官不会配 API Key 再体验）。抽层后
   `EMBEDDING_PROVIDER=auto` 才能在无 Key 时落到 `local_hash`、有 Key 时用真实模型，
   而**业务代码完全不变**。代价是多一层间接，收益是部署门槛降到 0。

2. **为什么限流要落库而不是放内存？**
   内存版在 `--workers > 1` 时每个进程各算一份，实际放行量 = 配置 × worker 数，护栏形同虚设。
   落库后 4 个真实子进程抢额度 40，实测**恰好放行 40 次**。
   性能上开启 WAL 后单次记账 **88.8µs**（未开 WAL 为 4071µs，差 46 倍），可接受。
   详见 `app/services/rate_limit_store.py` 模块文档。

3. **为什么向量要持久化到数据库而不是索引文件？**
   索引文件与数据库容易不一致（写了库没写索引 → 检索不到，且难以察觉）。
   现在向量存 `notice_embeddings`（BLOB + provider + dim），**启动时从库重建索引**；
   维度不符的行会被跳过并计数（provider 切换残留），可通过 reindex 恢复。

---

## 二、数据流向

### 写入链路（导入 → 结构化）

```
前端「导入」
  → POST /api/documents/upload | /text
  → 文件级去重（sha256 命中已解析文档 → 直接复用，不重复计费）
  → LangGraph: parse（按 MIME 选 parser）→ extract（VLM 抽取，失败降级规则抽取）
  → normalize（时间语义归一：相对日期 / 星期 / 改期）→ dedup（向量余弦 ≥ 0.90 判重）
  → persist（notices）+ tasks（按规则拆待办）→ 生成 embedding 入库
  → 返回：文档信息 + 抽取字段 + 置信度 + 待办列表 + pipeline trace
```

### 读取链路（检索 / 问答）

```
前端「知识库问答」
  → POST /api/search | /api/qa
  → 双路召回：向量（FAISS IndexFlatIP，1024 维）+ BM25（字符 bigram）
  → 加权 RRF 融合（权重 0.1 : 0.9，只用名次，免疫两路分数尺度差异）
  → 相关性过滤（阈值判定）→ filtered 标志，区分「不相关」与「库内确实没有」
  → 句级选片（按查询挑出最相关句子，供引用与 LLM 上下文）
  → [/api/qa] LLM 生成答案，用 [n] 标注来源；失败则降级为抽取式回答
  → 返回：命中/答案 + 命中来源（语义/关键词/混合）+ 可溯源引用片段
```

---

## 三、降级矩阵

系统设计原则：**可用性优先于精度**。任一项外部能力失效都不返回 500，而是降级 + 告警，
并把降级态暴露在 `/health`，避免"悄悄变差"。

| 失效项 | 降级行为 | 可观测方式 |
|---|---|---|
| VLM 调用失败 / 无 Key | 规则抽取（正则 + 中文时间解析） | 抽取 `confidence` 下降；`/health` 的 `vlm.mock` |
| OCR 未安装 | Stub（不阻断，VLM 仍可处理图片） | `/health` 的 `ocr` 字段 |
| LLM 生成失败 / 无 Key | 抽取式回答（直接给最相关片段原文） | 响应 `degraded=true`；前端显示「抽取式」标记 |
| Embedding 上游失败 | `local_hash` 本地哈希向量 | `/health` 的 `embedding_degraded` |
| FAISS 不可用 | NumPy 余弦相似度（**结果等价**，仅速度差异） | `/health` 的 `vector.backend` |
| LangGraph 不可用 | 顺序执行（逻辑等价） | `/health` 的 `pipeline_engine` |
| 限流存储读写异常 | 本次放行 + WARNING 日志 | 日志 |

---

## 四、目录结构（按职责）

```
backend/app/
├── api/              # ② 接入层：路由（documents / notices / tasks / search / qa）
├── middleware.py     #   ↑ 限流中间件（置于 CORS 内层）
├── graph/            # ③ 编排层：LangGraph 流水线（pipeline / nodes / state）
├── parsers/          #   ↑ 多模态解析（image / pdf / text）
├── services/         # ④ 检索与领域服务
│   ├── hybrid.py     #     加权 RRF 融合 + 相关性过滤
│   ├── bm25.py       #     自实现 BM25（零新依赖）
│   ├── vector_store.py #   FAISS 封装（numpy 兜底、从库重建）
│   ├── snippet.py    #     句级选片
│   ├── ingest.py     #     入库编排 + sha256 去重 + 落盘
│   ├── rule_extract.py #   规则抽取（VLM 降级路径）
│   ├── datetime_utils.py # 中文时间语义
│   ├── rate_limit.py #     三层限流策略
│   └── rate_limit_store.py # 限流计数存储（SQLite / 内存 / 文件）
├── providers/        # ⑤ 能力层：VLM / OCR / Embedding / LLM 客户端
├── models.py         # ⑥ 存储层：ORM 模型
└── db.py             #   ↑ 引擎与会话

frontend/src/
├── components/       # ① 五个视图组件
├── api.ts            #   请求层（统一错误处理）
└── common.tsx        #   展示辅助（类别色板、时间格式化）
```

---

## 五、架构图的维护

- **生成脚本（真正的源）**：`docs/make_architecture.py` —— 运行 `python docs/make_architecture.py`
  会重新生成 `docs/architecture.svg`，并在系统临时目录输出一个渲染用包装页。
- **矢量产物**：`docs/architecture.svg`（可直接编辑，任意缩放不失真）
- **README 用图**：`docs/images/architecture.png`（1560×1190）

> 为什么用脚本生成而不是手画 SVG：分层盒子的坐标容易算错，且改一处要挪全部。
> 脚本里层高、间距、字号都是参数，改内容时只需改文字，布局自动重排。

图中已尽量只写**稳定抽象**（层、模块、数据流）。模型名、向量维度、RRF 权重这类
易变值标注为**当前默认值**，改配置时记得同步图与本文档。

---

## 六、想进一步了解的三个入口

| 想了解 | 去这里 |
|---|---|
| 检索为什么这样融合、指标如何 | [`backend/eval/RETRIEVAL_BASELINE.md`](../backend/eval/RETRIEVAL_BASELINE.md) |
| 限流持久化的完整设计与被推翻的方案 | `backend/app/services/rate_limit_store.py` 模块 docstring |
| 三步跑起来 / 配置项 | [`README.md`](../README.md) 的「快速开始」与「配置」 |
