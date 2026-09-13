# 代码审查报告 — campus-assistant

- **审查对象**：`main` @ `f991346`（含本轮修复）
- **审查日期**：2026-09-13
- **测试基线**：328 passed（Python 3.13 / pytest 9.1.1）
- **CI**：run#7 @ `f991346` → success（后端测试 + 检索质量门禁 + 前端 tsc/test/build 全部通过）
- **静态检查**：ruff 0.16.7，`ruff check app` 的 **F/E9 级别问题为零**
- **审查方式**：逐模块阅读 + 实机复现验证（所有「潜在问题」均标注是否已实测复现）

---

## 一、本轮已修复（2 项，均已推送）

| 编号 | 问题 | 严重度 | 提交 |
|---|---|---|---|
| P0-1 | `rate_limit.py::_release()` 异常分支调用未定义的 `logger` | 高（潜在 500） | `eac49da` |
| P0-2 | `/api/qa` 缓存无失效机制，新通知入库后返回过期否定答案 | 高（演示致命） | `f991346` |

### P0-1：`_release()` 的 except 分支引用未定义的 `logger`

`services/rate_limit.py` 全文未 `import logging`、未定义 `logger`，但
`_release()` 的 `except` 分支调用 `logger.warning(...)`。触发路径是：
L3 全局额度占位成功后，L2 每 IP 额度判定失败 → 回退 L3 占位 → 若回退抛异常
→ 本应返回 429，实际抛 `NameError` → **500**。

**它为什么逃过了测试**：`DailyCounter.release()`（`rate_limit_store.py:614`）
内部自带 `try/except` 吞掉存储异常，异常根本传不到外层 `_release`。
该 `except` 分支实为**休眠代码**——第一次写的回归测试（走完整 `check()` 路径）
在**有缺陷的代码上照样通过**，因为压根没执行到那一行。

**修复与验证**：
- 补 `import logging` + `logger = logging.getLogger(__name__)`；
- 回归测试改为**分层单测**：直接 monkeypatch `limiter._counter.release` 抛异常，
  驱动 `_release()` 本体；
- **反证**：临时撤掉修复 → 测试复现 `NameError: name 'logger' is not defined`。

> **模式性教训**：项目里 `except Exception: logger.warning(...)` + 内部已吞异常的
> 双层容错，会让外层 except 成为永不执行的死代码。此类分支**只能靠分层单测覆盖**，
> 端到端测试给不了任何保证。

### P0-2：缓存无失效机制（实测复现）

缓存存的是**完整 `AnswerOut`**，包含「知识库中暂时没有与该问题相关的通知」
这类**否定答案**。修复前，知识库变更不会触碰缓存：

| 步骤 | 实测结果 |
|---|---|
| 1. 问「旧手机拆解工作坊」（库中无此通知） | `answer="知识库中暂时没有…"`, `citations=0` |
| 2. 入库一条恰好能回答它的通知 | HTTP 200 |
| 3. 立刻再问同一问题 | **仍返回否定答案**，`cache_hit=True` |
| 4. 手动清缓存后再问 | 正确回答，`citations=1` |

演示现场的表现是「刚上传完通知，招聘官一问却说没有」——等同于功能坏掉。

**修复**：
- `TTLRUCache.invalidate_all()`：清条目但**保留 hits/misses 统计**
  （不用 `clear()`，那会连命中率指标一起抹掉，而失效只是正常业务动作）；
- 新增 `services/qa_cache.py::invalidate_qa_cache(reason)`，挂在三处：
  - `services/ingest.py` 新通知入库（**放在 commit 之后**——若先失效再提交而提交失败，
    会白白清空缓存，虽不致错但无意义；反序最坏只是多清几个 key）；
  - `services/ingest.py::regenerate_tasks()` 通知人工修正；
  - `api/notices.py::update_notice()` 的 `regenerate=false` 分支
    （这条单独 commit，最容易漏）。

**为什么不按 query 精细失效**：key 是 `(query, top_k)`，一次入库影响哪些 query
在写入时无从得知，需要维护「通知 → 引用它的 query」反向索引，成本远高于收益。
缓存上限 128 条，全量失效只值几个 key 的重算。

**验证**：新增 3 条用例，全量 325 → **328 passed**；
反证（把 `invalidate_qa_cache` 改成 no-op）→ 2 条用例精准失败（`assert 1 == 0`）。

---

## 二、可改进之处

### 2.1 缓存命中响应绕过了 `response_model` 校验（已验证安全，但有结构风险）

`QaCacheMiddleware` 命中时用 `JSONResponse(content=hit.model_dump(mode="json"))`
直接返回，**不经过 FastAPI 的 `response_model` 序列化**。当前实测结论：

```
r1 keys: sorted → 与 r2 完全一致
除 cache_hit 外内容完全一致: True
字段类型逐项比对: 无 mismatch
CORS allow-origin on hit: http://localhost:5173   ← 正常
```

**但这是「当前恰好一致」而非「结构上保证一致」**：一旦 `AnswerOut` 加入
`datetime` / `Enum` / 自定义类型且忘记 `mode="json"` 的兼容处理，命中路径与
未命中路径就会返回不同格式，且**没有任何测试会失败**。

**建议**：给中间件命中路径加一条契约测试，断言命中 JSON 能通过
`AnswerOut.model_validate()` 往返且字段集与未命中路径完全一致。
成本约 15 行，收益是把这个假设变成护栏。

### 2.2 `/health` 未暴露缓存与限流的运行时状态

`qa_cache.stats()` 与 `rate_limit.used_today()` 都已实现但**没有任何出口**。
`stats()` 的 docstring 写着「生产可挂到 /health」，实际没挂。

`/health` 已经上报了 `vector` / `vlm` / `ocr` 的运行时能力，
`cache` / `rate_limit` 属于同类信息，且对演示排障价值很高
（「额度还剩多少」「缓存命中率多少」）。建议补上，例如：

```json
"cache": {"enabled": true, "size": 3, "hit_rate": 0.42, "ttl_seconds": 300},
"rate_limit": {"store": "sqlite", "qa_used_today": 17, "qa_per_day": 300}
```

### 2.3 `app/` 下 69 处 `# noqa: BLE001` / `ANN001` 是装饰性注释

`ruff check app` 报 69 条 `RUF100 unused-noqa`——因为项目**没有配置 ruff 规则集**，
`BLE001`（盲捕异常）、`ANN`（类型注解）等规则根本未启用，这些 `noqa` 全无作用。

**建议**：在 `backend/pyproject.toml` 或 `ruff.toml` 里显式声明
`select = ["E","F","W","I","UP","B","SIM","BLE","ANN","RUF"]`，
让这些 `noqa` 真正生效，并把 `RUF100` 纳入 CI——无效 `noqa` 会让后来者误以为
某处异常已被审查过。或者反过来，若确定不启用这些规则，就删掉这些注释。
**现状是最差的**：两边的意图都没实现。

### 2.4 前端未消费 `cache_hit`

`cache_hit` 已在 `AnswerOut` 中就位，但 `frontend/src` 全域搜索**无任何引用**。

演示场景里这是一个很好的「技术可见化」素材：回答上方显示
「本次命中缓存（未调用模型）」或「实时生成」，能直观展示 P1 优化的价值。
对招聘官而言，比在 README 里写「加了缓存」有说服力得多。

### 2.5 `tests/` 有 5 处真实的未使用变量/导入

```
test_embedding_backend_truth.py:140  F401  Session as _Session 未使用
test_hybrid.py:10                    F401  math 未使用
test_llm_client.py:15                F401  RateLimitError 未使用
test_qa_cache.py:262                 F841  calls 赋值后未使用
test_qa_cache.py:488                 F401  CitationOut 未使用
```

其中 `test_qa_cache.py:262` 的 `calls = _install(...)` 值得留意：该用例
`test_no_recall_response_is_also_cached` 的本意是验证「空召回也缓存」，但
`calls` 未使用意味着它**没有断言 LLM 调用次数**——如果实现改成空召回不写缓存，
这条用例仍会通过。建议补上 `assert calls["count"] == 1`。

---

## 三、潜在问题（按优先级）

### 3.1 `_release()` 的 except 分支是死代码（结构性隐患，未修复）

同 P0-1 的根因：`DailyCounter.release()` 已经吞掉所有异常，外层
`RateLimiter._release()` 的 `except` **永远不会执行**。修 `logger` 只是
**堵住了 500**，没有消除这层无意义的重复容错。

两种收敛方式，二选一：
- **删掉外层 try/except**，让 `DailyCounter.release()` 承担唯一容错职责
  （它已经做了，且有 logger 和 warning）；
- 或**去掉内层的吞异常**，让异常向上传播，由外层统一处理与记录。

现状是两层都想兜底，结果外层沦为永不执行的代码——而**永不执行的代码意味着
下一次改动时没人会发现它已经坏了**（P0-1 就是这么来的）。

### 3.2 embedding 降级会抹掉历史向量，且无自动恢复（可用性风险）

`providers/embedding.py::_degrade()` 是**粘性**的：DashScope 一旦失败，
本进程内全部改用 local_hash（维度 256，原 1024）。

问题在于启动时 `main.py` 的自动 reindex 逻辑：

```python
if store.skipped_mismatched and settings.reindex_on_dim_mismatch:
    store.reindex(db)   # 用「当前 embedder」重算全部向量
```

若某个 worker 在启动时恰好 dashscope 不可用 → 降级到 local_hash →
`skipped_mismatched > 0` → **用 256 维的 local_hash 重算并覆盖库里的全部 1024 维向量**。
等 dashscope 恢复、进程重启后，库里的向量已经全是 256 维，与 1024 维的查询向量不可比 →
**检索静默返回 0 条**，直到再触发一次 reindex。

`VectorStore` 已记录 `provider` 列用于「自证向量来源」，但 reindex 并没有
基于它做**保护性判断**（例如「当前后端是降级态时，拒绝覆盖历史向量」）。

**影响**：演示现场若上游抖一下，检索会整体失效且无报错。
**建议**：`reindex` 增加前置条件——`embedder.degraded is True` 时**拒绝执行**
（或降级为「只告警不重建」），因为降级态不是「后端切换」，而是「暂时不可用」。

### 3.3 缓存是进程级单例，多 worker 命中率会被稀释

`qa_cache` 单例是每进程一份。若用 `uvicorn --workers 4`，
同一 query 落到不同 worker 就是 4 次独立计算，命中率约为单 worker 的 1/4。

单机单进程演示下无影响（`docs` 与 `rate_limit_store.py` 已说明多实例才需 Redis），
但**这一点在任何地方都没有记录**——`rate_limit_store.py` 详细写了多 worker 语义，
`qa_cache.py` 的「为什么不用 Redis」一节只讲了「演示部署没有 Redis」，
**没有讲清多 worker 的命中率后果**。建议补一句，避免将来扩 worker 时误判。

### 3.4 TTL 内的答案对「同一通知被更新」也不会刷新（已随本轮修复大幅缓解）

本轮修复覆盖了**入库 / 修正 / 删除**三类写入。剩下的窗口是：
缓存的 5 分钟内，如果**通知被别的路径改动而没走这三条路径**，答案仍会过期。
目前看写入路径只有这三条，风险已很低——但这是「靠枚举完整性成立」的保证，
不是结构性保证。若将来新增批量导入、后台任务更新等路径，**需要同步挂失效**。

### 3.5 `documents.py` 的 `logger.exception` + 返回 500 会把内部异常暴露给前端

```python
except Exception as exc:
    db.rollback()
    logger.exception("处理上传文件失败")
    raise HTTPException(status_code=500, detail=f"处理失败：{exc}")
```

`detail` 直接拼接原始异常文本。演示环境（ngrok 公开）会把栈内信息
（可能含文件路径、SQL 片段）返回给任意访问者。

**建议**：对外返回固定文案（或脱敏后的类别），完整异常只进日志。
至少把 `{exc}` 换成「请联系管理员」或错误码。

### 3.6 `search_min_bigram_overlap` 的「零误杀」结论建立在 2 条噪声样本上

`config.py` 已如实标注了这个局限（「⚠️ 该结论建立在**仅 2 条噪声查询**之上」），
这是很好的自我披露。仅作提醒：这个数字在简历/README 里**不宜单独强调**
（「零误杀」听起来很强，实际样本极小），若被追问需要能说清样本量。

### 3.7 `C:\Users\86182` 被误 `git init` 成空仓库（环境问题，非代码问题）

家目录下存在 `.git`（`No commits yet on master`、无 remote、无提交），
应为此前在错误目录执行 `git init` 所致。

**风险**：将来若在该目录执行 `git add .`，会把 `AppData` / `NTUSER.DAT` /
`anaconda3` 等全部纳入版本控制。**建议删除该 `.git` 目录**（它没有任何提交，
删除无损失）。

---

## 四、后续优化方向

按「投入产出比」排序：

### 4.1 缓存命中率指标 + 前端可见化（低成本，高演示价值）

- `/health` 暴露 `stats()`（见 2.2）；
- 前端把 `cache_hit` 渲染成一个小标签（见 2.4）。

两项加起来工作量很小，但能把 P1 优化的价值**从代码变成可演示的效果**。

### 4.2 缓存键升级为「语义相似」而非「字符串归一化」

当前 key 是 `(" ".join(query.split()).lower(), top_k)`。「作业什么时候截止」
与「作业截止时间是什么」是**不同 key**，会各烧一次 LLM。

升级路径：对 query 做 embedding → 与缓存里已存 query 向量比对 →
余弦相似度 > 阈值（如 0.95）则命中。

**成本**：多一次 embedding 计算，但换来「措辞不同的同一问题」也能命中。
演示现场招聘官不会问出完全一样的字面，这个改进的**实际命中率提升会远大于
当前的字面归一化**。

**注意**：这会引入「相似但不同的问题被误判为同一问题」的风险，
阈值需要用小规模实测定标（与 `hybrid_weight_vector` 同样的方法论）。

### 4.3 检索质量的持续回归（已有基础，建议扩样本）

`eval/run_retrieval_eval.py --gate` 已在 CI 里做质量门禁，这是项目里
**工程成熟度最高的一环**。唯一短板是语料规模（55 条 / 129 查询）。
建议优先扩「真实无答案查询」样本，把 `search_min_bigram_overlap`
的阈值判定从「2 条噪声样本」升级为可信结论。

### 4.4 把 `rate_limit` / `qa_cache` 的重复容错模式收敛掉

见 3.1。建议在收敛时顺手统一「存储故障时的取舍方向」——
目前 `rate_limit_store.py` 的写法是「护栏故障时放行」（可用性优先），
这个取向是对的，值得写成模块级约定而不是散落在各处。

### 4.5 引入 ruff 配置 + CI 静态检查步骤（低成本）

装上 `ruff.toml` 并在 CI 加一步 `ruff check app`（仅 `E,F,W` 起步，
避免一次引入上百条告警）。这能在**编码阶段**拦住 P0-1 那类缺陷
（未定义符号属于 `F821`）。

---

## 五、总评

**强项**（这些是真实优势，建议在面试中作为谈资）：

1. **诊断深度**：`rate_limit_store.py` 明确记录了两版被推翻的设计及其实测数据
   （「4 worker × 额度 40 → 放行 60 次」「4 子进程并发 → 偶发放行 41 次」），
   并说明为什么最终选择原子占位。这种「记录失败路径」的工程习惯很少见。
2. **自我披露**：多处 docstring 主动标注结论的样本量局限与已知缺陷
   （如 bigram 阈值的 2 条噪声样本、L1 令牌桶的多 worker 局限），
   没有为了好看而夸大。
3. **可回滚性**：`hybrid_enabled` / `qa_cache_enabled` / `rate_limit_store`
   都留了「一键回退到改造前行为」的开关，改造风险控制得好。
4. **CI 质量门禁**：不只有单测，还有检索指标的回归门禁，且说明了
   「单测覆盖结构、门禁覆盖质量」的互补关系。

**短板**（本轮已修两个，其余按优先级）：

1. ~~未定义符号类缺陷无静态检查兜底~~ → ruff 可解（4.5）；
2. ~~缓存无失效~~ → 已修；
3. 双层容错导致死代码（3.1）——**这是最值得优先收敛的结构问题**；
4. 降级态覆盖历史向量的风险（3.2）——潜在可用性事故；
5. 观测能力有实现无出口（2.2 / 2.4）——低成本高收益的空白。
