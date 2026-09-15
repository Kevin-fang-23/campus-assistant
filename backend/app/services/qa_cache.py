"""/api/qa 响应的进程内缓存。

## 为什么需要

`/api/qa` 每次提问会消耗：
- 1 次 LLM 调用（生成答案；无 Key 时退化为抽取式，但仍要走检索层）
- 1 次 embedding 调用（向量召回；BM25 路径不计费但仍占 CPU）

演示现场一个常见场景：多个招聘官依次打开 demo、问同一个问题
（例如「作业截止时间」、「旧手机拆解工作坊」），会被重复收费。
更糟的是问答结果对同一 query 是稳定的（答案引用同一批通知），
完全没有必要每次都重新生成。

加一层**短 TTL** 的 LRU 缓存：5 分钟内重复提问直接复用上次响应，
**不发起 LLM 与 embedding 调用**，从而：
- 额度：演示现场对同一问题的反复追问不再烧 token；
- 延迟：缓存命中时跳过网络往返，首字节时间从 ~1s 降到 ~1ms；
- 体感：招聘官反复点同一问题时响应稳定。

## 设计取向

1. **内存 LRU + TTL**：项目一贯「零新依赖」，与 `rate_limit_store.py` 的
   `InMemoryCountStore` 同风格。128 条 × 假设 5KB ≈ 640KB，可忽略。

2. **缓存完整 `AnswerOut`**：缓存命中即直接复用 answer / citations /
   backend / degraded。**不重建 citations** —— 重建要重新查 DB + snippet
   生成，等于退化成"半缓存"，失去了意义。代价是新通知入库后
   5 分钟内不会出现在缓存答案里，这是缓存的应有语义。

3. **任何路径都缓存**（LLM 成功 / LLM 降级 / 空召回）：这三类都消耗
   embedding（空召回仍要走 hybrid_search）。即使降级回答也对前端
   有意义（top1 摘要），缓存避免重复 embedding。

4. **key 归一化**：(strip + collapse_whitespace + lower, top_k) —
   `  今天天气   ` 与 `今天天气` 视为同一 key。大小写归一化只对
   英文/数字生效（中文 lowercase 是 noop），属于免费保险。

5. **缓存命中不入限流**：限流是保护上游（LLM/embedding）的护栏，
   缓存命中不消耗上游资源，本就不该挤占额度。FastAPI 同步端点
   跑在线程池，加锁保证线程安全。

## 语义相似匹配（第二阶段）

字面归一化只处理"同一个字符串的格式变体"，无法处理**同一个意图的不同问法**：

    作业什么时候截止  ‖  作业截止时间是什么

这两句归一化后是两个不同的 key，会各走一次完整流程（各烧一次 LLM + embedding），
而它们的答案其实是同一份。演示现场招聘官用不同措辞问同一件事，是必然发生的。

因此在精确 key 之外再加一道**查询向量近邻**判定（`QaSemanticCache.find_similar`）：

    sim(query_vec, cached_vec) >= qa_cache_semantic_threshold  →  判定命中，复用结果

### 四个关键设计取舍

1. **阈值由实测定标，不拍数字**。方法与混合权重调参（`run_retrieval_eval.py
   --sweep / --calibrate`）一致：用标注数据夹出可行区间 —— 改写对的**最低**
   相似度是下界（阈值须 ≤ 它才不漏改写），"相近但异义"对的**最高**相似度是上界
   （阈值须 > 它才不误配）。脚本与数据见 `eval/run_cache_threshold_eval.py`
   + `eval/cache_pairs.json`。取「满足零误配的最低阈值」：漏命中只是多花一次
   LLM，误配是用户拿到**另一个问题的答案**且无从察觉，代价不对称。

2. **阈值随 embedding 后端变化，换后端必须重跑标定**。余弦绝对值依赖向量空间，
   local_hash（256 维字符哈希）与 dashscope text-embedding-v4（1024 维语义向量）
   分布不同，同一个阈值不可通用。`find_similar` 因此带**维度守卫**：只有与查询
   向量同维度的条目才参与比对，避免跨后端向量算出无意义的余弦后"看起来命中了"
   （同源教训见 `VectorStore._append` 的维度守卫）。

3. **精确 key 永远先查**。字面命中零成本（不调 embedding），只有精确 key 未命中
   且语义开关打开时才计算查询向量；算出的向量会**透传给 hybrid_search 复用**
   （见 `api/qa.py`）。因此整条链路不存在"为了探测缓存而多付一次 embedding"：
   命中省一次 LLM，未命中也不比原来多花 embedding。

4. **语义命中仍计入限流**。既有约定"缓存命中不入限流"的理由是命中不消耗上游
   资源，精确命中确实如此；但**语义命中要算一次 embedding**，是真实的上游开销，
   因此它不绕过限流 —— 见 `middleware.py` 中"为何语义路径不在中间件里"的说明。

## 为什么不用 `functools.lru_cache`

| 能力 | `lru_cache` | `TTLRUCache` |
|---|---|---|
| TTL | ✗ | ✓ |
| 按 maxsize（条目数）淘汰 | 弱（按调用次数） | ✓ |
| 运行时清空 | ✗（要重建装饰器） | ✓ |
| 命中率统计 | ✗ | ✓ |
| 单例替换（测试隔离） | ✗ | ✓ |

测试必须能「清空缓存再发请求」，`lru_cache` 不行。

## 为什么不用 Redis

- 演示部署（ngrok + 单机）没有 Redis；
- 单 worker 内存缓存已能解决演示场景的重复提问；
- 多实例部署才需要 Redis（参见 `rate_limit_store.py` 模块说明）；
- 留出 `CountStore`/`Cache` 协议，后续替换实现即可，**不必改 `qa.py`**。
"""
from __future__ import annotations

import logging
import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class _Entry:
    """单条缓存项。dataclass 而非 dict：避免运行期键名拼错。"""

    value: Any
    expire_at: float
    created_at: float = field(default_factory=time.monotonic)


@dataclass(frozen=True)
class CachedAnswer:
    """/api/qa 的缓存条目：答案本体 + 语义匹配所需的元信息。

    为什么把向量**存进条目值**，而不是另开一张 `key -> vector` 的旁路表：
    TTLRUCache 的淘汰与过期都只作用于 `_data` 一个结构，旁路表必然会
    在被淘汰/过期时与它失去同步（"向量还在、条目已没了"，或反之），
    而那种不一致只会在生产上偶发，极难复现。把向量塞进条目里，
    生命周期就只有一处，不需要任何回调来同步。

    `vector is None` 表示写入时未启用语义匹配（或向量计算失败）——
    这类条目只服务精确命中，直到被淘汰。
    """

    answer: Any
    query: str
    top_k: int
    vector: np.ndarray | None = None


@dataclass(frozen=True)
class SimilarHit:
    """语义命中结果。"""

    entry: CachedAnswer
    similarity: float


class TTLRUCache:
    """线程安全的「按 key 写入 + TTL 到期 + LRU 淘汰」缓存。

    锁粒度：每次 `get` / `set` / `clear` / `stats` 各自完整持有锁。
    业务读热点（`ask()` 每次请求最多一次 get + 一次 set），
    持锁时间在 µs 级，不会成为瓶颈 —— 真实 LLM 调用要 500ms~3s，
    锁的开销可忽略。
    """

    def __init__(
        self,
        *,
        max_size: int = 128,
        ttl_seconds: float = 300.0,
        wall_fn: Callable[[], float] = time.monotonic,
    ) -> None:
        if max_size < 1:
            raise ValueError(f"max_size 必须 >= 1，实际 {max_size}")
        if ttl_seconds <= 0:
            raise ValueError(f"ttl_seconds 必须 > 0，实际 {ttl_seconds}")
        self._max_size = int(max_size)
        self._ttl = float(ttl_seconds)
        self._wall = wall_fn
        # OrderedDict 维持访问顺序：最近访问的 key 在末尾，
        # `popitem(last=False)` 弹出最久未用的。
        self._data: OrderedDict[Any, _Entry] = OrderedDict()
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0
        # 语义匹配的**独立**计数：与精确命中分开统计，否则 /health 上看不出
        # "语义层到底救回了多少次改写问法"，也就无法判断这个阈值配得值不值。
        self._semantic_hits = 0
        self._semantic_lookups = 0

    # ---------------- 对外 API ----------------
    def get(self, key: Any) -> Any | None:
        """读 key 的 value；过期或不存在返回 None。

        命中：刷新 expire_at，并把 key 移到 LRU 末尾。
        过期：从表里删除（不复活），并记账为 miss —— 否则
        "过期但还在表里"会让 size 永远等于 max_size，观察不到淘汰。
        """
        with self._lock:
            entry = self._data.get(key)
            if entry is None:
                self._misses += 1
                return None
            now = self._wall()
            if entry.expire_at <= now:
                del self._data[key]
                self._misses += 1
                return None
            # LRU：移到末尾表示「最近用过」
            self._data.move_to_end(key)
            self._hits += 1
            return entry.value

    def set(self, key: Any, value: Any) -> None:
        """写入。已存在则刷新 expire_at；满了则淘汰最久未用的。

        设计取向：写入不传 ttl，每次写入都用实例默认 ttl。
        /api/qa 的缓存语义是「最新一次的回答有效 5 分钟」，
        不是「原始版本固化 5 分钟」 —— 因为新的缓存条目有更新的
        citations（如果用户刚刚入库新通知），应该重置 TTL。
        """
        with self._lock:
            now = self._wall()
            existing = self._data.get(key)
            if existing is not None:
                existing.value = value
                existing.expire_at = now + self._ttl
                self._data.move_to_end(key)
                return
            self._data[key] = _Entry(value=value, expire_at=now + self._ttl)
            while len(self._data) > self._max_size:
                self._data.popitem(last=False)

    def clear(self) -> None:
        """清空全部条目与统计。**仅测试用**。"""
        with self._lock:
            self._data.clear()
            self._hits = 0
            self._misses = 0
            self._semantic_hits = 0
            self._semantic_lookups = 0

    def invalidate_all(self) -> int:
        """失效全部条目，**保留**命中率统计。返回清掉的条目数。

        为什么不能用 `clear()` 做生产失效：它会连 `_hits`/`_misses` 一起清零，
        把「命中率」这个运维指标一起抹掉 —— 而失效是正常业务动作
        （新通知入库），不该污染观测数据。

        为什么需要失效：缓存的是完整 `AnswerOut`（含 citations 与"未找到"这类
        否定答案）。新通知入库后，同一 query 的正确答案已经变了，但缓存仍会
        返回旧答案 —— 实测过的最坏情形是：先问一个库中无答案的问题（缓存了
        "知识库中暂时没有…"），随后入库能回答它的通知，5 分钟内再问依旧拿到
        否定答案，用户会以为系统坏了。因此入库/重建索引后必须主动失效。
        """
        with self._lock:
            n = len(self._data)
            self._data.clear()
            return n

    def stats(self) -> dict[str, Any]:
        """命中率快照。生产可挂到 /health，调试/演示可打印。"""
        with self._lock:
            total = self._hits + self._misses
            return {
                "size": len(self._data),
                "max_size": self._max_size,
                "ttl_seconds": self._ttl,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": round(self._hits / total, 4) if total else 0.0,
                # ---- 语义匹配观测项 ----
                # semantic_lookups：精确 key 未命中后、真正去算过向量的次数；
                # semantic_hits：其中靠"改写问法"救回来的次数。
                # 演示时这两个数字最能说明该功能是否在干活。
                "semantic_lookups": self._semantic_lookups,
                "semantic_hits": self._semantic_hits,
            }

    def __len__(self) -> int:
        with self._lock:
            return len(self._data)


class QaSemanticCache(TTLRUCache):
    """TTLRUCache + 「按查询向量近邻查找同 top_k 条目」的语义命中能力。

    为什么用**子类**而不是改造 TTLRUCache：
    TTLRUCache 是通用的 key→value 结构，它的契约（`set`/`get`/`invalidate_all`
    /`stats`）有专门的测试钉住，且不假设 value 里有什么。语义匹配需要
    "value 是 CachedAnswer 且带向量"这一额外前提，把它塞进基类会让基类的
    契约变得含糊（`get` 到底返回 value 还是 entry？）。子类继承全部行为，
    只新增两个方法，既有调用点与测试都不受影响。

    查找复杂度：线性扫描。max_size 默认 128 条 × 1024 维点积 ≈ 13 万次乘加，
    在 µs 量级 —— 相较于它要省掉的那次 LLM 调用（数百 ms ~ 数秒），
    引入向量索引（FAISS）反而是过度设计。上限一旦调到千级再考虑。
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        # 维度不符只告警一次：切换 embedding 后端后，缓存里会短暂留着旧维度
        # 的向量，每次请求都打一行告警会把日志刷爆，而问题本身在 TTL 到期后自愈。
        self._dim_mismatch_warned = False

    # ---------------- 精确命中 ----------------
    def lookup_exact(self, query: str, top_k: int) -> CachedAnswer | None:
        """按归一化 key 精确查找。返回 None 表示未命中（含已过期）。

        返回**条目**而不仅是 answer：调用方还要打日志看命中的是哪句原问法，
        也便于后续把相似度写进响应头。非 CachedAnswer 的值（测试往同一个
        缓存里塞的裸字符串）一律视为未命中，不做任何猜测。
        """
        value = self.get(cache_key(query, top_k))
        return value if isinstance(value, CachedAnswer) else None

    # ---------------- 语义命中 ----------------
    def put_answer(
        self,
        query: str,
        top_k: int,
        answer: Any,
        *,
        vector: np.ndarray | None = None,
    ) -> None:
        """写入回答。vector 为 None 时该条目不参与后续语义匹配。

        向量在此处**归一化一次**并存储：查询向量与缓存向量都已是单位向量时，
        余弦相似度退化为点积，查找路径上就不必每条都做除法。同时把 dtype
        固定为 float32 —— 与 embedding provider 的输出一致，避免混入
        float64 后点积悄悄升精度、数值在不同平台上出现末位差异。
        """
        stored: np.ndarray | None = None
        if vector is not None:
            vec = np.asarray(vector, dtype=np.float32).ravel()
            norm = float(np.linalg.norm(vec))
            # 零向量（理论上不该出现）不能归一化，直接丢弃：
            # 留着它会让余弦变成 0/0 或恒为 0，是"看起来在工作但永不命中"的坑。
            if norm > 0:
                stored = vec / norm
        # key 由 query + top_k 决定，与精确路径完全一致 ——
        # 两条路径必须共用同一个 key 空间，否则精确命中和语义命中的
        # 「同一问法重复写入」判断会分叉（同一 query 存出两个条目）。
        self.set(
            cache_key(query, top_k),
            CachedAnswer(answer=answer, query=query, top_k=top_k, vector=stored),
        )

    def find_similar(
        self,
        query: str,
        top_k: int,
        vector: np.ndarray | None,
        *,
        threshold: float,
    ) -> SimilarHit | None:
        """在缓存中找与查询向量最相近的同 top_k 条目。

        返回最高分且 >= threshold 的那条；否则 None。`threshold <= 0` 或
        `vector is None` 时直接返回 None（语义匹配关闭 / 本轮拿不到向量）。

        **只比对 top_k 相同的条目**：top_k 影响召回条数与 citations，
        与精确 key 的语义保持一致 —— 不然 `top_k=3` 的问法会命中
        `top_k=5` 的答案，用户拿到的引用列表与他请求的参数不符。

        同分时取**先写入**的那条（遍历顺序 = 写入顺序，用严格 `>` 比较）：
        让命中结果与线程调度无关，测试里的断言才稳定。
        """
        if vector is None or threshold <= 0:
            return None
        q = np.asarray(vector, dtype=np.float32).ravel()
        q_norm = float(np.linalg.norm(q))
        if q_norm <= 0:
            return None
        q = q / q_norm
        q_dim = int(q.shape[0])

        best: SimilarHit | None = None
        dim_mismatch = 0
        now = self._wall()
        with self._lock:
            self._semantic_lookups += 1
            for entry in self._data.values():
                if entry.expire_at <= now:
                    # 过期条目直接跳过，不在这里删除：删除会让遍历中途
                    # 改变字典大小，也让"过期清理"有两个职责重叠的实现。
                    # 惰性过期由 get() 负责，语义查找只需不采信它们。
                    continue
                cached = entry.value
                if not isinstance(cached, CachedAnswer):
                    continue
                if cached.vector is None or cached.top_k != top_k:
                    continue
                if cached.vector.shape[0] != q_dim:
                    dim_mismatch += 1
                    continue
                sim = float(np.dot(q, cached.vector))
                if best is None or sim > best.similarity:
                    best = SimilarHit(entry=cached, similarity=sim)
            if best is not None and best.similarity < threshold:
                best = None
            if best is not None:
                self._semantic_hits += 1

        if dim_mismatch and not self._dim_mismatch_warned:
            self._dim_mismatch_warned = True
            logger.warning(
                "语义缓存中有 %d 条条目与当前查询向量维度不符（查询 %d 维），已跳过比对。"
                "通常是 EMBEDDING_PROVIDER 切换后缓存里还留着旧后端的向量；"
                "这些条目在 TTL 到期前无法参与语义匹配（精确命中仍正常）。",
                dim_mismatch, q_dim,
            )
        return best


# --------------------------------------------------------------------------
# 进程级单例 + key 规范化
# --------------------------------------------------------------------------
_cache: QaSemanticCache | None = None
_cache_lock = threading.Lock()


def get_cache(
    *, max_size: int = 128, ttl_seconds: float = 300.0
) -> QaSemanticCache:
    """获取进程级单例。**首次调用时按参数构造**，之后参数变化被忽略。

    与 `get_settings()` 的 `lru_cache` 模式同理：模块级单例让 `qa.py`
    无须关心 cache 实例的生命周期，测试可以通过 `reset_cache_for_tests`
    重建一个干净实例。
    """
    global _cache
    with _cache_lock:
        if _cache is None:
            _cache = QaSemanticCache(max_size=max_size, ttl_seconds=ttl_seconds)
        return _cache


def reconfigure_cache(
    *,
    max_size: int,
    ttl_seconds: float,
    wall_fn: Callable[[], float] | None = None,
) -> QaSemanticCache:
    """**测试用**：丢弃旧实例，按新参数重建。

    生产代码不调它（配置应在启动时通过环境变量传入）；
    让测试可以「先用 TTL=10s 写条目，再切到 TTL=1s 触发过期」，
    同时保持单例接口，避免每个测试都注入一次。

    `wall_fn` 注入用于 TTL 过期测试：在构造时就注入 mock 时钟，
    比之后改 `_wall` 属性更可靠（旧 expire_at 用旧 wall 计算，
    而 wall 改后读出来的"当前时间"可能仍然小于旧 expire_at）。
    """
    global _cache
    with _cache_lock:
        _cache = QaSemanticCache(
            max_size=max_size,
            ttl_seconds=ttl_seconds,
            wall_fn=wall_fn if wall_fn is not None else time.monotonic,
        )
        return _cache


def reset_cache_for_tests() -> None:
    """**测试用**：清空单例条目与统计（不重建实例）。

    与 `reconfigure_cache` 的区别：
      · reset：保留实例配置，仅清条目（适合 "换一个 query 测试"）；
      · reconfigure：丢掉实例 + 重置 stats（适合 "换 TTL 测试过期"）。
    """
    cache = get_cache()
    cache.clear()


def invalidate_qa_cache(reason: str = "") -> int:
    """**业务失效**：知识库内容变化后调用，让旧答案立即作废。

    调用时机（见 `services/ingest.py` 与 `api/notices.py`）：
      · 新通知入库（含文件去重首次落库）；
      · 通知被人工修正（标题/时间/地点改了，答案自然要变）；
      · 通知被删除。

    为什么用「全量失效」而不是「按 query 精细失效」：
    缓存的 key 是 `(query, top_k)`，而一次入库影响的是**哪些 query**无法在
    写入时得知 —— 那需要反向索引「哪个通知被哪些 query 引用」，成本远高于
    收益。本项目量级下缓存最多 128 条，全量失效的代价是几个 key 的重新计算；
    而漏失效的代价是用户拿到错误的否定答案。方向性取舍很明确。

    返回被清掉的条目数，便于调用方打日志观察失效频率。
    """
    n = get_cache().invalidate_all()
    if n:
        logger.info("QA 缓存已失效 %d 条（原因：%s）", n, reason or "知识库变更")
    return n


def normalize_query(query: str) -> str:
    """query 的字面归一化（缓存 key 的第一段）。

    - `split()` + `" ".join(...)`：把任意连续空白（含半角/全角空格、
      Tab、换行）压成一个半角空格，并顺带去掉首尾空白 —— 这是用户输入
      最容易踩的"看起来一样但 key 不同"的坑；
    - `lower()`：英文/数字大小写归一化；中文是 noop。

    单独成函数（而不是内联在 cache_key 里）是为了给标定脚本复用：
    eval/run_cache_threshold_eval.py 要统计「有多少对改写样本**字面**就能命中」，
    以此量化语义层的增量价值 —— 那份统计必须用与生产完全相同的归一化口径，
    否则算出来的增量是假的。
    """
    return " ".join(query.split()).strip().lower()


def cache_key(query: str, top_k: int) -> tuple[str, int]:
    """构造缓存 key = (归一化 query, top_k)。

    top_k 进 key 的原因：它影响召回条数与 citations，结果本就不同。
    """
    return (normalize_query(query), int(top_k))
