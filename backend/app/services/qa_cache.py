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
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger(__name__)


@dataclass
class _Entry:
    """单条缓存项。dataclass 而非 dict：避免运行期键名拼错。"""

    value: Any
    expire_at: float
    created_at: float = field(default_factory=time.monotonic)


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
            }

    def __len__(self) -> int:
        with self._lock:
            return len(self._data)


# --------------------------------------------------------------------------
# 进程级单例 + key 规范化
# --------------------------------------------------------------------------
_cache: TTLRUCache | None = None
_cache_lock = threading.Lock()


def get_cache(
    *, max_size: int = 128, ttl_seconds: float = 300.0
) -> TTLRUCache:
    """获取进程级单例。**首次调用时按参数构造**，之后参数变化被忽略。

    与 `get_settings()` 的 `lru_cache` 模式同理：模块级单例让 `qa.py`
    无须关心 cache 实例的生命周期，测试可以通过 `reset_cache_for_tests`
    重建一个干净实例。
    """
    global _cache
    with _cache_lock:
        if _cache is None:
            _cache = TTLRUCache(max_size=max_size, ttl_seconds=ttl_seconds)
        return _cache


def reconfigure_cache(
    *,
    max_size: int,
    ttl_seconds: float,
    wall_fn: Callable[[], float] | None = None,
) -> TTLRUCache:
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
        _cache = TTLRUCache(
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


def cache_key(query: str, top_k: int) -> tuple[str, int]:
    """构造缓存 key。

    - `strip()`：去首尾空白；
    - `split()` + `" ".join(...)`：把任意连续空白（含半角/全角空格、
      Tab、换行）压成一个半角空格 —— 这是用户输入最容易踩的"看起来
      一样但 key 不同"的坑；
    - `lower()`：英文/数字大小写归一化；中文是 noop。
    """
    return (" ".join(query.split()).strip().lower(), int(top_k))
