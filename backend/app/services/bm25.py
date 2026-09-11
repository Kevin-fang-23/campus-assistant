"""BM25 关键词检索（中文场景，零新依赖）。

## 为什么向量之外还要 BM25

向量检索擅长**语义相近**（"作业什么时候交" ↔ "提交截止时间"），
但对**精确字符串**很不可靠 —— 这是实测过的：查"操作系统大作业"时，
top1 竟是《计算机网络》实验报告通知，分数只有 0.5865。
校园场景里恰恰充斥这类精确串：课程名（《线性代数》）、房间号（教一 302 / B203）、
手机邮箱（138-0000-1234）、缩写（CET-4）。这些用 BM25 一次就命中。

两路互补：向量管"意思对"，BM25 管"字面对"。

## 为什么用字符 bigram 而不是分词

中文分词需要 jieba 之类的词典依赖，与项目"兜底不引依赖"的取向相悖
（`LocalHashEmbedding` 也是同样理由）。用**相邻双字**作 term 是中文信息检索的
经典免分词做法：既能保留大部分词的信息，又不会因分词错误丢掉召回。
例如"线性代数" → 线性/性代/代数，查询里出现同样串就能命中。

## 评分公式（标准 BM25，Okapi）

    IDF(t)  = ln(1 + (N - df(t) + 0.5) / (df(t) + 0.5))
    score   = Σ_t IDF(t) · tf(t,d)·(k1+1) / (tf(t,d) + k1·(1 - b + b·|d|/avgdl))

其中 tf 饱和项让词频收益递减（避免长文档靠堆词刷分），
|d|/avgdl 做文档长度归一化。k1 默认 1.5、b 默认 0.75 是文献常用值。
"""
from __future__ import annotations

import logging
import math
import re
import threading
from collections import defaultdict

from ..config import settings

logger = logging.getLogger(__name__)

# 中日韩统一表意文字（含扩展 A 区）
_CJK_RUN = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf]+")
# 英文单词与数字串（作为整体术语，不再拆 bigram）
_ASCII_RUN = re.compile(r"[a-zA-Z0-9]+")


def tokenize(text: str) -> list[str]:
    """把文本切成 BM25 的 term 序列。

    规则：
    - 中文串 → 相邻双字（bigram）；单字串保留原字；
    - 英文/数字串 → 整串小写（如 `B203`、`cet4`），保留整体语义不拆碎。

    注意：查询与文档必须用同一个函数，否则 term 对不上、命中率会归零。
    """
    if not text:
        return []

    tokens: list[str] = []
    for m in _CJK_RUN.finditer(text):
        run = m.group()
        if len(run) == 1:
            tokens.append(run)
        else:
            tokens.extend(run[i : i + 2] for i in range(len(run) - 1))
    for m in _ASCII_RUN.finditer(text):
        tokens.append(m.group().lower())
    return tokens


class BM25Index:
    """内存倒排索引 + BM25 打分。

    数据结构（都是增量维护，查询时不需要扫全库）：
        _postings: term -> {notice_id}      命中哪些文档
        _tf:       notice_id -> {term: 词频}
        _lengths:  notice_id -> 文档长度（term 数）
        _df:       term -> 文档频率

    `_postings` 是关键：查询时先用它取出候选文档集合，只对这些文档算分，
    而不是遍历全库。语料小时差别不明显，但结构上是对的。
    """

    def __init__(self, *, k1: float | None = None, b: float | None = None) -> None:
        self._k1 = settings.bm25_k1 if k1 is None else k1
        self._b = settings.bm25_b if b is None else b
        self._lock = threading.Lock()
        self._postings: dict[str, set[int]] = defaultdict(set)
        self._tf: dict[int, dict[str, int]] = {}
        self._lengths: dict[int, int] = {}
        self._df: dict[str, int] = {}
        self._total_length = 0

    # ---- 属性 ----
    @property
    def size(self) -> int:
        return len(self._tf)

    @property
    def avgdl(self) -> float:
        if not self._tf:
            return 0.0
        return self._total_length / len(self._tf)

    # ---- 增删 ----
    def add(self, notice_id: int, text: str) -> None:
        """写入/覆盖一条文档。重复调用同 id 会先移除旧版本，保证幂等。"""
        tokens = tokenize(text)
        with self._lock:
            self._remove_locked(notice_id)
            if not tokens:
                return
            tf: dict[str, int] = {}
            for t in tokens:
                tf[t] = tf.get(t, 0) + 1
            self._tf[notice_id] = tf
            self._lengths[notice_id] = len(tokens)
            self._total_length += len(tokens)
            for t in tf:
                self._postings[t].add(notice_id)
                self._df[t] = self._df.get(t, 0) + 1

    def remove(self, notice_id: int) -> None:
        with self._lock:
            self._remove_locked(notice_id)

    def _remove_locked(self, notice_id: int) -> None:
        """**调用方必须持有 self._lock。**"""
        tf = self._tf.pop(notice_id, None)
        if tf is None:
            return
        self._total_length -= self._lengths.pop(notice_id, 0)
        for t in tf:
            postings = self._postings.get(t)
            if postings is not None:
                postings.discard(notice_id)
                if not postings:
                    # 该 term 已无文档引用：彻底清理，避免 _df 残留导致 IDF 失真
                    del self._postings[t]
                    self._df.pop(t, None)
                else:
                    self._df[t] = len(postings)

    def clear(self) -> None:
        with self._lock:
            self._postings.clear()
            self._tf.clear()
            self._lengths.clear()
            self._df.clear()
            self._total_length = 0

    # ---- 载入 ----
    def load_documents(self, pairs) -> int:  # noqa: ANN001
        """用 (notice_id, text) 序列重建索引。

        刻意不接 DB 会话：BM25 是纯内存索引，文本从哪来由调用方决定。
        （早期版本在这里 import 了 Notice 并自己查库，把索引层和存储层耦合在一起，
        也没法在离线评测里复用。）
        """
        self.clear()
        for notice_id, text in pairs:
            self.add(notice_id, text)
        logger.info("BM25 索引重建完成: %d 条文档 | 词表 %d | 平均长度 %.1f",
                    self.size, len(self._postings), self.avgdl)
        return self.size

    # ---- 检索 ----
    def _idf(self, term: str) -> float:
        df = self._df.get(term, 0)
        if df == 0:
            return 0.0
        n = len(self._tf)
        # +1 包裹保证 IDF 恒正：避免高频词拿到负分把整体分数压成负数
        return math.log(1.0 + (n - df + 0.5) / (df + 0.5))

    def search(self, query: str, top_k: int = 5) -> list[tuple[int, float]]:
        """返回 [(notice_id, bm25_score)]，按分数降序。

        查询里出现的 term 若不在词表中（df=0）直接跳过 —— 这类词对区分度没有贡献，
        强行给负分反而会让"命中更多生僻词"的文档被惩罚。
        """
        terms = [t for t in tokenize(query) if t in self._postings]
        if not terms:
            return []

        with self._lock:
            n = len(self._tf)
            if n == 0:
                return []
            avgdl = self._total_length / n if n else 0.0
            idf = {t: self._idf(t) for t in set(terms)}
            k1, b = self._k1, self._b

            # 候选集：所有命中任一查询词的文档
            candidates: set[int] = set()
            for t in set(terms):
                candidates |= self._postings[t]

            scores: list[tuple[int, float]] = []
            for nid in candidates:
                tf_map = self._tf[nid]
                dl = self._lengths[nid]
                denom_len = 1.0 - b + b * (dl / avgdl if avgdl else 1.0)
                total = 0.0
                for t in set(terms):
                    f = tf_map.get(t, 0)
                    if not f:
                        continue
                    total += idf[t] * (f * (k1 + 1.0)) / (f + k1 * denom_len)
                if total > 0:
                    scores.append((nid, total))

        # 同分按 id 升序，保证结果确定（测试里有"前后一致"的断言）
        scores.sort(key=lambda p: (-p[1], p[0]))
        return scores[:top_k]


_bm25: BM25Index | None = None


def get_bm25_index() -> BM25Index:
    global _bm25
    if _bm25 is None:
        _bm25 = BM25Index()
    return _bm25


def reset_bm25_index() -> None:
    """重置单例（测试用；也便于后续做热重载）。"""
    global _bm25
    _bm25 = None
