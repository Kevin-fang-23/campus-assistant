"""混合检索：向量召回 + BM25 关键词召回，加权 RRF 融合。

## 整体架构

                        query
                          │
          ┌───────────────┴───────────────┐
          ▼                               ▼
  VectorStore.search_vector          BM25Index.search
  （FAISS/numpy 余弦，语义相近）      （倒排索引，字面精确）
          │                               │
          └───────────────┬───────────────┘
                          ▼
                 加权 RRF 融合 + 归一化
                          │
                          ▼
              [(notice_id, score, match, 分量分)]
                          │
              ┌───────────┴───────────┐
              ▼                       ▼
        /api/search               /api/qa
        （检索命中列表）           （拼 [编号] 上下文）

## 相关性阈值（"未找到相关内容"）

检索**恒返回 top-k** 会带来一个产品问题：问"今天天气怎么样"也会给出 3 条通知，
看着像在胡答。因此在融合后加一道相关性判定（`filter_by_relevance`）：

    候选文档与查询共享的「二字及以上词」个数  >=  search_min_bigram_overlap

为什么**不是**用分数做阈值 —— 实测三种分数判据全部失败（区间重叠）：

| 判据 | 噪声查询最高分 | 真实查询最低分 | 可分 |
|---|---|---|---|
| BM25 绝对分 | 3.773 | 2.872 | ❌ |
| 余弦相似度 | 0.154 | 0.043 | ❌ |
| 句级选片分 | 26.000 | 3.422 | ❌ |

根因是**单字重合**在中文里必然发生（"今天天气"撞"明天"里的"天"）。
改数二字及以上的重合即可过滤掉偶然碰撞。

阈值效果的实测（噪声样本 30 条，生产同款判据，见
`run_retrieval_eval.py::threshold_report` 与 RETRIEVAL_BASELINE.md 第五节）：
T=1 拦下大部分校外噪声（6/10）且真实误杀 0/129；「校园相关但语料无答案」
类查询共享通用校园词汇（期末/宿舍/预约），词面计数无法区分，属已知边界。

详见 `backend/eval/RETRIEVAL_BASELINE.md` 与 `run_retrieval_eval.py --calibrate`。

## 为什么用 RRF（Reciprocal Rank Fusion）而不是加权分数相加

两路分数**不可直接相加**：余弦相似度落在 [-1,1] 且随 embedding 模型变化，
BM25 是开区间上的无界正值（实测常见 3~30）。硬做 min-max 归一化又会引入
"每个 query 的分布不同"的不稳定性（同一文档在不同 query 下得分漂移）。

RRF 只用**排名**，天然免疫尺度差异：

    score(d) = Σ_r  w_r / (k + rank_r(d))

k 是平滑常数（默认 60，来自原论文）：k 越大，靠前名次的优势越平缓。
`w_r` 是每一路的权重，可配、可评测调优（见 eval/run_retrieval_eval.py）。

## 展示分数怎么来

RRF 原始分数量级是 1/(k+1) ≈ 0.016，直接给前端会被渲染成 "2%"。
因此按**理论最大值**（同时被两路都排第 1）归一化到 [0,1]：

    score = rrf(d) / ((w_vec + w_bm25) / (k + 1))

读作"相对于两路完全一致的最优情形的相关性"，**不是概率**。
同时把两路的分量分原样带出去，前端/排查可以看出一条文是靠哪一路命中的。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from sqlalchemy.orm import Session

from ..config import settings
from .bm25 import get_bm25_index, tokenize
from .vector_store import get_store

logger = logging.getLogger(__name__)

# 只有单路命中时，另一路的贡献为 0（而不是当作"最后一名"），
# 否则单路命中的文档会被系统性地压低，两路都命中反而不占优势。
_MATCH_BOTH = "both"
_MATCH_VECTOR = "vector"
_MATCH_BM25 = "bm25"


@dataclass(frozen=True)
class RetrievalHit:
    """一条混合检索结果。

    score        归一化融合分 [0,1]，用于排序与展示（相对相关性，非概率）
    score_vector 余弦相似度（该文档未被向量路召回时为 None）
    score_bm25   原始 BM25 分（该文档未被 BM25 路召回时为 None）
    match        命中来源：both / vector / bm25
    """

    notice_id: int
    score: float
    score_vector: float | None
    score_bm25: float | None
    match: str


def _contribution(rank: int | None, weight: float, k: int) -> float:
    return weight / (k + rank) if rank else 0.0


# ---------------------------------------------------------------------------
# 相关性判定（阈值）
# ---------------------------------------------------------------------------
# 「共享二字词」= 查询与文档共同拥有的、长度 ≥2 的 term。
# 直接复用 BM25 的 tokenize：它产出的中文 term 本身就是相邻双字（bigram），
# 英文/数字串则是整串（如 `b203`、`cet`）—— 两者都满足「长度 ≥2」，
# 且与检索路径用的是同一套切词，不会出现"检索能命中但判定说无关"的不一致。
def _meaningful_terms(text: str) -> set[str]:
    """取出文本里所有长度 ≥2 的 term（即"二字及以上"的信息单位）。

    刻意排除 **单字**（长度 1 的 CJK 字符）：
    中文里单字碰撞是必然的 —— "今天天气怎么样" 会与 "明天上午10点" 共享 '天'，
    "量子纠缠" 会与 "期末周期间" 共享 '期'。这类重合不携带语义信息，
    实测正是它导致分数阈值全部失效（见 RETRIEVAL_BASELINE.md 第六节）。
    """
    return {t for t in tokenize(text) if len(t) >= 2}


def shared_term_count(query: str, doc_text: str) -> int:
    """查询与文档共享的「二字及以上词」个数（去重）。

    这是阈值判定的核心信号。返回计数而非比率：比率会被长文档稀释，
    而我们要回答的是"这两个文本有没有实质的词面交集"，与文档长度无关。
    """
    q = _meaningful_terms(query)
    if not q:
        return 0
    return len(q & _meaningful_terms(doc_text))


def is_relevant(hit: RetrievalHit, query: str, doc_texts: dict[int, str]) -> bool:
    """该命中是否「与查询足够相关」。

    doc_texts 由调用方提供（notice_id -> 索引文本），避免在这里查库 ——
    保持本模块的纯函数性质，便于单测与离线评测复用。

    查不到文本时返回 True（**放开**而非收紧）：宁可多给一条结果，
    也不要因为拿不到文本就把真实命中丢掉。
    """
    threshold = settings.search_min_bigram_overlap
    if threshold <= 0:  # 关闭判定
        return True
    text = doc_texts.get(hit.notice_id)
    if text is None:
        return True
    return shared_term_count(query, text) >= threshold


def filter_by_relevance(
    hits: list[RetrievalHit], query: str, doc_texts: dict[int, str]
) -> tuple[list[RetrievalHit], bool]:
    """按相关性阈值过滤命中，返回 (保留的命中, 是否**因过滤而结果为空**)。

    第二个返回值的语义刻意收窄为「是否因为阈值导致一条都没剩下」，
    而不是「是否有任何一条被丢弃」。原因：
      · 候选池本来就故意多召回（`hybrid_fetch_k`），几乎总会有边缘候选被丢，
        若那种情况也置 True，这个标志位就恒为 True、失去信息量；
      · 前端唯一需要它区分的是两种**空结果**的文案：
          空 + filtered=True  → 「未找到相关内容」（有候选，但都不相关）
          空 + filtered=False → 「没有匹配的历史通知」（库里确实没有）
    """
    if settings.search_min_bigram_overlap <= 0:
        return hits, False
    kept = [h for h in hits if is_relevant(h, query, doc_texts)]
    emptied = not kept and bool(hits)
    if emptied and settings.search_keep_if_filtered:
        # 灰度观察模式：只标记不丢弃，便于在生产上看阈值会拦掉什么
        return hits, True
    return kept, emptied


def fuse(
    vector_ranked: list[tuple[int, float]],
    bm25_ranked: list[tuple[int, float]],
    *,
    k: int,
    w_vector: float,
    w_bm25: float,
) -> list[RetrievalHit]:
    """把两路排名融成一个有序列表。纯函数，便于单测与离线评测复用。"""
    rank_v = {nid: i + 1 for i, (nid, _) in enumerate(vector_ranked)}
    rank_b = {nid: i + 1 for i, (nid, _) in enumerate(bm25_ranked)}
    cos = dict(vector_ranked)
    bm = dict(bm25_ranked)

    # 归一化分母按**实际可用的路径**算，不是按配置权重算。
    # 否则 BM25 索引尚未构建（或某一路完全无结果）时，唯一可用那一路的第一名
    # 会被系统性压到 w_该路/(w_v+w_b)（例如 0.3），界面上表现为"检索分只有 30%"，
    # 看着像坏了，实际是分母算错了。
    w_v_eff = w_vector if vector_ranked else 0.0
    w_b_eff = w_bm25 if bm25_ranked else 0.0
    max_possible = (w_v_eff + w_b_eff) / (k + 1)
    if max_possible <= 0:
        return []

    hits: list[RetrievalHit] = []
    for nid in set(rank_v) | set(rank_b):
        rrf = _contribution(rank_v.get(nid), w_vector, k) + _contribution(
            rank_b.get(nid), w_bm25, k
        )
        if nid in rank_v and nid in rank_b:
            match = _MATCH_BOTH
        elif nid in rank_v:
            match = _MATCH_VECTOR
        else:
            match = _MATCH_BM25
        hits.append(
            RetrievalHit(
                notice_id=nid,
                score=min(1.0, rrf / max_possible),
                score_vector=cos.get(nid),
                score_bm25=bm.get(nid),
                match=match,
            )
        )

    # 同分按 notice_id 升序 → 结果确定，测试里的"前后一致"断言才稳定
    hits.sort(key=lambda h: (-h.score, h.notice_id))
    return hits


def hybrid_search(
    query: str,
    top_k: int | None = None,
    *,
    query_vector: np.ndarray | None = None,
) -> list[RetrievalHit]:
    """混合检索入口。

    候选池取 `hybrid_fetch_k` 而不是 `top_k` —— 融合前多召回一些，
    才有机会让"向量排 12 但 BM25 排 2"的文档被正确顶上来。

    `query_vector`：调用方已经算过查询向量时传进来**复用**，避免同一请求
    重复调用 embedding。目前唯一的生产调用方是 `/api/qa` —— 它为语义缓存
    探测算过一次向量，若不复用就要再算一次（白花一次上游调用）。
    不传则本函数自己算，行为与改造前完全一致。

    降级保证（任一环节不可用都不影响问答主流程）：
    - `hybrid_enabled=False` → 退回纯向量（与改造前行为一致，可随时回滚）；
    - BM25 索引为空（尚未构建）→ 退回纯向量；
    - 向量路为空但 BM25 有结果 → 照常返回 BM25 结果。
    """
    store = get_store()
    k = top_k or settings.search_top_k
    fetch_k = max(settings.hybrid_fetch_k, k * 3)

    # 向量路：复用现有检索，不另起一套
    vector_ranked: list[tuple[int, float]] = []
    try:
        vector_ranked = store.search_vector(
            query_vector if query_vector is not None else store.embed(query), fetch_k
        )
    except Exception as exc:  # noqa: BLE001
        # embedding 上游故障时仍应能靠关键词兜住检索
        logger.warning("向量检索失败，本次仅用 BM25：%s", exc)

    bm25_index = get_bm25_index()
    bm25_ranked: list[tuple[int, float]] = (
        bm25_index.search(query, fetch_k) if bm25_index.size else []
    )

    if not settings.hybrid_enabled or not bm25_ranked:
        if settings.hybrid_enabled and vector_ranked and not bm25_ranked:
            logger.debug("BM25 索引为空，本次退回纯向量检索")
        return [
            RetrievalHit(
                notice_id=nid,
                score=score,
                score_vector=score,
                score_bm25=None,
                match=_MATCH_VECTOR,
            )
            for nid, score in vector_ranked[:k]
        ]

    fused = fuse(
        vector_ranked,
        bm25_ranked,
        k=settings.hybrid_rrf_k,
        w_vector=settings.hybrid_weight_vector,
        w_bm25=settings.hybrid_weight_bm25,
    )
    return fused[:k]


# ---------------------------------------------------------------------------
# 带相关性判定的检索入口（供 API 使用）
# ---------------------------------------------------------------------------
def search_docs_text(db: Session, notice_ids: list[int]) -> dict[int, str]:
    """批量取出若干通知的**索引文本**（等价 BM25 路文本），用于相关性判定。

    必须与 `bm25_text` 一致：判定要看的正是"检索时到底比对了什么字符串"，
    若这里换成 title 或 summary，就会出现"检索命中但判定说无关"的割裂。
    """
    if not notice_ids:
        return {}
    from sqlalchemy import select

    from ..models import Document, Notice

    rows = (
        db.execute(select(Notice).where(Notice.id.in_(notice_ids))).scalars().all()
    )
    # 先把用到的正文一次性捞出来，避免逐条 db.get 造成 N+1
    doc_ids = [n.document_id for n in rows if n.document_id]
    raws: dict[int, str] = {}
    if doc_ids:
        docs = (
            db.execute(select(Document).where(Document.id.in_(doc_ids))).scalars().all()
        )
        raws = {d.id: d.raw_text or "" for d in docs}

    base_text = get_store().build_text
    out: dict[int, str] = {}
    for n in rows:
        parts = [base_text(n)]
        raw = raws.get(n.document_id) if n.document_id else None
        if raw:
            parts.append(raw)
        out[n.id] = "\n".join(p for p in parts if p)
    return out


def hybrid_search_filtered(
    db: Session, query: str, top_k: int | None = None
) -> tuple[list[RetrievalHit], bool]:
    """混合检索 + 相关性阈值过滤，返回 (命中, 是否有命中被过滤)。

    多召回一些再过滤：若直接按 top_k 取，过滤后可能只剩 1 条，
    而其实第 k+1 条是相关的。因此候选池取 `hybrid_fetch_k`，过滤完再截断。
    """
    k = top_k or settings.search_top_k
    # 阈值开启时多取候选，避免"过滤后不足 top_k 但后面其实有相关结果"
    fetch = max(settings.hybrid_fetch_k, k) if settings.search_min_bigram_overlap > 0 else k
    hits = hybrid_search(query, fetch)
    texts = search_docs_text(db, [h.notice_id for h in hits])
    kept, filtered = filter_by_relevance(hits, query, texts)
    return kept[:k], filtered


# ---------------------------------------------------------------------------
# 索引写入
# ---------------------------------------------------------------------------
def bm25_text(db: Session, notice) -> str:
    """BM25 索引的文本 = 结构化字段 **+ 文档正文**。

    为什么必须带正文（这是端到端测试抓出来的缺陷）：
    只索引 `build_text`（标题/摘要/课程/地点/标签）时，**正文里的精确串搜不到**。
    实测查询 `027-87659999` 时，该电话从未被抽取进任何结构化字段，
    BM25 只能匹配到 `027` 这个共享前缀，结果被另一条同样以 027 开头的通知抢了首位 ——
    而"精确串一次命中"正是 BM25 存在的理由。

    向量路仍只用结构化字段：语义检索吃提炼后的信息更干净，
    且改动它需要重算全库向量。**两路不必用同一份文本** ——
    RRF 只用排名，不需要两路分数可比；各自索引自己擅长的那部分才对。
    """
    from ..models import Document

    parts = [get_store().build_text(notice)]
    document = db.get(Document, notice.document_id)
    if document is not None and document.raw_text:
        parts.append(document.raw_text)
    return "\n".join(p for p in parts if p)


def index_notice(db: Session, notice) -> None:
    """把一条通知同时写入向量索引与 BM25 索引。

    这是唯一的生产写入入口 —— 之前直接在 ingest 里调 `get_store().upsert()`，
    加第二路索引后若不收口，很容易出现"向量更新了而 BM25 没更新"的静默不一致，
    表现为检索时而命中时而命不中，极难定位。
    """
    store = get_store()
    store.upsert(db, notice)
    get_bm25_index().add(notice.id, bm25_text(db, notice))


def rebuild_bm25(db: Session) -> int:
    """按库内通知重建 BM25 索引（启动、重算向量后调用）。"""
    from sqlalchemy import select

    from ..models import Notice

    rows = db.execute(select(Notice).order_by(Notice.id)).scalars().all()
    return get_bm25_index().load_documents((n.id, bm25_text(db, n)) for n in rows)


def reindex_all(db: Session) -> tuple[int, int]:
    """向量全量重算 + BM25 重建，返回 (向量条数, BM25 条数)。

    两者必须一起重建：向量重算会改写 `notice_embeddings`，
    若 BM25 还用旧文本，两路排名就会基于不同内容，融合结果失去意义。
    """
    n_vec = get_store().reindex(db)
    n_bm = rebuild_bm25(db)
    return n_vec, n_bm


def reset_indexes() -> None:
    """重置 BM25 单例（测试用）。向量单例的重置由 vector_store 负责。"""
    from .bm25 import reset_bm25_index

    reset_bm25_index()
