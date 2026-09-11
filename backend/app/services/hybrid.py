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

from sqlalchemy.orm import Session

from ..config import settings
from .bm25 import get_bm25_index
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


def hybrid_search(query: str, top_k: int | None = None) -> list[RetrievalHit]:
    """混合检索入口。

    候选池取 `hybrid_fetch_k` 而不是 `top_k` —— 融合前多召回一些，
    才有机会让"向量排 12 但 BM25 排 2"的文档被正确顶上来。

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
        vector_ranked = store.search_vector(store.embed(query), fetch_k)
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
# 索引写入
# ---------------------------------------------------------------------------
def bm25_text(db: Session, notice) -> str:  # noqa: ANN001
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


def index_notice(db: Session, notice) -> None:  # noqa: ANN001
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
