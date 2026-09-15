"""FAISS 向量库封装。

向量本体落 DB（notice_embeddings），进程启动时重建内存索引，
避免索引文件与数据库不一致；FAISS 不可用时自动退化为 numpy 余弦相似度（结果等价，仅速度差异）。
"""
from __future__ import annotations

import logging
import threading

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..config import settings
from ..models import Notice, NoticeEmbedding
from ..providers.embedding import DASHSCOPE_DIM as _DASHSCOPE_DIM
from ..providers.embedding import BaseEmbedding, get_embedding

logger = logging.getLogger(__name__)

try:  # pragma: no cover
    import faiss  # type: ignore

    _HAS_FAISS = True
except Exception:  # noqa: BLE001  pragma: no cover
    faiss = None  # type: ignore
    _HAS_FAISS = False


class VectorStore:
    def __init__(self, embedder: BaseEmbedding | None = None) -> None:
        self._embedder = embedder or get_embedding()
        self._lock = threading.Lock()
        self._ids: list[int] = []
        self._matrix: np.ndarray | None = None   # numpy 兜底用
        self._index = None                       # faiss index
        self._dim: int | None = None
        self._skipped_mismatched = 0             # 上次 load_from_db 跳过的维度不符行数

    # ---- 属性 ----
    @property
    def embedder(self) -> BaseEmbedding:
        """暴露当前 embedder，供 /health 上报 configured/active 后端差异。"""
        return self._embedder

    @property
    def backend(self) -> str:
        """如实反映实际生效的后端（含 embedding 运行时回退）。"""
        return f"{'faiss' if _HAS_FAISS else 'numpy'}+{self._embedder.active_name}"

    @property
    def size(self) -> int:
        return len(self._ids)

    @property
    def skipped_mismatched(self) -> int:
        """上次加载时因维度与当前 embedder 不符而跳过的向量条数。

        大于 0 说明库里的历史向量与当前后端不兼容（通常是 embedding
        provider 切换或运行时回退留下的），需要 reindex() 才能真正恢复检索。
        """
        return self._skipped_mismatched

    # ---- 内部 ----
    def _append(self, notice_id: int, vector: np.ndarray) -> bool:
        """追加一条向量，返回是否成功。**调用方必须持有 self._lock。**

        维度不一致时**拒绝该向量**，而不是重建索引。

        原实现用 _ensure_dim() 在维度变化时清空 self._ids —— 那是毁灭性的：
        DashScopeEmbedding 回退到 local_hash 时维度会从 1024 变 256，
        于是**一次网络抖动就能把已入库的全部向量静默丢掉**，检索随即返回空，
        且没有任何错误提示。宁可少一条向量 + 打一条告警，也不能丢掉整库。
        """
        vec = np.asarray(vector, dtype=np.float32).reshape(1, -1)
        dim = int(vec.shape[1])
        if self._dim is None:
            self._dim = dim
            self._index = faiss.IndexFlatIP(dim) if _HAS_FAISS else None
        elif dim != self._dim:
            logger.warning(
                "向量维度不一致（索引 %d 维，新向量 %d 维），跳过 notice_id=%s。"
                "通常是 embedding 后端在 dashscope(%d 维)/local_hash(%d 维) 之间切换导致；"
                "执行 reindex 重建全部向量即可恢复。",
                self._dim, dim, notice_id, _DASHSCOPE_DIM, settings.embedding_dim,
            )
            return False
        if notice_id in self._ids:
            return False
        if _HAS_FAISS:
            self._index.add(vec)  # type: ignore[union-attr]
        else:
            self._matrix = vec if self._matrix is None else np.vstack([self._matrix, vec])
        self._ids.append(notice_id)
        return True

    # ---- 对外 ----
    def build_text(self, notice: Notice) -> str:
        parts = [notice.title or "", notice.summary or "", notice.course or "",
                 notice.location or "", " ".join(notice.tags or [])]
        return "\n".join(p for p in parts if p)

    def embed(self, text: str) -> np.ndarray:
        return self._embedder.embed_one(text)

    def index_text(self, notice_id: int, text: str) -> np.ndarray:
        """把一段文本直接索引进向量库（不落库）。

        供离线评测与批量重建使用 —— 评测脚本要在内存里搭一套与生产同源的
        索引，若走 `upsert` 就必须有 DB 会话，评测会被数据库绑住。
        """
        vec = self.embed(text)
        with self._lock:
            self._append(notice_id, vec)
        return vec

    def load_from_db(self, db: Session) -> int:
        """启动时重建索引。

        以**当前 embedder 的期望维度**为准筛选历史向量，而不是"第一条记录
        说了算"。原因：库里的行可能是历史后端留下的（如早期 dashscope 不可用
        时写入的 local_hash 256 维向量），若以它们为准建索引，之后所有的查询
        向量（1024 维）都会因维度不符被 search_vector 直接判空 ——
        表现为"检索莫名其妙返回 0 条"，且没有任何报错。
        """
        rows = db.execute(select(NoticeEmbedding).order_by(NoticeEmbedding.id)).scalars().all()
        expected = int(self._embedder.dim)
        skipped = 0
        with self._lock:
            self._ids, self._matrix, self._index, self._dim = [], None, None, None
            for row in rows:
                vec = np.frombuffer(row.vector, dtype=np.float32)
                if vec.shape[0] != expected:
                    skipped += 1
                    continue
                self._append(row.notice_id, vec)
        self._skipped_mismatched = skipped
        if skipped:
            logger.warning(
                "向量索引重建：跳过 %d 条维度与当前后端不符的历史向量"
                "（当前 %s 期望 %d 维，库内共 %d 条）。"
                "这些通知暂时无法被检索，执行 reindex 可全部恢复。",
                skipped, self._embedder.active_name, expected, len(rows),
            )
        logger.info("向量索引重建完成: %d 条 (%s)", len(self._ids), self.backend)
        return len(self._ids)

    def reindex(self, db: Session) -> int:
        """用当前 embedder 重新计算**全部**通知的向量，并重建索引。

        这是维度/后端变更后的唯一恢复路径：把库里混杂的历史向量统一成
        当前后端的维度，使索引与查询向量重新可比。
        代价是每个通知一次 embedding 调用（本项目量级下可忽略）。

        **降级态拒绝执行**：embedder 粘性降级（如 dashscope 抖动 → local_hash）
        时，降级是「上游暂时不可用」而非「后端切换」。此时若照常重算，
        会用 256 维的降级向量**覆盖**库里全部 1024 维历史向量 ——
        等上游恢复、进程重启后，查询向量与库内向量不可比，检索静默返回 0 条，
        且没有任何报错。因此降级态下直接拒绝（返回 0），只告警；
        上游恢复后重启服务，load_from_db 即可正常加载历史向量。
        """
        if self._embedder.degraded:
            logger.warning(
                "当前 embedding 处于降级态（%s，维度 %d），拒绝 reindex："
                "用降级向量覆盖历史向量会污染数据库。"
                "上游恢复后重启服务即可自动恢复检索。",
                self._embedder.active_name, self._embedder.dim,
            )
            return 0
        notices = db.execute(select(Notice).order_by(Notice.id)).scalars().all()
        with self._lock:  # 先清空，让新后端重新决定索引维度
            self._ids, self._matrix, self._index, self._dim = [], None, None, None
        for notice in notices:
            self.upsert(db, notice)
        self._skipped_mismatched = 0
        db.commit()
        logger.info(
            "向量索引全量重建完成: %d/%d 条 (%s)",
            self.size, len(notices), self.backend,
        )
        return self.size

    def upsert(self, db: Session, notice: Notice) -> np.ndarray:
        text = self.build_text(notice)
        vec = self.embed(text)
        # 记录**实际**产出该向量的后端名（回退时是 local_hash），
        # 否则 DB 里的 provider 列会与向量真实来源不符，重建索引时无法自证。
        provider = self._embedder.active_name
        existing = db.execute(
            select(NoticeEmbedding).where(NoticeEmbedding.notice_id == notice.id)
        ).scalar_one_or_none()
        if existing:
            existing.vector = vec.astype(np.float32).tobytes()
            existing.text = text
            existing.dim = int(vec.shape[0])
            existing.provider = provider
        else:
            db.add(
                NoticeEmbedding(
                    notice_id=notice.id,
                    dim=int(vec.shape[0]),
                    provider=provider,
                    text=text,
                    vector=vec.astype(np.float32).tobytes(),
                )
            )
        db.flush()
        with self._lock:
            self._append(notice.id, vec)
        return vec

    def search_vector(
        self, vector: np.ndarray, top_k: int = 5, exclude: set[int] | None = None
    ) -> list[tuple[int, float]]:
        with self._lock:
            if not self._ids:
                return []
            vec = np.asarray(vector, dtype=np.float32).reshape(1, -1)
            if self._dim and vec.shape[1] != self._dim:
                # 不能悄悄返回空：调用方会把它当成"没有相关通知"展示给用户，
                # 而真实原因是索引与查询向量不可比（后端切换所致）。
                logger.warning(
                    "查询向量维度（%d）与索引维度（%d）不一致，检索结果为空。"
                    "当前后端=%s，执行 reindex 可恢复。",
                    vec.shape[1], self._dim, self._embedder.active_name,
                )
                return []
            k = min(top_k + (len(exclude) if exclude else 0), len(self._ids))
            if _HAS_FAISS:
                scores, idxs = self._index.search(vec, k)  # type: ignore[union-attr]
                # FAISS 契约保证 scores 与 idxs 同形；strict=True 顺带兜住异常返回
                pairs = [
                    (self._ids[i], float(s))
                    for s, i in zip(scores[0], idxs[0], strict=True)
                    if i >= 0
                ]
            else:
                sims = (self._matrix @ vec.T).ravel()  # type: ignore[operator]
                order = np.argsort(-sims)[:k]
                pairs = [(self._ids[i], float(sims[i])) for i in order]
        if exclude:
            pairs = [p for p in pairs if p[0] not in exclude]
        return pairs[:top_k]

    def search(self, query: str, top_k: int | None = None) -> list[tuple[int, float]]:
        return self.search_vector(self.embed(query), top_k or settings.search_top_k)

    def find_duplicate(
        self, vector: np.ndarray, exclude_id: int | None = None
    ) -> tuple[int | None, float]:
        hits = self.search_vector(vector, top_k=1, exclude={exclude_id} if exclude_id else None)
        if not hits:
            return None, 0.0
        nid, score = hits[0]
        return (nid, score) if score >= settings.dedup_threshold else (None, score)


_store: VectorStore | None = None
_store_lock = threading.Lock()


def get_store() -> VectorStore:
    """进程级单例（线程安全）。

    FastAPI 的同步端点跑在线程池里，多个请求线程可能并发地**首次**调用
    本函数（例如启动后第一批检索请求）——无锁时两个线程会各自构造一个
    VectorStore，后写者覆盖先写者，先构造的索引里已写入的向量全部丢失，
    表现为「检索时而召回时而空」。与 qa_cache.get_cache() 的单例模式保持
    一致：双重检查 + 模块级锁。
    """
    global _store
    if _store is None:
        with _store_lock:
            if _store is None:
                _store = VectorStore()
    return _store
