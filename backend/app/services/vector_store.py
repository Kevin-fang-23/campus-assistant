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

    # ---- 属性 ----
    @property
    def backend(self) -> str:
        return f"{'faiss' if _HAS_FAISS else 'numpy'}+{self._embedder.name}"

    @property
    def size(self) -> int:
        return len(self._ids)

    # ---- 内部 ----
    def _ensure_dim(self, dim: int) -> None:
        if self._dim == dim:
            return
        self._dim = dim
        self._ids = []
        self._matrix = None
        self._index = faiss.IndexFlatIP(dim) if _HAS_FAISS else None

    def _append(self, notice_id: int, vector: np.ndarray) -> None:
        vec = np.asarray(vector, dtype=np.float32).reshape(1, -1)
        self._ensure_dim(vec.shape[1])
        if notice_id in self._ids:
            return
        if _HAS_FAISS:
            self._index.add(vec)  # type: ignore[union-attr]
        else:
            self._matrix = vec if self._matrix is None else np.vstack([self._matrix, vec])
        self._ids.append(notice_id)

    # ---- 对外 ----
    def build_text(self, notice: Notice) -> str:
        parts = [notice.title or "", notice.summary or "", notice.course or "",
                 notice.location or "", " ".join(notice.tags or [])]
        return "\n".join(p for p in parts if p)

    def embed(self, text: str) -> np.ndarray:
        return self._embedder.embed_one(text)

    def load_from_db(self, db: Session) -> int:
        """启动时重建索引。"""
        rows = db.execute(select(NoticeEmbedding).order_by(NoticeEmbedding.id)).scalars().all()
        with self._lock:
            self._ids, self._matrix, self._index, self._dim = [], None, None, None
            for row in rows:
                vec = np.frombuffer(row.vector, dtype=np.float32)
                if self._dim is not None and vec.shape[0] != self._dim:
                    continue  # provider 变更导致维度不一致，跳过旧向量
                self._append(row.notice_id, vec)
        logger.info("向量索引重建完成: %d 条 (%s)", len(self._ids), self.backend)
        return len(self._ids)

    def upsert(self, db: Session, notice: Notice) -> np.ndarray:
        text = self.build_text(notice)
        vec = self.embed(text)
        existing = db.execute(
            select(NoticeEmbedding).where(NoticeEmbedding.notice_id == notice.id)
        ).scalar_one_or_none()
        if existing:
            existing.vector = vec.astype(np.float32).tobytes()
            existing.text = text
            existing.dim = int(vec.shape[0])
            existing.provider = self._embedder.name
        else:
            db.add(
                NoticeEmbedding(
                    notice_id=notice.id,
                    dim=int(vec.shape[0]),
                    provider=self._embedder.name,
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
                return []
            k = min(top_k + (len(exclude) if exclude else 0), len(self._ids))
            if _HAS_FAISS:
                scores, idxs = self._index.search(vec, k)  # type: ignore[union-attr]
                pairs = [(self._ids[i], float(s)) for s, i in zip(scores[0], idxs[0]) if i >= 0]
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


def get_store() -> VectorStore:
    global _store
    if _store is None:
        _store = VectorStore()
    return _store
