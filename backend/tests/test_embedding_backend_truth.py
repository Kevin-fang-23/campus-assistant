"""验证 `backend` 字段误报问题（修复前应失败）。

发现的两个问题：
1. 【显示层】DashScopeEmbedding.name 是类级常量 "dashscope"，
   embed() 内部回退到 local_hash 后 name 不变 → backend 谎报。
2. 【正确性】回退产生的向量维度（local_hash 256）与 dashscope（1024）不同，
   VectorStore._ensure_dim() 遇到维度变化会**清空整个索引**（self._ids = []），
   即网络抖动可导致已入库向量被静默丢弃、检索返回空。
"""
from __future__ import annotations

import numpy as np
import pytest

from app.config import settings
from app.providers.embedding import DashScopeEmbedding, LocalHashEmbedding
from app.providers.llm_client import TransportError
from app.services.vector_store import VectorStore


class _FailingClient:
    """模拟 dashscope embedding 接口不可用。"""

    def embed(self, texts, model):  # noqa: ANN001, ANN201
        raise TransportError("连接被拒绝")


@pytest.fixture
def dashscope_embedding(monkeypatch) -> DashScopeEmbedding:
    monkeypatch.setattr(settings, "dashscope_api_key", "test-key")
    return DashScopeEmbedding()


# --------------------------------------------------------------------------
# 问题 1：回退后 backend 名不真实
# --------------------------------------------------------------------------
def test_name_reflects_actual_backend_after_fallback(dashscope_embedding) -> None:
    """回退到本地哈希后，暴露给外部的后端名必须变成 local_hash。"""
    emb = dashscope_embedding
    emb._client = _FailingClient()  # type: ignore[assignment]

    vec = emb.embed(["操作系统作业截止时间"])

    assert vec.shape[0] == 1
    assert emb.active_name == "local_hash", (
        f"已回退但仍报 {emb.active_name!r}，前端/health 会被误导"
    )


def test_degraded_flag_exposed(dashscope_embedding) -> None:
    """需要一个显式标志供上层判断"当前是否处于降级状态"。"""
    emb = dashscope_embedding
    assert emb.degraded is False
    emb._client = _FailingClient()  # type: ignore[assignment]
    emb.embed(["x"])
    assert emb.degraded is True


def test_fallback_is_sticky(dashscope_embedding) -> None:
    """失败后应粘滞在本地哈希，避免后端来回切换（切换会引发维度翻转）。"""
    emb = dashscope_embedding
    calls = {"n": 0}

    class _Flaky:
        def embed(self, texts, model):  # noqa: ANN001, ANN201
            calls["n"] += 1
            raise TransportError("boom")

    emb._client = _Flaky()  # type: ignore[assignment]
    emb.embed(["first"])
    emb.embed(["second"])
    emb.embed(["third"])

    assert calls["n"] == 1, "粘滞后不应反复重试已判定不可用的上游"
    assert emb.active_name == "local_hash"


# --------------------------------------------------------------------------
# 问题 2：维度变化会清空整个索引
# --------------------------------------------------------------------------
def test_dimension_mismatch_does_not_wipe_index() -> None:
    """单个维度不一致的向量不得清空已建好的索引。"""
    store = VectorStore(embedder=LocalHashEmbedding(dim=8))
    store._append(101, np.ones(8, dtype=np.float32))
    store._append(102, np.ones(8, dtype=np.float32))
    assert store.size == 2

    # 模拟回退后端产出的不同维度向量
    store._append(103, np.ones(64, dtype=np.float32))

    assert 101 in store._ids, "维度不一致不应丢弃既有向量"
    assert 102 in store._ids, "维度不一致不应丢弃既有向量"
    assert store.size == 2, "维度不一致的向量应被拒绝而非触发全量清空"


def test_backend_string_after_fallback(dashscope_embedding) -> None:
    """端到端：store.backend 必须如实反映回退后的后端。"""
    emb = dashscope_embedding
    emb._client = _FailingClient()  # type: ignore[assignment]
    store = VectorStore(embedder=emb)

    emb.embed(["探测"])  # 触发一次回退
    assert "local_hash" in store.backend
    assert "dashscope" not in store.backend


# --------------------------------------------------------------------------
# 问题 3：库内历史向量维度与当前后端不符 → 检索静默返回空
# --------------------------------------------------------------------------
class _StubEmbedder:
    """可控维度的假 embedder，用于构造"库内维度落后于当前后端"的场景。"""

    name = "stub"

    def __init__(self, dim: int) -> None:
        self.dim = dim
        self._degraded = False

    @property
    def active_name(self) -> str:
        return self.name

    @property
    def degraded(self) -> bool:
        return self._degraded

    def embed(self, texts):  # noqa: ANN001, ANN201
        return np.tile(np.arange(self.dim, dtype=np.float32), (len(texts), 1))

    def embed_one(self, text):  # noqa: ANN001, ANN201
        return self.embed([text])[0]


@pytest.fixture
def isolated_db():
    """独立的内存库：这些用例要写 documents/notices/embeddings，
    绝不能污染 conftest 里那个被全局共享的测试库（否则 stats 类断言会漂移）。
    """
    from sqlalchemy import create_engine
    from sqlalchemy.orm import Session as _Session
    from sqlalchemy.orm import sessionmaker

    import app.models  # noqa: F401  确保所有表都注册到 Base.metadata
    from app.db import Base

    engine = create_engine("sqlite://", future=True)
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False, future=True)()
    try:
        yield session
    finally:
        session.close()
        engine.dispose()


def _make_notice(db, filename: str, title: str):
    from app.models import Document, Notice

    doc = Document(filename=filename, kind="text", sha256=f"sha-{filename}")
    db.add(doc)
    db.flush()
    notice = Notice(document_id=doc.id, category="other", title=title)
    db.add(notice)
    db.flush()
    return notice


def _put_embedding(db, notice_id: int, dim: int, provider: str = "local_hash") -> None:
    from app.models import NoticeEmbedding

    db.add(
        NoticeEmbedding(
            notice_id=notice_id,
            dim=dim,
            provider=provider,
            text="历史遗留向量",
            vector=np.ones(dim, dtype=np.float32).tobytes(),
        )
    )
    db.commit()


def test_load_from_db_skips_rows_with_wrong_dim(isolated_db) -> None:
    """库内向量维度与当前 embedder 不符时必须跳过并计数（而非加载成不可比索引）。"""
    notice = _make_notice(isolated_db, "dim_probe.txt", "维度探针通知")
    _put_embedding(isolated_db, notice.id, dim=8)

    # 当前后端期望 16 维 → 该历史 8 维向量应被跳过并计数
    store = VectorStore(embedder=_StubEmbedder(dim=16))  # type: ignore[arg-type]
    loaded = store.load_from_db(isolated_db)

    assert store.skipped_mismatched == 1, "维度不符的行应被计数"
    assert loaded == 0, "维度不符的历史向量不应进入索引"
    assert store.size == 0
    # 关键：索引维度绝不能由历史向量决定（否则查询向量永远比不上）
    assert store._dim != 8, "索引维度被历史向量污染了"
    assert store._dim is None, "本次全部被跳过，索引为空，维度自然未确定"
    # 后端名不得谎报成库内历史 provider
    assert store.backend.endswith("+stub")


def test_reindex_heals_dimension_mismatch(isolated_db) -> None:
    """reindex 应把全部通知按当前后端重算，使检索恢复可用。"""
    notice = _make_notice(isolated_db, "heal_probe.txt", "愈合探针通知")
    _put_embedding(isolated_db, notice.id, dim=4)

    store = VectorStore(embedder=_StubEmbedder(dim=16))  # type: ignore[arg-type]
    store.load_from_db(isolated_db)
    assert store.size == 0, "维度不符的历史向量不应进入索引"
    assert store.skipped_mismatched == 1

    # 执行 reindex 后应恢复，且维度统一到当前后端
    n = store.reindex(isolated_db)
    assert n == 1
    assert store.size == 1
    assert store.skipped_mismatched == 0
    assert store._dim == 16

    # 库里的行也应被改写为当前后端的维度与 provider
    from app.models import NoticeEmbedding

    row = (
        isolated_db.query(NoticeEmbedding)
        .filter(NoticeEmbedding.notice_id == notice.id)
        .one()
    )
    assert row.dim == 16
    assert row.provider == "stub"

    # 恢复后检索应真正可用（不再返回空）
    hits = store.search_vector(np.ones(16, dtype=np.float32), top_k=3)
    assert hits, "reindex 后检索必须可用"


def test_search_warns_and_returns_empty_on_dim_mismatch(caplog) -> None:
    """查询维度与索引不符时应返回空**并留下告警**，不能静默。"""
    import logging

    store = VectorStore(embedder=_StubEmbedder(dim=8))  # type: ignore[arg-type]
    store._append(1, np.ones(8, dtype=np.float32))
    assert store.size == 1

    with caplog.at_level(logging.WARNING, logger="app.services.vector_store"):
        hits = store.search_vector(np.ones(32, dtype=np.float32), top_k=3)

    assert hits == []
    assert any(
        "不一致" in r.message and "reindex" in r.message for r in caplog.records
    ), f"应给出可诊断的告警，实际日志: {[r.message for r in caplog.records]}"
