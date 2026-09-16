"""Alembic 迁移的契约测试。

要钉住的三条不变量：

1. **迁移脚本是模型的忠实快照**：对空库 `upgrade head` 后，
   autogenerate 比对必须零差异。一旦有人改了模型却忘了补迁移
   （或手改迁移改偏了），这条测试立刻红 —— 这正是引入迁移的初衷：
   让「模型与库结构漂移」从静默事故变成 CI 可拦截的失败。
2. **create_all 时代的存量库可无缝接入**：先 create_all 建表，
   再跑 apply_migrations，必须走 stamp（打基线）而非重跑 DDL，
   且已有数据不丢。存量部署升级时这是唯一不炸的路径。
3. **运行时自管表不被迁移误伤**：rate_limit_counters 由
   SqliteCountStore 裸 SQL 自建，迁移比对必须跳过它，
   否则 autogenerate 会误报 drop_table（删掉真实计数数据）。
"""
from __future__ import annotations

import pytest
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import Session

from app.db import Base
from app.db_migrations import apply_migrations, include_name


@pytest.fixture
def fresh_engine(tmp_path):
    """每个用例一个独立的全新 SQLite 文件库，互不污染。"""
    engine = create_engine(f"sqlite:///{(tmp_path / 'mig.db').as_posix()}")
    yield engine
    engine.dispose()


def _autogen_diff(engine) -> list:
    """对 engine 当前库结构与模型元数据做 autogenerate 比对，返回差异列表。"""
    from alembic.autogenerate import compare_metadata
    from alembic.migration import MigrationContext

    with engine.connect() as conn:
        ctx = MigrationContext.configure(
            conn,
            opts={
                "include_name": include_name,
                "compare_type": True,
                # 与 env.py 一致：SQLite 走 batch 语义比对
                "render_as_batch": conn.dialect.name == "sqlite",
            },
        )
        return compare_metadata(ctx, Base.metadata)


def test_upgrade_head_matches_models(fresh_engine) -> None:
    """契约 1：空库跑完迁移后，结构与模型零差异。"""
    assert apply_migrations(fresh_engine) is True

    # alembic_version 已写入且停在 head
    with fresh_engine.connect() as conn:
        rev = conn.execute(text("SELECT version_num FROM alembic_version")).scalar()
    assert rev, "迁移执行后必须留下版本标记"

    # 核心断言：模型里有的表/列/索引，库里全有且一致
    assert _autogen_diff(fresh_engine) == [], "迁移脚本与模型定义出现漂移"

    # 五张业务表齐全
    tables = set(inspect(fresh_engine).get_table_names())
    assert {"documents", "notices", "tasks", "task_events", "notice_embeddings"} <= tables


def test_legacy_create_all_db_is_stamped(fresh_engine) -> None:
    """契约 2：create_all 存量库接入迁移 = stamp 而非重跑，数据不丢。"""
    # 模拟旧版本部署：create_all 建表并写入一条数据
    #（裸 SQL 必须带全 NOT NULL 列 —— ORM 层的 default= 不会下沉到 DDL）
    Base.metadata.create_all(bind=fresh_engine)
    with Session(fresh_engine) as db:
        db.execute(
            text(
                "INSERT INTO documents (filename, kind, sha256, size_bytes, status, "
                "parse_meta, created_at) "
                "VALUES ('legacy.txt', 'text', 'abc', 0, 'parsed', '{}', "
                "'2026-01-01 00:00:00')"
            )
        )
        db.commit()

    assert apply_migrations(fresh_engine) is True

    with fresh_engine.connect() as conn:
        rev = conn.execute(text("SELECT version_num FROM alembic_version")).scalar()
        n = conn.execute(text("SELECT COUNT(*) FROM documents")).scalar()
    assert rev, "存量库必须被打上基线版本"
    assert n == 1, "stamp 不得重跑 DDL / 清空数据"


def test_rate_limit_counters_not_dropped(fresh_engine) -> None:
    """契约 3：代码自管的 rate_limit_counters 不会被迁移误删。"""
    assert apply_migrations(fresh_engine) is True
    # 模拟 SqliteCountStore 自建表并写入计数
    with fresh_engine.begin() as conn:
        conn.execute(
            text(
                "CREATE TABLE rate_limit_counters ("
                "day TEXT NOT NULL, key TEXT NOT NULL, value INTEGER NOT NULL, "
                "PRIMARY KEY (day, key))"
            )
        )
        conn.execute(
            text("INSERT INTO rate_limit_counters VALUES ('2026-09-16', 'global:qa', 3)")
        )

    # 比对必须跳过该表：既不误报 drop，也不误报其它操作
    diff = _autogen_diff(fresh_engine)
    assert diff == [], f"迁移比对不应触碰 rate_limit_counters：{diff}"

    with fresh_engine.connect() as conn:
        v = conn.execute(
            text("SELECT value FROM rate_limit_counters WHERE key = 'global:qa'")
        ).scalar()
    assert v == 3
