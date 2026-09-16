from __future__ import annotations

from collections.abc import Iterator

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from .config import settings

_IS_SQLITE = settings.database_url.startswith("sqlite")

connect_args = {"check_same_thread": False} if _IS_SQLITE else {}
engine = create_engine(settings.database_url, echo=False, future=True, connect_args=connect_args)
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False, future=True)

if _IS_SQLITE:
    # SQLite 默认**不校验外键**（`PRAGMA foreign_keys` 默认为 OFF，且是
    # **每连接**生效的），于是模型里声明的 `ondelete="CASCADE"` 全部形同虚设。
    #
    # 这不是理论问题，本项目实际踩到了：删除一条 Notice 后，
    # `notice_embeddings` 里对应的行**不会被级联删除**，成为孤儿。而
    # `VectorStore.load_from_db` 是按 `notice_embeddings` 全量重建索引的，
    # 它会忠实地把孤儿行也建成索引条目 —— 结果**向量索引里存在指向已删除
    # 通知的向量**：
    #   · 检索会召回这些 id，而 API 层 `db.get(Notice, id)` 拿到 None 后跳过，
    #     表现为"返回条数少于 top_k"，且随删除次数累积而恶化；
    #   · 相关性阈值判定拿不到文本 → 按 is_relevant 的约定**放开**，
    #     等于这些幽灵条目会绕过阈值；
    #   · 这些向量只能靠重启时 load_from_db 重建才消失（没有单条删除接口）。
    # 打开外键校验后，级联删除才真正生效，索引与库回到一致。
    #
    # 注意 PostgreSQL 无需此设置（外键约束天生强制执行）。
    @event.listens_for(Engine, "connect")
    def _enable_sqlite_foreign_keys(dbapi_connection, _connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()


class Base(DeclarativeBase):
    pass


def get_db() -> Iterator[Session]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """建表：优先走 Alembic 迁移，失败/未安装时回退 create_all。

    - 全新库：迁移脚本建出全部表（DDL 与模型同源，见 alembic/versions/）；
    - create_all 时代的存量库：自动打基线（stamp）而不重跑 DDL；
    - 迁移不可用（未装 alembic / 执行失败）：create_all 兜底，
      保证「最小依赖也能启动」的承诺不被运维工具绑架。
    """
    from . import models  # noqa: F401  确保模型已注册
    from .db_migrations import apply_migrations

    if not apply_migrations(engine):
        Base.metadata.create_all(bind=engine)
