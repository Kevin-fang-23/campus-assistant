"""Alembic 迁移环境：与应用的模型定义、数据库地址共用同一来源。

设计要点：
1. **target_metadata 直接取 `app.db.Base.metadata`** —— autogenerate 对比的
   就是线上那份模型，不存在"迁移元数据与模型不同步"的第二份真相。
2. **数据库地址不在 alembic.ini 写死**：优先用外部显式注入的
   `sqlalchemy.url`（应用启动/测试会这样注入），未注入则回退
   `app.config.settings.database_url` —— 与应用读同一份 .env，
   保证迁移打的库与应用连的库永远是同一个。
3. **render_as_batch**：SQLite 的 ALTER TABLE 能力极弱（不支持改列/删列），
   Alembic 的 batch 模式通过"建新表→拷数据→改名"绕过。本项目默认库就是
   SQLite，不开 batch 等于未来的迁移脚本写出来就报错。
4. **排除 `rate_limit_counters`**：该表由 `SqliteCountStore` 用裸 SQL
   `CREATE TABLE IF NOT EXISTS` 自建（见 services/rate_limit_store.py），
   不在 ORM 模型里。迁移既不该建它（代码自管），也不该删它（有真实数据）。
"""
from __future__ import annotations

import sys
from logging.config import fileConfig
from pathlib import Path

from sqlalchemy import engine_from_config, pool

from alembic import context

# 让 `import app` 可用：无论从 backend/ 还是别处调用 alembic，
# 都把 backend 根目录加进 sys.path（alembic.ini 的 prepend_sys_path
# 只保证 cwd 场景，这里兜住其余场景）。
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import models  # noqa: F401  确保所有模型注册进 metadata
from app.config import settings
from app.db import Base
from app.db_migrations import include_name

config = context.config

# 应用内编程式调用时会置 configure_logger=False：fileConfig 默认
# disable_existing_loggers=True，会**禁用应用已配置好的所有 logger**，
# 导致迁移之后应用日志静默消失。CLI 场景仍按 .ini 正常配置。
if config.config_file_name is not None and config.attributes.get("configure_logger", True):
    fileConfig(config.config_file_name)

# 外部未显式指定 url 时，回退到应用配置（同一份 .env）
if not config.get_main_option("sqlalchemy.url"):
    config.set_main_option("sqlalchemy.url", settings.database_url)

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """离线模式：只生成 SQL 不连库（给 DBA 审 SQL 用）。"""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        render_as_batch=True,
        include_name=include_name,
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """在线模式：连库执行迁移。"""
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            # SQLite 必须开 batch，否则涉及改列/删列的迁移无法执行
            render_as_batch=connection.dialect.name == "sqlite",
            # 检测类型变化（autogenerate 默认忽略部分类型差异）
            compare_type=True,
            include_name=include_name,
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
