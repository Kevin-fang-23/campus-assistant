"""以编程方式运行 Alembic 迁移 —— 启动钩子调用，无需部署者手动执行命令。

## 三种库状态，三种处理

1. **全新库**（无表）→ `upgrade head`，DDL 全部来自迁移脚本；
2. **已有 alembic_version** → 增量 `upgrade head`（只跑新修订）；
3. **create_all 时代的旧库**（有业务表、无 alembic_version）→
   `stamp head` 打基线而**不重跑 DDL**：表已存在，重跑初始迁移必然报
   "table already exists"。这是存量部署接入迁移的唯一正确姿势。

## 失败为什么只告警不阻断

本项目的既定承诺是「最小依赖也能启动」。alembic 未安装（极简环境）
或脚本异常时退回 `create_all` —— 全新库照样能建出正确结构，
只是失去增量演进能力。把迁移失败变成启动失败，等于让一个
运维工具反过来拖累核心可用性，权衡之下降级更合理。
"""
from __future__ import annotations

import logging
from pathlib import Path

from sqlalchemy import inspect
from sqlalchemy.engine import Engine

logger = logging.getLogger(__name__)

BACKEND_DIR = Path(__file__).resolve().parent.parent

# 不归迁移管理的表：由 SqliteCountStore 用裸 SQL `CREATE TABLE IF NOT EXISTS`
# 自建（见 services/rate_limit_store.py），且持有真实计数数据。
# 迁移既不建它（代码自管）也不删它；autogenerate 比对同样要跳过，
# 否则会误报 drop_table。单一事实来源放这里，alembic/env.py 与
# 迁移一致性测试共用。
EXCLUDED_TABLES = frozenset({"rate_limit_counters"})


def include_name(name: str | None, type_: str, _parent_names: dict) -> bool:
    """autogenerate 对象过滤器（Alembic include_name 协议）。"""
    return not (type_ == "table" and name in EXCLUDED_TABLES)


def make_config(database_url: str):
    """构造指向本仓库迁移脚本的 Alembic Config。

    - script_location 用绝对路径：调用方的 cwd 不一定在 backend/；
    - configure_logger=False：env.py 据此跳过 fileConfig —— 它默认会
      禁用进程里已配置好的所有 logger，应用内跑迁移会让业务日志静默消失。
    """
    from alembic.config import Config

    cfg = Config(str(BACKEND_DIR / "alembic.ini"))
    cfg.set_main_option("script_location", str(BACKEND_DIR / "alembic"))
    cfg.set_main_option("sqlalchemy.url", database_url)
    cfg.attributes["configure_logger"] = False
    return cfg


def apply_migrations(engine: Engine) -> bool:
    """对 engine 指向的库应用迁移；成功返回 True，失败/不可用返回 False。"""
    try:
        from alembic import command
    except ModuleNotFoundError:
        logger.warning("未安装 alembic，回退 create_all 建表（pip install alembic 可启用迁移）")
        return False

    try:
        insp = inspect(engine)
        tables = set(insp.get_table_names())
        # render_as_string(hide_password=False)：str(url) 会把密码隐藏成 ***，
        # 拿它另建连接会认证失败。SQLite 无密码，不受影响。
        cfg = make_config(engine.url.render_as_string(hide_password=False))
        if "alembic_version" in tables:
            command.upgrade(cfg, "head")
            logger.info("数据库迁移已应用（alembic upgrade head）")
        elif tables & {"documents", "notices", "tasks"}:
            command.stamp(cfg, "head")
            logger.info("检测到 create_all 创建的存量库，打基线至 head（alembic stamp，不重跑 DDL）")
        else:
            command.upgrade(cfg, "head")
            logger.info("全新库已由迁移脚本初始化（alembic upgrade head）")
        return True
    except Exception as exc:  # noqa: BLE001
        logger.warning("Alembic 迁移执行失败，回退 create_all：%s", exc)
        return False
