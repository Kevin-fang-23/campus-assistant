"""生产路径现场确认：真实启动应用（走 lifespan），检查限流后端是否真挂上。

为什么需要单独确认：**测试环境通过 ≠ 部署配置生效**。
单测里的 limiter 是手工注入 store 的，而生产靠 `main.py` 的启动钩子挂载 ——
若钩子写错、或 `RATE_LIMIT_STORE` 配错，单测全绿但线上护栏是空的。

检查 6 项：
  1. 启动后 `/health` 能通（没有因为限流持久化初始化失败而起不来）
  2. 限流器挂的后端类型是 SqliteCountStore
  3. 生产库 campus.db 里真建了 rate_limit_counters 表
  4. journal_mode 真的是 wal（WAL 没开则单请求开销 4ms，等于方案失效）
  5. 真正打一次 /api/qa，确认计数落到了库里
  6. 用**独立连接**读库，排除"读的是内存"的可能

用法（在 backend/ 目录下）：
    python scripts/verify_rate_limit_prod_path.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# 本文件位于 backend/scripts/，故 backend 根目录是上两级
BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND))

from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy import text  # noqa: E402

from app.config import settings  # noqa: E402
from app.db import engine  # noqa: E402
from app.main import app  # noqa: E402
from app.services.rate_limit import get_limiter  # noqa: E402

results: list[str] = []


def check(name: str, ok: bool, detail: str) -> None:
    results.append(f"{'✅' if ok else '❌'} {name}: {detail}")
    return ok


print("=" * 70)
print("生产路径确认（走真实 lifespan 启动）")
print("=" * 70)
print(f"RATE_LIMIT_STORE = {settings.rate_limit_store!r}")
print(f"database_url     = {settings.database_url}")
print()

with TestClient(app) as client:
    # 1. 应用能起来 + /health 通
    r = client.get("/health")
    ok1 = check("/health 可访问", r.status_code == 200, f"status={r.status_code}")

    # 2. 限流器挂的是不是 SQLite 后端
    store = get_limiter()._store
    ok2 = check(
        "限流后端已切换",
        type(store).__name__ == "SqliteCountStore",
        f"实际 = {type(store).__name__}",
    )

    # 3. 表真的建了 + journal_mode 是 wal
    with engine.connect() as conn:
        tbl = conn.execute(text(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='rate_limit_counters'"
        )).scalar()
        mode = conn.execute(text("PRAGMA journal_mode")).scalar()
    ok3 = check("rate_limit_counters 表存在", tbl == "rate_limit_counters", f"{tbl!r}")
    ok4 = check("journal_mode 为 wal", str(mode).lower() == "wal", f"{mode!r}")

    # 4. 真打一次 /api/qa，看计数是否落库（端到端贯通）
    day = __import__("datetime").datetime.now().strftime("%Y-%m-%d")
    before = get_limiter().used_today("global:qa")
    resp = client.post(
        "/api/qa",
        json={"query": "作业什么时候截止", "top_k": 1},
        headers={"X-Forwarded-For": "203.0.113.250"},
    )
    after = get_limiter().used_today("global:qa")
    ok5 = check(
        "真实请求后计数落库",
        after == before + 1,
        f"/api/qa status={resp.status_code}，global:qa {before} → {after}（day={day}）",
    )

    # 5. 用完全独立的连接读库，排除「读的是内存」的可能
    with engine.connect() as conn:
        db_val = conn.execute(text(
            "SELECT value FROM rate_limit_counters WHERE day=:d AND key='global:qa'"
        ), {"d": day}).scalar()
    ok6 = check(
        "独立连接可从库中读到该计数",
        db_val == after,
        f"库中 = {db_val}，内存读数 = {after}",
    )

print()
for line in results:
    print(" ", line)
print()
allok = all(l.startswith("✅") for l in results)
print(f"总判定：{'✅ 全部通过 —— 生产路径确实在用 SQLite 持久化日计数' if allok else '❌ 存在失败项'}")
print("=" * 70)
sys.exit(0 if allok else 1)
