"""#8/#9 真实多进程端到端验证。

单元测试里用「同一进程内多个 RateLimiter 实例」模拟多 worker，
但那毕竟还是共享一个 Python 进程内存。这里用**真正独立的子进程**
（各自 import、各自单例、只通过 SQLite 文件通信）来验证：

  场景 A（#8 多 worker）：N 个子进程并发抢同一个每日额度，
      放行总数必须恰好等于额度 —— 不多不少。
  场景 B（#9 重启清零）：先跑一个子进程把额度用光，进程完全退出后
      再起一个新进程，额度必须仍是「已用光」。

用法（在 backend/ 目录下）：
    python scripts/verify_rate_limit_multiproc.py
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

# 本文件位于 backend/scripts/，故 backend 根目录是上两级
BACKEND = Path(__file__).resolve().parent.parent

# 子进程要执行的脚本：抢额度，把「放行了几次」打印到 stdout
WORKER_SRC = r'''
import os, sys
sys.path.insert(0, os.environ["CA_BACKEND"])
os.environ["DATABASE_URL"] = os.environ["CA_DB"]
os.environ["RATE_LIMIT_STORE"] = "sqlite"
# 只留全局日额度，排除令牌桶干扰
os.environ["RATE_LIMIT_QA_PER_MIN"] = "0"
os.environ["RATE_LIMIT_QA_PER_IP_DAY"] = "0"
os.environ["RATE_LIMIT_QA_PER_DAY"] = os.environ["CA_QUOTA"]

from app.config import settings
from app.db import engine
from app.services.rate_limit import RateLimiter
from app.services.rate_limit_store import SqliteCountStore

assert settings.rate_limit_qa_per_day == int(os.environ["CA_QUOTA"]), \
    f"额度未生效: {settings.rate_limit_qa_per_day}"

lim = RateLimiter(store=SqliteCountStore(engine))
allowed = 0
total = int(os.environ["CA_ATTEMPTS"])
for i in range(total):
    if lim.check("/api/qa", f"10.0.{i // 256}.{i % 256}").allowed:
        allowed += 1
print(f"ALLOWED={allowed} ATTEMPTS={total}")
'''


def run_worker(env_extra: dict[str, str], timeout: int = 120) -> str:
    env = dict(os.environ)
    env.update({
        "CA_BACKEND": str(BACKEND),
        "PYTHONPATH": str(BACKEND),
        "PYTHONIOENCODING": "utf-8",
    })
    env.update(env_extra)
    proc = subprocess.run(
        [sys.executable, "-c", WORKER_SRC],
        capture_output=True, text=True, env=env, timeout=timeout,
        encoding="utf-8", errors="replace",
    )
    if proc.returncode != 0:
        raise RuntimeError(f"子进程失败 rc={proc.returncode}\nSTDOUT:{proc.stdout}\nSTDERR:{proc.stderr}")
    for line in proc.stdout.splitlines():
        if line.startswith("ALLOWED="):
            return line.strip()
    raise RuntimeError(f"未找到结果行\nSTDOUT:{proc.stdout}\nSTDERR:{proc.stderr}")


def parse(line: str) -> tuple[int, int]:
    d = dict(kv.split("=") for kv in line.split())
    return int(d["ALLOWED"]), int(d["ATTEMPTS"])


def main() -> int:
    tmpdir = Path(tempfile.mkdtemp(prefix="rl_e2e_"))
    db_path = tmpdir / "e2e.db"
    db_url = f"sqlite:///{db_path.as_posix()}"

    print("=" * 70)
    print("场景 A（#8 多 worker）：4 个真实子进程抢同一个每日额度 40")
    print("=" * 70)
    quota, workers, attempts = 40, 4, 30
    procs = []
    for _w in range(workers):
        env = dict(os.environ)
        env.update({
            "CA_BACKEND": str(BACKEND), "PYTHONPATH": str(BACKEND),
            "PYTHONIOENCODING": "utf-8",
            "CA_DB": db_url, "CA_QUOTA": str(quota), "CA_ATTEMPTS": str(attempts),
        })
        procs.append(subprocess.Popen(
            [sys.executable, "-c", WORKER_SRC],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, env=env, encoding="utf-8", errors="replace",
        ))

    allowed_total = 0
    for i, p in enumerate(procs):
        out, err = p.communicate(timeout=180)
        if p.returncode != 0:
            print(f"  worker{i} 失败 rc={p.returncode}\n  {err[-800:]}")
            return 1
        line = next((ln for ln in out.splitlines() if ln.startswith("ALLOWED=")), "")
        a, t = parse(line)
        allowed_total += a
        print(f"  worker{i}: 尝试 {t} 次，放行 {a} 次")

    # 直接查库确认落盘总量
    import sqlite3
    conn = sqlite3.connect(db_path)
    row = conn.execute(
        "SELECT value FROM rate_limit_counters WHERE key='global:qa'"
    ).fetchone()
    conn.close()
    db_total = row[0] if row else 0

    print(f"\n  放行总数 = {allowed_total}（期望恰好 {quota}）")
    print(f"  库中 global:qa = {db_total}（期望恰好 {quota}）")
    ok_a = (allowed_total == quota and db_total == quota)
    print(f"  → {'✅ 通过：多 worker 额度精确，无超发' if ok_a else '❌ 失败：超发或丢更新'}")

    print()
    print("=" * 70)
    print("场景 B（#9 重启清零）：额度用光后，另起全新进程验证仍被拦住")
    print("=" * 70)
    db2 = tmpdir / "restart.db"
    env_common = {
        "CA_DB": f"sqlite:///{db2.as_posix()}",
        "CA_QUOTA": str(quota), "CA_ATTEMPTS": str(quota + 10),
    }
    line1 = run_worker(env_common)
    a1, t1 = parse(line1)
    print(f"  第 1 个进程（进程号不同、内存全新）：尝试 {t1} 次，放行 {a1} 次")

    # 第 2 个进程完全独立；换一批 IP（证明是全局额度而非 IP 额度在起作用）
    line2 = run_worker(env_common)
    a2, _ = parse(line2)
    print(f"  第 2 个进程（模拟重启）：放行 {a2} 次")
    ok_b = (a1 == quota and a2 == 0)
    print(f"  → {'✅ 通过：重启后额度未清零' if ok_b else '❌ 失败：重启把额度清零了（#9 未修复）'}")

    print()
    print("=" * 70)
    verdict = ok_a and ok_b
    print(f"总判定：{'✅ 场景 A 与 B 全部通过' if verdict else '❌ 存在失败场景'}")
    print(f"（临时库位置：{tmpdir}）")
    print("=" * 70)
    return 0 if verdict else 1


if __name__ == "__main__":
    sys.exit(main())
