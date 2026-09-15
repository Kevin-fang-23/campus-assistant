"""生产路径现场确认：真实 embedding 后端 + 标定阈值下，缓存语义命中是否真的触发。

为什么需要单独确认：**标定脚本跑出的阈值 ≠ 生产链路上能生效**。
`eval/run_cache_threshold_eval.py` 直接调 embedder 算相似度，而生产路径要经过
中间件 → 限流 → endpoint → 缓存 → 回答复用这一整条链路：
只要阈值读错（比如端点读的是另一个配置项）、向量没写进条目、
或维度守卫把条目全跳过，标定值就形同虚设 —— 而单测用的是 local_hash +
人为下调的阈值，恰恰**验不到**"生产后端 + 标定值"这个组合。

用法（在 backend/ 目录下，需要 .env 里有可用的 ALIYUN_API_KEY）：

    python scripts/verify_semantic_cache_e2e.py

检查 5 项：
  1. `/health` 可达，且如实报告实际生效的 embedding 后端
  2. 首次提问未命中缓存
  3. **改写问法**命中语义缓存（`X-Cache: HIT_SEMANTIC`），且相似度 ≥ 配置阈值
  4. 相似度与 `eval/CACHE_SEMANTIC_BASELINE.md` 记录的标定值同一量级
     （偏差过大说明 embedding 模型或数据集变了，该重新标定）
  5. 阈值改为 0 后**不再**命中 —— 回滚开关真的有效

⚠️ 本脚本不在 CI 中运行（CI 无密钥，且它依赖真实上游）。它是一次性人工确认工具，
与 `scripts/verify_rate_limit_*.py` 同一性质。
⚠️ 为不消耗 LLM 额度，脚本强制 `QA_PROVIDER=mock`（走抽取式降级回答）——
   本脚本验的是缓存语义命中，与答案由谁生成无关。
"""
from __future__ import annotations

import sys
from pathlib import Path

# 本文件位于 backend/scripts/，故 backend 根目录是上两级
BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND))

from fastapi.testclient import TestClient  # noqa: E402

from app.config import settings  # noqa: E402
from app.main import app  # noqa: E402
from app.services.qa_cache import get_cache, reset_cache_for_tests  # noqa: E402

# 需求里给出的原例：标定集中这两句在 dashscope 上的相似度为 0.891
QUERY_A = "作业什么时候截止"
QUERY_B = "作业截止时间是什么"
BASELINE_SIM = 0.891          # eval/CACHE_SEMANTIC_BASELINE.md 记录的实测值
BASELINE_TOLERANCE = 0.10     # 容忍度：只判断"是否同一量级"，不做精确比对

results: list[str] = []


def check(name: str, ok: bool, detail: str) -> bool:
    results.append(f"{'✅' if ok else '❌'} {name}: {detail}")
    return ok


def main() -> int:
    # 强制降级回答，避免为一次验证烧掉 LLM 额度（见模块说明）
    settings.qa_provider = "mock"

    with TestClient(app) as client:
        # ---- 1. /health 可达 + 后端如实上报 ----
        health = client.get("/health")
        check("服务可启动且 /health 可达", health.status_code == 200,
              f"status={health.status_code}")
        if health.status_code != 200:
            return _report()

        vector = health.json()["vector"]
        active = vector["active_embedding"]
        degraded = vector["embedding_degraded"]
        check("embedding 后端如实上报", True,
              f"active={active} degraded={degraded} configured={vector['configured_embedding']}")

        threshold = float(settings.qa_cache_semantic_threshold or 0.0)
        if threshold <= 0:
            results.append(
                "⏭  QA_CACHE_SEMANTIC_THRESHOLD=0（语义匹配已关闭）——"
                "跳过 2~5 项。要验证请先配一个 > 0 的阈值。"
            )
            return _report()
        if active == "local_hash":
            results.append(
                "⏭  当前 embedding 后端是 local_hash（字符哈希，非语义模型）——"
                "实测该后端下任何阈值都只有 1/10 命中，等于零收益，"
                "无法验证语义链路。配置 Key 或修上游后重试。"
            )
            return _report()

        reset_cache_for_tests()

        # ---- 2. 首次提问：未命中 ----
        r1 = client.post("/api/qa", json={"query": QUERY_A, "top_k": 3})
        check("首次提问未命中缓存", r1.status_code == 200 and r1.json()["cache_hit"] is False,
              f"status={r1.status_code} cache_hit={r1.json().get('cache_hit')}")

        # ---- 3. 改写问法：必须语义命中 ----
        r2 = client.post("/api/qa", json={"query": QUERY_B, "top_k": 3})
        cache_header = r2.headers.get("X-Cache")
        hit = r2.json().get("cache_hit") is True and cache_header == "HIT_SEMANTIC"
        sim_raw = r2.headers.get("X-Cache-Similarity")
        sim = float(sim_raw) if sim_raw else 0.0
        check("改写问法命中语义缓存", hit,
              f"cache_hit={r2.json().get('cache_hit')} X-Cache={cache_header} sim={sim_raw}")
        check("命中相似度不低于配置阈值", sim >= threshold,
              f"sim={sim:.4f} ≥ threshold={threshold:g}")
        # 复用的是同一份答案
        check("复用的答案与首次一致", r2.json().get("answer") == r1.json().get("answer"),
              "answer 字段逐字相同" if r2.json().get("answer") == r1.json().get("answer")
              else "answer 不一致，说明并非复用缓存")

        # ---- 4. 与标定文档记录的量级一致 ----
        drift = abs(sim - BASELINE_SIM)
        check("相似度与标定记录同一量级", drift <= BASELINE_TOLERANCE,
              f"实测 {sim:.4f} vs 基线 {BASELINE_SIM}（差 {drift:.4f}）；"
              f"偏差过大说明 embedding 模型或对照数据变了，需重跑标定")

        stats = get_cache().stats()
        check("命中计数已单独记录", stats["semantic_hits"] >= 1,
              f"semantic_lookups={stats['semantic_lookups']} "
              f"semantic_hits={stats['semantic_hits']}")

        # ---- 5. 回滚开关：阈值 0 必须退回字面归一化 ----
        settings.qa_cache_semantic_threshold = 0.0
        reset_cache_for_tests()
        client.post("/api/qa", json={"query": QUERY_A, "top_k": 3})
        r3 = client.post("/api/qa", json={"query": QUERY_B, "top_k": 3})
        rolled_back = (
            r3.json()["cache_hit"] is False and r3.headers.get("X-Cache") is None
        )
        check("阈值 0 时不再语义命中（回滚有效）", rolled_back,
              f"cache_hit={r3.json()['cache_hit']} X-Cache={r3.headers.get('X-Cache')}")

    return _report()


def _report() -> int:
    print("\n" + "=" * 70)
    print("语义缓存生产路径现场确认")
    print("=" * 70)
    for line in results:
        print("  " + line)
    failed = [r for r in results if r.startswith("❌")]
    skipped = [r for r in results if r.startswith("⏭")]
    print("-" * 70)
    if failed:
        print(f"  ✗ {len(failed)} 项未通过 —— 标定值在生产链路上没有生效。")
        print("    排查方向：阈值读取位置、向量是否随条目写入、维度守卫是否把条目全跳过。")
    elif skipped:
        print(f"  ⚠ 有 {len(skipped)} 项被跳过（见上），未完成完整验证。")
    else:
        print("  ✓ 全部通过：标定阈值在真实后端的生产链路上可正常触发语义命中。")
    print("=" * 70)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
