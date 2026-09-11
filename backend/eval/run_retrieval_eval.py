"""混合检索评测：向量 / BM25 / 混合三路对比 + 权重扫描。

用法：
    python -m eval.run_retrieval_eval                  # 默认 local_hash 向量（离线可复现）
    python -m eval.run_retrieval_eval --embedding dashscope   # 用真实 embedding（需 API Key）
    python -m eval.run_retrieval_eval --sweep          # 额外打印权重扫描表
    python -m eval.run_retrieval_eval --verbose        # 打印逐条命中明细

设计要点：
1. **自带语料**（retrieval_set.json），不连数据库 —— 评测结果可复现，
   不会因为本地库内容变化而漂移。
2. **复用生产代码**：向量用 VectorStore.index_text/search_vector，
   BM25 用 BM25Index，融合用 hybrid.fuse —— 评测的就是线上那条链路，
   而不是另写一份"评测专用"实现（那种做法测不出真实问题）。
3. 指标：Recall@1/@3/@5、MRR@5、nDCG@5（二值相关性）。
4. `--sweep` 扫描两路权重，用实测数据回答"权重该设多少"，
   而不是凭直觉拍一个数字。
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

from app.config import settings  # noqa: E402
from app.providers.embedding import get_embedding  # noqa: E402
from app.services.bm25 import BM25Index  # noqa: E402
from app.services.hybrid import fuse  # noqa: E402
from app.services.vector_store import VectorStore  # noqa: E402

SET_FILE = _HERE / "retrieval_set.json"
FETCH_K = 20
K_VALUES = (1, 3, 5)


# ---------------------------------------------------------------------------
# 指标
# ---------------------------------------------------------------------------
def recall_at_k(ranked: list[int], expected: set[int], k: int) -> float:
    return 1.0 if any(nid in expected for nid in ranked[:k]) else 0.0


def reciprocal_rank(ranked: list[int], expected: set[int]) -> float:
    for i, nid in enumerate(ranked, 1):
        if nid in expected:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked: list[int], expected: set[int], k: int) -> float:
    """二值相关性的 nDCG@k。"""
    dcg = sum(
        1.0 / math.log2(i + 1) for i, nid in enumerate(ranked[:k], 1) if nid in expected
    )
    ideal_hits = min(len(expected), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    return dcg / idcg if idcg else 0.0


# ---------------------------------------------------------------------------
# 索引构建（复用生产实现）
# ---------------------------------------------------------------------------
class Harness:
    def __init__(self, corpus: list[dict], embedder_name: str) -> None:
        if embedder_name == "dashscope":
            settings.embedding_provider = "dashscope"
        else:
            settings.embedding_provider = "local_hash"
        get_embedding.cache_clear()

        self.vector = VectorStore(embedder=get_embedding())
        self.bm25 = BM25Index()
        for doc in corpus:
            text = f"{doc['title']}\n{doc['text']}"
            self.vector.index_text(doc["id"], text)
            self.bm25.add(doc["id"], text)
        self.docs = {d["id"]: d for d in corpus}

    def ranked(self, query: str, mode: str, *, w_vector: float, w_bm25: float) -> list[int]:
        vec = self.vector.search_vector(self.vector.embed(query), FETCH_K)
        bm = self.bm25.search(query, FETCH_K)
        if mode == "vector":
            return [nid for nid, _ in vec]
        if mode == "bm25":
            return [nid for nid, _ in bm]
        return [
            h.notice_id
            for h in fuse(
                vec, bm, k=settings.hybrid_rrf_k, w_vector=w_vector, w_bm25=w_bm25
            )
        ]


def evaluate(
    harness: Harness,
    cases: list[dict],
    mode: str,
    *,
    w_vector: float,
    w_bm25: float,
    top_k: int = 5,
) -> dict:
    totals = {f"recall@{k}": 0.0 for k in K_VALUES}
    totals["mrr@5"] = 0.0
    totals["ndcg@5"] = 0.0
    details = []
    for case in cases:
        expected = set(case["expected"])
        ranked = harness.ranked(
            case["query"], mode, w_vector=w_vector, w_bm25=w_bm25
        )[:top_k]
        row = {
            "query": case["query"],
            "expected": sorted(expected),
            "ranked": ranked,
            "hit": bool(expected & set(ranked)),
            "rr": reciprocal_rank(ranked, expected),
        }
        details.append(row)
        for k in K_VALUES:
            totals[f"recall@{k}"] += recall_at_k(ranked, expected, k)
        totals["mrr@5"] += row["rr"]
        totals["ndcg@5"] += ndcg_at_k(ranked, expected, top_k)

    n = len(cases) or 1
    return {key: value / n for key, value in totals.items()} | {
        "details": details,
        "n": len(cases),
    }


# ---------------------------------------------------------------------------
# 输出
# ---------------------------------------------------------------------------
def fmt(metrics: dict) -> str:
    return (
        f"Recall@1={metrics['recall@1']:.3f}  "
        f"Recall@3={metrics['recall@3']:.3f}  "
        f"Recall@5={metrics['recall@5']:.3f}  "
        f"MRR@5={metrics['mrr@5']:.3f}  "
        f"nDCG@5={metrics['ndcg@5']:.3f}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="混合检索评测")
    parser.add_argument("--embedding", choices=["local_hash", "dashscope"], default="local_hash")
    parser.add_argument("--sweep", action="store_true", help="扫描两路权重")
    parser.add_argument("--verbose", action="store_true", help="打印逐条明细")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    data = json.loads(SET_FILE.read_text(encoding="utf-8"))
    corpus, cases = data["corpus"], data["cases"]

    harness = Harness(corpus, args.embedding)
    print("=" * 78)
    print(f"混合检索评测 | 语料 {len(corpus)} 条 | 查询 {len(cases)} 条")
    print(f"embedding = {harness.vector.embedder.active_name} | "
          f"BM25 词表 = {len(harness.bm25._postings)} | 平均文档长度 = {harness.bm25.avgdl:.1f}")
    print("=" * 78)

    default_wv = settings.hybrid_weight_vector
    default_wb = settings.hybrid_weight_bm25

    configs = [
        ("仅向量", "vector", 1.0, 0.0),
        ("仅 BM25", "bm25", 0.0, 1.0),
        (f"混合 (默认 {default_wv:.1f}:{default_wb:.1f})", "hybrid", default_wv, default_wb),
    ]

    results: dict[str, dict] = {}
    for label, mode, wv, wb in configs:
        m = evaluate(harness, cases, mode, w_vector=wv, w_bm25=wb, top_k=args.top_k)
        results[label] = m
        print(f"\n【{label}】")
        print(f"  {fmt(m)}")

    # 哪些查询只有 BM25 能救回来 / 只有向量能救回来
    vec_m = results["仅向量"]["details"]
    bm_m = results["仅 BM25"]["details"]
    hyb_m = results[f"混合 (默认 {default_wv:.1f}:{default_wb:.1f})"]["details"]

    only_bm25 = [
        v["query"] for v, b in zip(vec_m, bm_m) if not v["hit"] and b["hit"]
    ]
    only_vec = [
        v["query"] for v, b in zip(vec_m, bm_m) if v["hit"] and not b["hit"]
    ]
    fixed = [
        (v["query"], vec_m[i]["hit"], bm_m[i]["hit"], hyb_m[i]["hit"])
        for i, v in enumerate(hyb_m)
        if not vec_m[i]["hit"] and hyb_m[i]["hit"]
    ]
    broken = [
        v["query"] for i, v in enumerate(hyb_m) if vec_m[i]["hit"] and not hyb_m[i]["hit"]
    ]

    print("\n" + "-" * 78)
    print("两路互补性分析")
    print("-" * 78)
    print(f"  仅 BM25 召回成功（向量漏掉）: {len(only_bm25)} 条")
    for q in only_bm25:
        print(f"      · {q}")
    print(f"  仅向量召回成功（BM25 漏掉）: {len(only_vec)} 条")
    for q in only_vec:
        print(f"      · {q}")
    print(f"  混合把向量的失败救回        : {len(fixed)} 条")
    for q, _, _, _ in fixed:
        print(f"      · {q}")
    print(f"  混合相比向量反而失败        : {len(broken)} 条")
    for q in broken:
        print(f"      ⚠ {q}")

    if args.sweep:
        print("\n" + "-" * 78)
        print("权重扫描（w_bm25 = 1 - w_vector）")
        print("-" * 78)
        print(f"  {'w_vec':>6} {'w_bm25':>7} | {'Recall@1':>9} {'Recall@3':>9} {'MRR@5':>8} {'nDCG@5':>8}")
        best = None
        for i in range(0, 11):
            wv = round(i / 10, 1)
            wb = round(1.0 - wv, 1)
            m = evaluate(harness, cases, "hybrid", w_vector=wv, w_bm25=wb, top_k=args.top_k)
            marker = ""
            if best is None or m["mrr@5"] > best[1]["mrr@5"]:
                best = ((wv, wb), m)
                marker = ""
            print(
                f"  {wv:>6.1f} {wb:>7.1f} | {m['recall@1']:>9.3f} {m['recall@3']:>9.3f} "
                f"{m['mrr@5']:>8.3f} {m['ndcg@5']:>8.3f}{marker}"
            )
        assert best is not None
        (bwv, bwb), bm = best
        print(f"\n  MRR@5 最优: w_vector={bwv}  w_bm25={bwb}  →  {fmt(bm)}")
        print(f"  当前配置    : w_vector={default_wv}  w_bm25={default_wb}")

    if args.verbose:
        print("\n" + "-" * 78)
        print("逐条明细（混合检索）")
        print("-" * 78)
        for row in hyb_m:
            flag = "✓" if row["hit"] else "✗"
            print(f"  {flag} {row['query']}")
            print(f"      期望={row['expected']}  实际={row['ranked']}")

    print("\n" + "=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
