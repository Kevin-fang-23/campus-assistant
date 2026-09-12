"""混合检索评测：向量 / BM25 / 混合三路对比 + 权重扫描 + 相关性阈值标定 + 质量门禁。

用法：
    python -m eval.run_retrieval_eval                  # 默认 local_hash 向量（离线可复现）
    python -m eval.run_retrieval_eval --embedding dashscope   # 用真实 embedding（需 API Key）
    python -m eval.run_retrieval_eval --sweep          # 额外打印权重扫描表
    python -m eval.run_retrieval_eval --calibrate      # 标定「相关性阈值」（见下）
    python -m eval.run_retrieval_eval --gate           # 与 baseline.json 比对，不达标退出码 1
    python -m eval.run_retrieval_eval --update-baseline # 把当前指标写回基线（慎用）
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
5. `--calibrate` 标定相关性阈值：用两组**有标注**的查询夹出可行区间 ——
   噪声查询（本无答案）的最高分是上界，真实查询命中项的最低分是下界。
   阈值只能落在两者之间，**不是拍脑袋定的**。
6. `--gate` 把指标与 `baseline.json` 比对，低于"基线 - 容差"即以非零码退出 ——
   让 CI 能拦住"检索质量静默劣化"。这是本脚本从"手动工具"变成
   "质量门禁"的关键：没有它，改了分词/权重后指标下滑无人察觉。
7. 评测默认用 **local_hash** 向量：确定性、离线、毫秒级，
   适合当 CI 门禁基准；dashscope 受服务端状态影响会波动，
   只适合人工对照，不适合卡门禁。
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
from app.services.snippet import score_sentence, split_sentences  # noqa: E402
from app.services.vector_store import VectorStore  # noqa: E402

SET_FILE = _HERE / "retrieval_set.json"
BASELINE_FILE = _HERE / "baseline.json"
FETCH_K = 20
K_VALUES = (1, 3, 5)

# 门禁容差：不要求与基线完全持平 —— 浮点误差、依赖小版本差异都可能带来
# 千分位波动，卡零会让 CI 频繁假红，反而失去意义。
GATE_TOLERANCE = 0.005


# ---------------------------------------------------------------------------
# 文本构造：必须与生产一致，否则标定出的权重不可迁移
# ---------------------------------------------------------------------------
def build_text(doc: dict) -> str:
    """向量路文本 —— 等价生产的 `VectorStore.build_text`（只用结构化字段）。

    生产里向量索引的是抽取后的结构化信息（标题/摘要/课程/地点/标签），
    不含正文：语义检索吃提炼过的信息更干净。
    """
    parts = [
        doc.get("title") or "",
        doc.get("summary") or "",
        doc.get("course") or "",
        doc.get("location") or "",
        " ".join(doc.get("tags") or []),
    ]
    return "\n".join(p for p in parts if p)


def bm25_text(doc: dict) -> str:
    """BM25 路文本 —— 等价生产的 `hybrid.bm25_text`（结构化字段 + 正文）。

    BM25 的价值是精确串匹配，正文里的电话/房间号/邮箱必须可搜，
    因此它的索引文本比向量路多一份正文。**两路文本本就不必相同** ——
    RRF 只用排名，不需要两路分数可比。
    """
    return "\n".join(p for p in (build_text(doc), doc.get("text") or "") if p)


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
            # 两路用各自生产等价的文本（详见 build_text / bm25_text 的说明）
            self.vector.index_text(doc["id"], build_text(doc))
            self.bm25.add(doc["id"], bm25_text(doc))
        self.docs = {d["id"]: d for d in corpus}
        # 两路候选缓存：权重扫描会对同一批查询反复求排名，而候选集合
        #（向量 top-K 与 BM25 top-K）**与权重无关**。不缓存就要重复调用
        # embedding —— 21 个权重 × 129 条 ≈ 2700 次 API 调用，真实后端会直接超时。
        self._cand: dict[str, tuple[list, list]] = {}

    def candidates(self, query: str) -> tuple[list, list]:
        """取该查询的两路候选（带缓存）。"""
        if query not in self._cand:
            self._cand[query] = (
                self.vector.search_vector(self.vector.embed(query), FETCH_K),
                self.bm25.search(query, FETCH_K),
            )
        return self._cand[query]

    def ranked(self, query: str, mode: str, *, w_vector: float, w_bm25: float) -> list[int]:
        vec, bm = self.candidates(query)
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

    def ranked_hits(self, query: str, *, w_vector: float, w_bm25: float) -> list:
        """返回融合后的完整 RetrievalHit 列表（带 score_bm25 等分量分）。

        阈值标定必须看到**分量分**：展示用的归一化 `score` 由 RRF 排名算出，
        无关查询也必然产生排名，因此它**无法区分「唯一命中」与「唯一且相关」**。
        真正可当门限的是 BM25 的绝对分（或余弦的绝对相似度）——
        它们不随"是否还有别的候选"漂移。
        """
        vec, bm = self.candidates(query)
        return fuse(
            vec, bm, k=settings.hybrid_rrf_k, w_vector=w_vector, w_bm25=w_bm25
        )


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
# 相关性阈值标定（#6b）
# ---------------------------------------------------------------------------
# 标定两个候选判据，看哪个能把两类查询分开：
#   · bm25  —— 该文档的原始 BM25 分（None 表示未被 BM25 路召回）
#   · cos   —— 该文档的余弦相似度（None 表示未被向量路召回）
# 缺失值按 sentinel 处理：某文档只被一路召回时，另一路对它"没有意见"，
# 不能当成 0 参与比较（那等于用一路的缺失去惩罚它）。
_MISSING = float("-inf")


def _signal(hit, key: str) -> float:  # noqa: ANN001
    value = hit.score_bm25 if key == "bm25" else hit.score_vector
    return _MISSING if value is None else float(value)


def _snippet_signal(harness: "Harness", query: str, notice_id: int) -> float:
    """第三个候选判据：句级选片打分（`snippet.score_sentence` 的**最高句分**）。

    为什么试这个：前两个判据（BM25 绝对分、余弦）都因两类区间重叠而失败 ——
    它们衡量的是"词/语义是否出现在文档里"，而**无关查询也总会撞上一两个字**
    （"今天天气怎么样" 撞 "今天"、"量子纠缠" 撞 "子"）。

    而句级选片打分是**刻意设计过的**：它用 bigram 覆盖率而非单字计数
    （`_W_BIGRAM=2.0` vs `_W_UNIGRAM=0.6`），且 `_MIN_RELEVANCE=3.0` 这条
    现成的门槛就是为"排除单字重合噪声"而设的。它在生产里已经承担了
    「这句话是不是真答案」的判定，因此最有希望区分「无关」与「相关」。

    取**最高句分**而不是整篇分：只要文档里有一句真相关，这条就该留下。
    """
    doc = harness.docs.get(notice_id)
    if not doc:
        return _MISSING
    text = bm25_text(doc)  # 与生产 BM25 路的文本一致（结构化 + 正文）
    sentences = split_sentences(text)
    if not sentences:
        return 0.0
    return max(score_sentence(s, query, title=doc.get("title")) for s in sentences)


def calibrate(
    harness: Harness,
    cases: list[dict],
    noise_cases: list[dict],
    *,
    w_vector: float,
    w_bm25: float,
    top_k: int = 5,
) -> dict:
    """夹出相关性阈值的可行区间。

    思路：阈值要同时满足两条互相冲突的要求 ——
      1. 噪声查询（语料里确实没有答案）**不应**有结果通过阈值 → 需要"够高"；
      2. 真实查询的正确答案第一名**必须**通过阈值 → 需要"够低"。

    因此取噪声查询里的**最高分**作为上界、真实查询命中项的**最低分**作为下界。
    若下界 > 上界，说明存在一个干净的分离点，阈值可落在区间内；
    若下界 ≤ 上界，说明该判据**没有判别力**（两类查询的分数区间重叠），
    如实报告而不是硬凑一个数 —— 那种阈值只会在生产上误杀或漏放。
    """
    # --- 上界：噪声查询各自返回的 top-k 里的最高分 ---
    noise_rows: list[dict] = []
    for case in noise_cases:
        hits = harness.ranked_hits(
            case["query"], w_vector=w_vector, w_bm25=w_bm25
        )[:top_k]
        snip = [
            _snippet_signal(harness, case["query"], h.notice_id) for h in hits
        ]
        snip = [s for s in snip if s != _MISSING]
        noise_rows.append(
            {
                "query": case["query"],
                "bm25_max": max(
                    (
                        _signal(h, "bm25")
                        for h in hits
                        if _signal(h, "bm25") != _MISSING
                    ),
                    default=None,
                ),
                "cos_max": max(
                    (
                        _signal(h, "cos")
                        for h in hits
                        if _signal(h, "cos") != _MISSING
                    ),
                    default=None,
                ),
                "snip_max": max(snip) if snip else None,
                "n_hits": len(hits),
            }
        )

    # --- 下界：真实查询里，正确答案在 top-k 内的最低分 ---
    hit_rows: list[dict] = []
    for case in cases:
        expected = set(case["expected"])
        hits = harness.ranked_hits(
            case["query"], w_vector=w_vector, w_bm25=w_bm25
        )[:top_k]
        correct = [h for h in hits if h.notice_id in expected]
        if not correct:
            # 本来就召回失败的查询不该参与标定 —— 用失败样本定阈值没有意义
            continue
        snip = [
            _snippet_signal(harness, case["query"], h.notice_id) for h in correct
        ]
        snip = [s for s in snip if s != _MISSING]
        hit_rows.append(
            {
                "query": case["query"],
                "bm25_min": min(
                    (
                        _signal(h, "bm25")
                        for h in correct
                        if _signal(h, "bm25") != _MISSING
                    ),
                    default=None,
                ),
                "cos_min": min(
                    (
                        _signal(h, "cos")
                        for h in correct
                        if _signal(h, "cos") != _MISSING
                    ),
                    default=None,
                ),
                # 用 max 而非 min：正确答案的任一片段足够相关即可通过
                "snip_min": max(snip) if snip else None,
            }
        )

    def _bounds(key: str) -> tuple[float | None, float | None]:
        upper = max(
            (r[f"{key}_max"] for r in noise_rows if r[f"{key}_max"] is not None),
            default=None,
        )
        lower = min(
            (r[f"{key}_min"] for r in hit_rows if r[f"{key}_min"] is not None),
            default=None,
        )
        return upper, lower

    return {
        "noise_rows": noise_rows,
        "hit_rows": hit_rows,
        "bm25": _bounds("bm25"),
        "cos": _bounds("cos"),
        "snip": _bounds("snip"),
    }


# ---------------------------------------------------------------------------
# 质量门禁（#7）：把指标与基线比对，劣化即 fail
# ---------------------------------------------------------------------------
# 门禁要回答的问题只有一个：**这次改动有没有让检索变差**。
# 因此不追求绝对高分，只守住"不低于已记录的基线 - 容差"。
#
# 为什么必须有它：检索质量是**静默劣化**的典型场景 —— 改了分词、调了权重、
# 换了 prompt 或 embedding 后端，单元测试全绿（它们测的是结构与边界），
# 但 MRR 从 0.9 掉到 0.7 没有任何人会发现，直到用户抱怨"搜不到了"。
def load_baseline() -> dict:
    """读取基线文件。缺失即报错退出 —— 没有基线就谈不上门禁。"""
    if not BASELINE_FILE.exists():
        raise SystemExit(
            f"缺少基线文件 {BASELINE_FILE}。\n"
            "首次建立基线：python -m eval.run_retrieval_eval "
            "--update-baseline --embedding local_hash"
        )
    return json.loads(BASELINE_FILE.read_text(encoding="utf-8"))


def check_gate(baseline: dict, current: dict, *, tolerance: float) -> list[str]:
    """比对本次指标与基线，返回**劣化项**的描述列表（空列表 = 通过）。

    只查"变差"不查"变好"：指标上升是好事，不该因为与基线不一致而失败
    （否则任何人做了改进都要先改基线才能过 CI，反而阻碍优化）。
    """
    failures: list[str] = []
    base_metrics = baseline.get("metrics", {})
    for key, base_value in base_metrics.items():
        if key == "n":
            continue  # 样本量不参与门禁（见 main() 里的说明）
        if key not in current:
            failures.append(f"缺少指标 {key}（基线有、本次无）")
            continue
        now = current[key]
        floor = float(base_value) - tolerance
        if now < floor:
            failures.append(
                f"{key}: {now:.4f} < 基线 {float(base_value):.4f} "
                f"(容差 {tolerance:.3f}，下限 {floor:.4f})"
            )
    return failures


def build_baseline_payload(
    *, embedding: str, metrics: dict, weights: dict, n_corpus: int, n_cases: int
) -> dict:
    """构造基线文件内容。

    记录 embedding 后端与权重：**指标只有在同一条链路下才可比**。
    换了 embedding 或权重后 MRR 必然变化，那不是"劣化"，不能触发门禁 ——
    所以这两项一并写入，供人核对（更换时需重新 `--update-baseline`）。
    """
    return {
        "_comment": (
            "检索质量基线，供 CI 门禁比对。"
            "指标低于本文件记录值（容差见 run_retrieval_eval.GATE_TOLERANCE）即 fail。"
            "更新方式：python -m eval.run_retrieval_eval --update-baseline "
            "--embedding local_hash（权重/embedding 变更时需重新标定）。"
        ),
        "embedding": embedding,
        "weights": weights,
        "corpus_size": n_corpus,
        "case_count": n_cases,
        "metrics": metrics,
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
    parser.add_argument("--calibrate", action="store_true", help="标定相关性阈值（#6b）")
    parser.add_argument("--gate", action="store_true", help="与基线比对，不达标退出码 1（#7）")
    parser.add_argument(
        "--update-baseline",
        action="store_true",
        help="把本次指标写回 baseline.json（仅在确认指标变化可接受时使用）",
    )
    parser.add_argument("--verbose", action="store_true", help="打印逐条明细")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    data = json.loads(SET_FILE.read_text(encoding="utf-8"))
    corpus, all_cases = data["corpus"], data["cases"]

    # 「无答案」查询（expected 里没有语料 id）单独统计。
    # 它们永远不可能命中，若混进主指标，只会给所有配置一律加 0，
    # 把差异稀释掉 —— 指标就失去判别力了。
    corpus_ids = {d["id"] for d in corpus}
    cases = [c for c in all_cases if set(c["expected"]) & corpus_ids]
    noise_cases = [c for c in all_cases if not (set(c["expected"]) & corpus_ids)]

    harness = Harness(corpus, args.embedding)
    print("=" * 78)
    print(
        f"混合检索评测 | 语料 {len(corpus)} 条 | 查询 {len(cases)} 条"
        f"（另有 {len(noise_cases)} 条无答案噪声查询，不计入主指标）"
    )
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

    # 「无答案」查询：验证相关性阈值能否拦住它们
    if noise_cases:
        print("\n" + "-" * 78)
        print("无答案查询（语料里本就没有相关内容）")
        print("-" * 78)
        for case in noise_cases:
            ranked = harness.ranked(
                case["query"], "hybrid", w_vector=default_wv, w_bm25=default_wb
            )[:3]
            print(f"  · {case['query']}")
            for nid in ranked:
                doc = harness.docs.get(nid, {})
                print(f"      → #{nid} {(doc.get('title') or '?')[:36]}")
        print("  说明：检索本身恒返回 top-k；是否提示「未找到相关内容」由")
        print("       `SEARCH_MIN_BIGRAM_OVERLAP` 阈值决定（见 --calibrate 与")
        print("       app/services/hybrid.py::filter_by_relevance）。")
        print("       上表是**未过阈值**的原始召回，用于观察阈值要拦掉什么。")

    if args.calibrate:
        cal = calibrate(
            harness, cases, noise_cases, w_vector=default_wv, w_bm25=default_wb
        )
        print("\n" + "-" * 78)
        print("相关性阈值标定（用标注数据夹出可行区间，而不是拍一个数）")
        print("-" * 78)

        print("\n  【上界】噪声查询（语料本无答案）返回的最高分 —— 阈值必须高于它：")
        print(f"    {'查询':<22} {'BM25 最高':>10} {'余弦 最高':>10} {'选片 最高':>10}")
        for row in cal["noise_rows"]:
            b = "—" if row["bm25_max"] is None else f"{row['bm25_max']:.3f}"
            c = "—" if row["cos_max"] is None else f"{row['cos_max']:.3f}"
            s = "—" if row["snip_max"] is None else f"{row['snip_max']:.3f}"
            print(f"    {row['query']:<22} {b:>10} {c:>10} {s:>10}")

        print("\n  【下界】真实查询正确答案的最低分 —— 阈值必须低于它：")
        print(f"    {'查询':<22} {'BM25 最低':>10} {'余弦 最低':>10} {'选片 最低':>10}")
        # 只展示最接近上界的若干条：它们才是真正约束阈值的样本
        for row in sorted(
            cal["hit_rows"],
            key=lambda r: (r["snip_min"] is None, r["snip_min"] or 0.0),
        )[:8]:
            b = "—" if row["bm25_min"] is None else f"{row['bm25_min']:.3f}"
            c = "—" if row["cos_min"] is None else f"{row['cos_min']:.3f}"
            s = "—" if row["snip_min"] is None else f"{row['snip_min']:.3f}"
            print(f"    {row['query']:<22} {b:>10} {c:>10} {s:>10}")
        print(f"    （共 {len(cal['hit_rows'])} 条成功召回的查询参与标定，此处只列最低 8 条）")

        print("\n  【结论】")
        for key, name in (
            ("bm25", "BM25 绝对分"),
            ("cos", "余弦相似度"),
            ("snip", "句级选片最高分"),
        ):
            upper, lower = cal[key]
            if upper is None or lower is None:
                print(f"    {name}: 样本不足，无法标定（上界={upper} 下界={lower}）")
                continue
            if lower > upper:
                mid = (upper + lower) / 2
                print(f"    {name}: 存在分离点")
                print(f"       上界(噪声最高)={upper:.3f}  <  下界(命中最低)={lower:.3f}")
                print(f"       建议阈值区间 ({upper:.3f}, {lower:.3f})，取中点 {mid:.3f}")
            else:
                print(f"    {name}: 两类区间重叠，该判据**没有判别力**")
                print(f"       上界(噪声最高)={upper:.3f}  >=  下界(命中最低)={lower:.3f}")
                print("       说明：单纯靠这个分数无法把「无关」与「相关」分开。")

        # 为什么失败：把「上界最大」的那条噪声查询与「下界最小」的真实查询并排看，
        # 差异是**定性**的 —— 噪声查询的 BM25 命中的是极短的无关文档，
        # 长度归一化把小文档的分数抬高了。这是 BM25 的固有性质，不是 bug。
        print("\n  【为什么重叠】最坏噪声样本 vs 最好真实样本（BM25 口径）：")
        hard_noise = max(
            (r for r in cal["noise_rows"] if r["bm25_max"] is not None),
            key=lambda r: r["bm25_max"],
            default=None,
        )
        weak_hit = min(
            (r for r in cal["hit_rows"] if r["bm25_min"] is not None),
            key=lambda r: r["bm25_min"],
            default=None,
        )
        if hard_noise:
            print(f"    噪声(最高) {hard_noise['query']} -> BM25 {hard_noise['bm25_max']:.3f}")
        if weak_hit:
            print(f"    真实(最低) {weak_hit['query']} -> BM25 {weak_hit['bm25_min']:.3f}")
        print("    两者分数量级相同 → 用同一个阈值必然误杀或漏放。")

    if args.sweep:
        print("\n" + "-" * 78)
        print("权重扫描（w_bm25 = 1 - w_vector，步长 0.05）")
        print("-" * 78)
        print(f"  {'w_vec':>6} {'w_bm25':>7} | {'Recall@1':>9} {'Recall@3':>9} {'MRR@5':>8} {'nDCG@5':>8}")
        best = None
        # 步长取 0.05 而非 0.1：实测发现低向量权重段差异细密，
        # 0.1 的颗粒度看不出「代价何时变得可忽略」。
        steps = [round(i * 0.05, 2) for i in range(21)]
        for wv in steps:
            wb = round(1.0 - wv, 2)
            m = evaluate(harness, cases, "hybrid", w_vector=wv, w_bm25=wb, top_k=args.top_k)
            marker = ""
            if best is None or m["mrr@5"] > best[1]["mrr@5"]:
                best = ((wv, wb), m)
                marker = ""
            print(
                f"  {wv:>6.2f} {wb:>7.2f} | {m['recall@1']:>9.3f} {m['recall@3']:>9.3f} "
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

    # ---- 质量门禁 / 基线更新（#7）----
    # 门禁只针对**默认混合配置**的指标：其他两行（仅向量/仅 BM25）是诊断用的
    # 对照项，把它们也纳入门禁会让"为了看互补性而跑一次评测"变得容易误判。
    hybrid_label = f"混合 (默认 {default_wv:.1f}:{default_wb:.1f})"
    # 只把**质量指标**纳入门禁。`n` 是参与评测的查询条数（样本量而非质量），
    # 把它一起卡住会导致"新增/删减查询用例"这种正常维护也触发 CI 失败。
    gate_metrics = {
        k: v
        for k, v in results[hybrid_label].items()
        if k != "details" and k != "n"
    }
    weights = {"w_vector": default_wv, "w_bm25": default_wb}

    if args.update_baseline:
        payload = build_baseline_payload(
            embedding=harness.vector.embedder.active_name,
            metrics=gate_metrics,
            weights=weights,
            n_corpus=len(corpus),
            n_cases=len(cases),
        )
        BASELINE_FILE.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print("\n" + "-" * 78)
        print(f"已更新基线 → {BASELINE_FILE.name}")
        print(f"  {fmt(gate_metrics)}")
        print(f"  embedding={payload['embedding']}  权重={weights}")
        print("  注意：需要把该文件一并提交，CI 才能基于新基线把关。")

    if args.gate:
        baseline = load_baseline()
        print("\n" + "-" * 78)
        print(f"质量门禁（基线 {BASELINE_FILE.name}，容差 {GATE_TOLERANCE}）")
        print("-" * 78)
        print(f"  基线链路: embedding={baseline.get('embedding')} "
              f"权重={baseline.get('weights')}")
        print(f"  本次链路: embedding={harness.vector.embedder.active_name} "
              f"权重={weights}")
        if baseline.get("embedding") != harness.vector.embedder.active_name:
            print("  ⚠ embedding 后端与基线不一致 —— 指标不可直接比较，")
            print("    请用与基线相同的后端运行，或确认后执行 --update-baseline。")
        failures = check_gate(baseline, gate_metrics, tolerance=GATE_TOLERANCE)
        if failures:
            print("\n  ✗ 检索质量低于基线，门禁未通过：")
            for item in failures:
                print(f"      · {item}")
            print("\n  指标劣化通常来自：分词/权重改动、embedding 后端变化、")
            print("  或语料与查询集变更。确认可接受后再 --update-baseline。")
            print("=" * 78)
            return 1
        print("\n  ✓ 全部指标不低于基线，门禁通过。")
        for key, value in sorted(gate_metrics.items()):
            base_value = float(baseline["metrics"].get(key, 0.0))
            delta = value - base_value
            print(f"      {key:<10} {value:.4f}  (基线 {base_value:.4f}, "
                  f"{'+' if delta >= 0 else ''}{delta:.4f})")

    print("\n" + "=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
