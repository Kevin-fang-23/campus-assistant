"""缓存语义匹配的阈值标定：夹出可行区间 + 扫描命中/误配代价 + 给出推荐值。

用法：
    python -m eval.run_cache_threshold_eval                    # 两个后端都跑（默认）
    python -m eval.run_cache_threshold_eval --embedding dashscope
    python -m eval.run_cache_threshold_eval --embedding local_hash
    python -m eval.run_cache_threshold_eval --top 15           # 逐对相似度明细（按相似度降序）
    python -m eval.run_cache_threshold_eval --embedding local_hash --gate
                                                               # CI 门禁：配置的阈值不得误配

标定结论记录在 `CACHE_SEMANTIC_BASELINE.md`（改了数据集或后端后需一并更新）。

## 方法论：与混合权重调参（run_retrieval_eval.py --sweep / --calibrate）一致

调参不能靠直觉拍数字，必须用**标注数据**夹出可行区间。混合权重那边是扫描权重看
MRR，这里扫描阈值看两类代价：

| 判据 | 含义 | 阈值方向 |
|---|---|---|
| 改写命中 | paraphrase 对（同意图不同问法）被判为命中 | 阈值须**够低**才命中 |
| 误配 | distinct 对（相近但答案不同）被判为命中 | 阈值须**够高**才拦住 |

于是：
  · **下界** = paraphrase 对里**最低**的余弦相似度 —— 阈值低于它才不漏掉改写；
  · **上界** = distinct 对里**最高**的余弦相似度 —— 阈值高于它才不误配。
若下界 > 上界，说明存在干净分离点，阈值可落在区间内（本脚本取区间下限，见下）；
若下界 ≤ 上界，说明余弦相似度**无法安全区分**这两类，必须如实报告 ——
硬挑一个"好看"的阈值只会在生产上把别人的答案返回给用户。

## 推荐值的选法：零误配 → 命中最多 → 平台区取最高

两类错误的代价**不对称**：
  · 漏命中（该复用却没复用）→ 多花一次 LLM 调用，钱的问题，有上限；
  · 误配（不该复用却复用了）→ 用户拿到**另一个问题的答案**且无从察觉，
    属于正确性事故，且缓存会把它放大（TTL 内所有相近问法都拿到这个错答案）。
因此先把误配压到 0，再在零误配的阈值里取命中最多的一档；
若命中数在一段区间内相同（平台区），取该区间的**最高**阈值 ——
观测收益相同，但离最难的负样本最远，安全边际最大。
（实测 dashscope 上 0.80~0.86 同为 6/10，故取 0.86 而非紧贴负样本的 0.80。）

## 阈值随 embedding 后端变化，换后端必须重跑

余弦相似度的绝对值依赖向量空间：local_hash（256 维字符哈希）与
dashscope text-embedding-v4（1024 维语义向量）的分布**不同数量级**，
同一条样本的相似度会差很多 —— 实测同一个"最低改写样本"在两者上分别是
0.136 与 0.498。因此本脚本对每个后端**分别**标定，并在输出里显式标注
实际生效的后端（`active_name`）—— 不做区分会导致"在 local_hash 上标出来的
阈值拿到 dashscope 上用"，静默失效。
（同源教训见 `hybrid_weight_vector` 的注释："换语料后务必重跑 --sweep 重新定标"。）

另有一个实测踩到的坑写在 `embed_queries` 里：dashscope 单次批量上限 10 条，
超限会 400 并被**粘性降级**成 local_hash —— 脚本照常跑完、数字也"对"，
只是全是 local_hash 的数字。第一次运行时两个后端的标定表一模一样才发现。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

from app.config import settings  # noqa: E402
from app.providers.embedding import get_embedding  # noqa: E402
from app.services.qa_cache import normalize_query  # noqa: E402

PAIRS_FILE = _HERE / "cache_pairs.json"

# 扫描范围：低于 0.50 的相似度在语义空间里已近乎"无关"，作为缓存复用阈值
# 没有任何意义（会命中一切）；上限 0.99 之上只有完全相同字符串才够得着，
# 那等于退回字面匹配。步长 0.01 与 run_retrieval_eval 的 0.05 权重步长同理：
# 细到能看出"代价何时变得可接受"。
SWEEP_START = 0.50
SWEEP_END = 0.99
SWEEP_STEP = 0.01


# ---------------------------------------------------------------------------
# 数据
# ---------------------------------------------------------------------------
def load_pairs() -> tuple[list[dict], list[dict]]:
    data = json.loads(PAIRS_FILE.read_text(encoding="utf-8"))
    return data["paraphrase"], data["distinct"]


def build_embedder(backend: str):
    """按后端取 embedder。

    与 run_retrieval_eval.Harness 同样的做法：改 settings + 清 lru_cache 单例。
    `get_embedding` 是 lru_cache 单例，不清缓存就永远拿第一个后端。
    """
    settings.embedding_provider = backend
    get_embedding.cache_clear()
    return get_embedding()


def embed_queries(embedder, pairs) -> tuple[np.ndarray, list[str]]:
    """把所有出现的 query 去重后向量化，返回 (向量矩阵, 去重后的文本表)。

    逐对调用 embed 会产生 2N 次上游调用；去重后同一字符串只算一次，
    且在 dashscope 上能显著减少请求数（标定脚本常被反复运行）。

    **必须分批**：dashscope text-embedding-v4 单次请求上限 10 条，
    超限直接 HTTP 400 `batch size is invalid, it should not be larger than 10`，
    而 `DashScopeEmbedding.embed` 捕获异常后会**粘性降级**到 local_hash ——
    表现是"标定正常跑完、数字也对、但其实是 local_hash 的数字"。
    实测踩过一次：两个后端的标定结果完全相同（都是 256 维）才发现。
    生产路径每次只 embed 1 条（查询/单条通知），因此不受此限制影响。

    分批大小取 10 而不是更小：正好卡在上限，请求数最少。
    """
    unique: list[str] = []
    seen: set[str] = set()
    for group in pairs:
        for row in group:
            for key in ("a", "b"):
                text = row[key]
                if text not in seen:
                    seen.add(text)
                    unique.append(text)

    batch = 10
    chunks = [
        np.asarray(embedder.embed(unique[i : i + batch]), dtype=np.float32)
        for i in range(0, len(unique), batch)
    ]
    return np.vstack(chunks), unique


def similarity_matrix(vecs: np.ndarray) -> np.ndarray:
    """余弦相似度矩阵。

    embedder 契约已保证输出 L2 归一化（见 providers/embedding.py），
    但仍显式归一化一次 —— 万一有后端没归一化，这里算出的就是真余弦，
    而不是被误读成"点积"。标定脚本宁可多一次除法，也不能算错口径。
    """
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    unit = vecs / np.maximum(norms, 1e-9)
    return unit @ unit.T


# ---------------------------------------------------------------------------
# 标定
# ---------------------------------------------------------------------------
def score_pairs(pairs, sim: np.ndarray, unique_index: dict[str, int]) -> list[dict]:
    rows: list[dict] = []
    for group, kind in ((pairs[0], "paraphrase"), (pairs[1], "distinct")):
        for row in group:
            i = unique_index[row["a"]]
            j = unique_index[row["b"]]
            rows.append(
                {
                    "kind": kind,
                    "a": row["a"],
                    "b": row["b"],
                    "sim": float(sim[i, j]),
                    "note": row.get("note", ""),
                    # 字面归一化是否已经能命中：用于量化语义层的**增量**价值
                    "literal_hit": normalize_query(row["a"]) == normalize_query(row["b"]),
                }
            )
    return rows


def bounds(rows: list[dict]) -> tuple[float, float]:
    """返回 (上界, 下界) = (distinct 最高, paraphrase 最低)。"""
    pos = [r["sim"] for r in rows if r["kind"] == "paraphrase"]
    neg = [r["sim"] for r in rows if r["kind"] == "distinct"]
    return max(neg), min(pos)


def sweep(rows: list[dict]) -> list[dict]:
    out: list[dict] = []
    t = SWEEP_START
    while t <= SWEEP_END + 1e-9:
        thr = round(t, 2)
        hit = sum(1 for r in rows if r["kind"] == "paraphrase" and r["sim"] >= thr)
        bad = sum(1 for r in rows if r["kind"] == "distinct" and r["sim"] >= thr)
        out.append({"threshold": thr, "hit": hit, "bad": bad})
        t += SWEEP_STEP
    return out


def recommend(rows: list[dict]) -> tuple[float | None, str]:
    """推荐阈值：先保零误配，再取改写命中最多的一档，最后取该档里**最高**的阈值。

    三步的每一步都对应一个取舍：

    1. **零误配优先**：误配（复用别的答案）是正确性事故，漏命中只是多花一次
       LLM。先把 bad 压到 0。
    2. **命中最多**：在零误配的阈值里挑改写命中率最高的那些档 —— 这是功能的
       实际收益。
    3. **平台区内取最高**：命中数往往在一段阈值区间内相同（"平台区"），
       例如实测 dashscope 上 0.80~0.86 都是 6/10。此时取**最高**的那个值，
       因为距离"最高负样本"最远 = 安全边际最大，而观测到的收益完全相同。
       取最低值（0.80）会让阈值紧贴最难的负样本（0.797），新来一条稍高的
       负样本就会误配；往上取不损失任何已观测收益。

    这条规则是**代价不对称**的直接推论：宁可漏一次（多花钱），不可错一次（答错）。
    返回 (阈值 | None, 说明)。None 表示找不到任何零误配的阈值 ——
    由调用方决定是否接受残余风险，而不是硬给一个数。
    """
    n_pos = sum(1 for r in rows if r["kind"] == "paraphrase")
    upper, lower = bounds(rows)
    safe = [r for r in sweep(rows) if r["bad"] == 0]
    if not safe:
        return None, (
            f"不存在零误配的阈值：distinct 对最高相似度 {upper:.3f} 覆盖了整个扫描区间，"
            f"需要人工取舍"
        )
    best_hit = max(r["hit"] for r in safe)
    plateau = [r for r in safe if r["hit"] == best_hit]
    pick = max(plateau, key=lambda r: r["threshold"])
    lo, hi = plateau[0]["threshold"], plateau[-1]["threshold"]
    note = (
        f"零误配前提下最高命中档为 {best_hit}/{n_pos}，"
        f"该档覆盖阈值 {lo:.2f}~{hi:.2f}（平台区），取其中最高值 {pick['threshold']:.2f} "
        f"以获得最大安全边际（距最高负样本 {upper:.3f} 有 {pick['threshold'] - upper:.3f} 的余量）"
    )
    if lower > upper:
        mid = (upper + lower) / 2
        note += (
            f"；两类样本存在干净分离点（distinct 最高 {upper:.3f} < "
            f"paraphrase 最低 {lower:.3f}，中点 {mid:.3f}）"
        )
    else:
        note += (
            f"；⚠ 两类样本区间重叠（distinct 最高 {upper:.3f} >= "
            f"paraphrase 最低 {lower:.3f}），余弦单判据无法完全分离 —— "
            f"阈值之下的改写问法会被漏掉，属于本判据的已知边界"
        )
    if best_hit == 0:
        note += (
            "；⚠ 该后端下 0/10 命中 —— 语义匹配**没有增量价值**（字面归一化已覆盖），"
            "建议保持关闭（阈值配 0）而不是启用一个只增加误配风险的功能"
        )
    return pick["threshold"], note


# ---------------------------------------------------------------------------
# 输出
# ---------------------------------------------------------------------------
def report_backend(backend: str, *, only_show_pairs: int) -> dict | None:
    pairs = load_pairs()
    embedder = build_embedder(backend)
    # 先嵌入再读 active_name：DashScopeEmbedding 的降级发生在**第一次 embed 调用**
    # 之内（粘性回退到 local_hash）。若在嵌入前取名字，拿到的是配置名而非实际
    # 生效的后端，输出就会谎报 —— 实测踩过：脚本顶着 "embedding = dashscope"
    # 的表头打印 local_hash 的数字（256 维），看的人完全无从察觉。
    vecs, unique = embed_queries(embedder, pairs)
    active = embedder.active_name
    if active != backend:
        print(
            f"\n  ⚠ 请求后端 {backend}，实际生效 {active} "
            f"（通常是缺 API Key 或上游返回错误 → 已粘性降级）。"
            f"下面的数字属于 {active}，**不可**当作 {backend} 的标定结果。"
        )
    unique_index = {t: i for i, t in enumerate(unique)}
    sim = similarity_matrix(vecs)
    rows = score_pairs(pairs, sim, unique_index)

    n_pos = sum(1 for r in rows if r["kind"] == "paraphrase")
    n_neg = len(rows) - n_pos
    upper, lower = bounds(rows)

    print(f"\n{'=' * 78}")
    print(f"缓存语义匹配阈值标定 | embedding = {active} | 维度 {vecs.shape[1]}")
    print(f"  样本：改写对 {n_pos} 条（应命中） / 相近但异义对 {n_neg} 条（必须不命中）")
    print("=" * 78)

    # ---- 逐对明细：先看数据分布，再看结论 ----
    print(f"\n  逐对相似度（降序，共 {len(rows)} 对，此处列前 {only_show_pairs} 对）：")
    print(f"    {'相似度':>7}  {'类型':<12} {'字面':<5} 问法对")
    for r in sorted(rows, key=lambda x: -x["sim"])[:only_show_pairs]:
        kind = "改写(应命中)" if r["kind"] == "paraphrase" else "异义(须拦住)"
        lit = "命中" if r["literal_hit"] else "—"
        print(f"    {r['sim']:>7.3f}  {kind:<12} {lit:<5} {r['a']}  ‖  {r['b']}")
    print(f"    （字面=字面归一化能否命中；共 {sum(1 for r in rows if r['literal_hit'])} "
          f"对属于字面即可命中）")

    # ---- 夹逼结论 ----
    print("\n  【夹逼结果】")
    print(f"    下界（改写对最低相似度，阈值须 ≤ 它才不漏改写）: {lower:.3f}")
    print(f"    上界（异义对最高相似度，阈值须 > 它才不误配）  : {upper:.3f}")
    if lower > upper:
        print(f"    → 存在干净分离点，安全区间 ({upper:.3f}, {lower:.3f}]")
    else:
        print("    → 区间重叠：余弦单判据**无法完全分离**这两类（见上方最相似的那对异义样本）")

    # ---- 扫描表 ----
    print("\n  阈值扫描（⭐ 为推荐值，两列互为代价）：")
    print(f"    {'阈值':>5} | {'改写命中':>12} | {'误配':>10}")
    rec, rec_note = recommend(rows)
    for r in sweep(rows):
        marker = "  ⭐" if rec is not None and abs(r["threshold"] - rec) < 1e-9 else ""
        print(
            f"    {r['threshold']:>5.2f} | {r['hit']:>7}/{n_pos:<4} | "
            f"{r['bad']:>6}/{n_neg:<3}{marker}"
        )

    print("\n  【推荐】")
    if rec is None:
        print(f"    ⚠ {rec_note}")
        print("    建议：调低扫描起始或改用更强的判据（见 run_cache_threshold_eval 模块说明）")
    else:
        print(f"    阈值 {rec:.2f} —— {rec_note}")

    # ---- 增量价值：语义层比字面归一化多救回多少 ----
    literal_only = sum(
        1 for r in rows if r["kind"] == "paraphrase" and r["literal_hit"]
    )
    sem_hit = sum(
        1 for r in rows if r["kind"] == "paraphrase" and r["sim"] >= (rec or 1.0)
    )
    print("\n  【增量价值】")
    print(f"    字面归一化可命中改写对: {literal_only}/{n_pos}")
    if rec is not None:
        print(f"    语义匹配（阈值 {rec:.2f}）可命中: {sem_hit}/{n_pos}")
        print(f"    → 改写问法的复用率从 {literal_only}/{n_pos} 提升到 {sem_hit}/{n_pos}")

    # ---- 当前配置阈值的实测（供 --gate 判定）----
    # 关键：**不是**看推荐值，而是看 config 里真正生效的那个值。
    # 一个"标定得很好但配置写错"的阈值在生产上照样误配。
    cfg = float(settings.qa_cache_semantic_threshold or 0.0)
    cfg_bad = (
        sum(1 for r in rows if r["kind"] == "distinct" and r["sim"] >= cfg)
        if cfg > 0
        else 0
    )
    cfg_hit = sum(
        1 for r in rows if r["kind"] == "paraphrase" and r["sim"] >= cfg
    )
    print(f"\n  【当前配置】QA_CACHE_SEMANTIC_THRESHOLD = {cfg:g}"
          f"{'（关闭）' if cfg <= 0 else ''}")
    if cfg > 0:
        print(f"    实测改写命中 {cfg_hit}/{n_pos}，误配 {cfg_bad}/{n_neg}"
              f"{'  ⚠ 存在误配！' if cfg_bad else '  ✓ 零误配'}")

    return {
        "backend": active,
        "dim": int(vecs.shape[1]),
        "lower": lower,
        "upper": upper,
        "recommend": rec,
        "hit": sem_hit,
        "n_pos": n_pos,
        "literal_hit": literal_only,
        "configured": cfg,
        "configured_bad": cfg_bad,
        "n_neg": n_neg,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="缓存语义匹配阈值标定")
    parser.add_argument(
        "--embedding",
        choices=["local_hash", "dashscope", "both"],
        default="both",
        help="标定所用的 embedding 后端（阈值随之变化，换后端必须重跑）",
    )
    parser.add_argument(
        "--top", type=int, default=14, help="逐对明细展示的条数（按相似度降序）"
    )
    parser.add_argument(
        "--gate",
        action="store_true",
        help="门禁：当前配置的阈值在标注集上必须零误配，否则退出码 1（供 CI 使用）",
    )
    args = parser.parse_args()

    backends = (
        ["local_hash", "dashscope"] if args.embedding == "both" else [args.embedding]
    )
    results = []
    for backend in backends:
        res = report_backend(backend, only_show_pairs=args.top)
        if res:
            results.append(res)

    if len(results) > 1:
        print(f"\n{'=' * 78}")
        print("两个后端对照（阈值不可跨后端通用）")
        print("=" * 78)
        names = [r["backend"] for r in results]
        if len(set(names)) == 1:
            print(
                f"  ⚠ 两行都是 {names[0]} —— 说明有一个后端在嵌入阶段降级了，"
                f"本对照表**不成立**（不是「两个后端结果一致」，而是只跑到了一个后端）。"
            )
        print(f"  {'后端':<12} {'维度':>5} {'下界':>7} {'上界':>7} {'推荐':>6} "
              f"{'改写命中':>9} {'字面命中':>9}")
        for r in results:
            rec = f"{r['recommend']:.2f}" if r["recommend"] is not None else "—"
            print(
                f"  {r['backend']:<12} {r['dim']:>5} {r['lower']:>7.3f} "
                f"{r['upper']:>7.3f} {rec:>6} "
                f"{r['hit']:>5}/{r['n_pos']:<3} {r['literal_hit']:>5}/{r['n_pos']:<3}"
            )
        print("\n  说明：把 dashscope 上标出的阈值配给 local_hash（或反向）会静默失效，")
        print("        切换 EMBEDDING_PROVIDER 后必须重跑本脚本重新定标。")

    # ---- 质量门禁：配置的阈值必须仍然零误配 ----
    # 门禁只守一条**安全不变量**："相近但异义"的问法不得被判定命中。
    # 不去卡命中率（那是收益，会因为语料/后端变化而波动，卡死只会带来假红），
    # 只卡误配 —— 它一旦出问题就是用户拿到另一个问题的答案，属于事故。
    if args.gate:
        print("\n" + "=" * 78)
        print("门禁：当前配置阈值的误配检查（dataset = cache_pairs.json）")
        print("=" * 78)
        failures = []
        for r in results:
            if r["configured"] <= 0:
                print(f"  {r['backend']:<12} 阈值 {r['configured']:g} = 关闭，跳过")
                continue
            if r["configured_bad"]:
                failures.append(
                    f"{r['backend']}: 阈值 {r['configured']:g} 下有 "
                    f"{r['configured_bad']}/{r['n_neg']} 对异义样本被误判命中"
                )
            else:
                print(
                    f"  {r['backend']:<12} 阈值 {r['configured']:g} → "
                    f"误配 0/{r['n_neg']} ✓"
                )
        if failures:
            print("\n  ✗ 存在误配，门禁未通过：")
            for item in failures:
                print(f"      · {item}")
            print("\n  误配的代价是用户拿到另一个问题的答案，且会被缓存在 TTL 内放大。")
            print("  处理方向：调高阈值，或重跑标定（不要为了好看的数字放宽这条约束）。")
            print("=" * 78)
            return 1
        print("\n  ✓ 配置阈值在标注集上零误配，门禁通过。")

    print("\n" + "=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
