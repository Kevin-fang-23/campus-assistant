"""缓存语义阈值标定脚本的测试（数据集完整性 + 推荐规则 + 门禁判据）。

为什么标定脚本也需要测试：
阈值是**用这份数据集算出来的数字**，数据集本身错了、或推荐规则被改动，
配置里的阈值就不再有任何依据 —— 而它一旦偏松，后果是用户拿到另一个问题的
答案。因此这里钉住三件事：

1. **数据集完整性**：规模下限、标注齐备、改写对不得与字面归一化重合
   （否则"语义带来的增量价值"这个结论是假的）；
2. **推荐规则**：零误配优先 → 命中最多 → 平台区取最高（代价不对称的直接推论）；
3. **不可分离时如实返回**，而不是硬凑一个数。

这些全是纯函数测试，不联网、不依赖 embedding 后端。
"""
from __future__ import annotations

import numpy as np

from eval.run_cache_threshold_eval import (
    PAIRS_FILE,
    SWEEP_END,
    SWEEP_START,
    SWEEP_STEP,
    bounds,
    load_pairs,
    score_pairs,
    sweep,
)


def _rows(*, pos: list[float], neg: list[float]) -> list[dict]:
    """构造与 score_pairs 输出同形的样本行。"""
    rows = [
        {"kind": "paraphrase", "sim": s, "a": f"p{i}", "b": f"p{i}'"}
        for i, s in enumerate(pos)
    ]
    rows += [
        {"kind": "distinct", "sim": s, "a": f"n{i}", "b": f"n{i}'"}
        for i, s in enumerate(neg)
    ]
    return rows


# ---------------------------------------------------------------------------
# 1. 数据集完整性
# ---------------------------------------------------------------------------
def test_pairs_file_exists() -> None:
    assert PAIRS_FILE.exists(), f"缺少标定数据集 {PAIRS_FILE}"


def test_dataset_has_enough_samples_on_both_sides() -> None:
    """两组样本都要有下限规模 —— 样本太少时标出来的阈值不可信。

    下限取 8：少于这个数，"最高负样本"和"最低正样本"都极易被单条样本左右，
    推荐值会随数据集的小改动来回跳。
    """
    paraphrase, distinct = load_pairs()
    assert len(paraphrase) >= 8, f"改写对只有 {len(paraphrase)} 条，样本不足"
    assert len(distinct) >= 8, f"异义对只有 {len(distinct)} 条，样本不足"


def test_every_pair_is_fully_annotated() -> None:
    """每条都必须有 a / b / note，且 a != b。

    note 不是装饰：它记录"为什么这一对算改写/算异义"，是复核标注质量时
    唯一的依据 —— 一段数字没有理由，就无法判断它是否可信。
    """
    for group_name, group in zip(("paraphrase", "distinct"), load_pairs(), strict=True):
        for row in group:
            assert row.get("a", "").strip(), f"{group_name} 有空 a：{row}"
            assert row.get("b", "").strip(), f"{group_name} 有空 b：{row}"
            assert row.get("note", "").strip(), f"{group_name} 缺 note：{row}"
            assert row["a"] != row["b"], f"{group_name} 的 a 与 b 完全相同：{row}"


def test_no_duplicate_pairs() -> None:
    """同一对问法不得重复计入 —— 重复会让扫描表上的计数虚高。"""
    seen: set[tuple[str, str]] = set()
    for group in load_pairs():
        for row in group:
            key = (row["a"], row["b"])
            assert key not in seen, f"重复样本对：{key}"
            seen.add(key)


def test_paraphrase_pairs_are_not_literal_identical() -> None:
    """改写对里最多只有 1 对是"字面归一化就能命中"的（空白/大小写对照）。

    这条是**指标可信度**的护栏：如果改写对大多是同一字符串的格式变体，
    那么"语义匹配把复用率从 1/10 提到 6/10"这个结论就是假的 ——
    增量会被字面归一化白白算进去。留 1 对作对照是有意的。
    """
    paraphrase, _ = load_pairs()
    # score_pairs 只用到 unique_index 与 sim，这里给全零矩阵即可
    unique = sorted({r[k] for r in paraphrase for k in ("a", "b")})
    index = {t: i for i, t in enumerate(unique)}
    sim = np.zeros((len(unique), len(unique)), dtype=np.float32)
    rows = score_pairs((paraphrase, []), sim, index)

    literal = [r for r in rows if r["literal_hit"]]
    assert len(literal) <= 1, (
        f"有 {len(literal)} 对改写样本字面归一化就能命中，"
        f"会让「语义增量价值」的结论失真：{[r['a'] for r in literal]}"
    )


# ---------------------------------------------------------------------------
# 2. 夹逼边界
# ---------------------------------------------------------------------------
def test_bounds_are_max_negative_and_min_positive() -> None:
    rows = _rows(pos=[0.90, 0.80, 0.70], neg=[0.60, 0.50])
    upper, lower = bounds(rows)
    assert upper == 0.60, "上界应为异义对的最高相似度"
    assert lower == 0.70, "下界应为改写对的最低相似度"


# ---------------------------------------------------------------------------
# 3. 推荐规则：零误配 → 命中最多 → 平台区取最高
# ---------------------------------------------------------------------------
def test_recommend_picks_top_of_max_hit_plateau() -> None:
    """命中数相同的平台区内，必须取**最高**阈值（安全边际最大）。

    正样本 0.90/0.85、负样本 0.60：
      · 零误配要求阈值 > 0.60；
      · 阈值在 (0.60, 0.85] 内都命中 2/10，是平台区；
      · 故推荐 0.85（而不是 0.61）—— 观测收益相同，但离负样本最远。
    """
    from eval.run_cache_threshold_eval import recommend

    rec, note = recommend(_rows(pos=[0.90, 0.85], neg=[0.60]))
    assert rec == 0.85, f"应取平台区最高值 0.85，实际 {rec}；说明：{note}"
    assert "平台区" in note


def test_recommend_never_trades_false_hits_for_recall() -> None:
    """"零误配"是硬约束：宁可全不命中，也不接受任何一条误配。

    正样本 0.96/0.90、负样本 0.97 —— 阈值取到 0.90 以下能命中 2 条，
    但会把 0.97 那条异义样本判成命中。规则必须选"阈值 0.99、零命中"，
    而不是"命中 2 条但误配 1 条"。
    """
    from eval.run_cache_threshold_eval import recommend

    rec, note = recommend(_rows(pos=[0.96, 0.90], neg=[0.97]))
    assert rec is not None
    assert rec > 0.97, f"推荐值 {rec} 会让 0.97 的异义样本被误配"
    assert "没有增量价值" in note, f"零命中时应提示无增量价值，实际：{note}"


def test_recommend_returns_none_when_no_safe_threshold() -> None:
    """异义样本相似度高过扫描上限时，如实返回 None，不硬凑一个数。"""
    from eval.run_cache_threshold_eval import recommend

    rec, note = recommend(_rows(pos=[0.99, 0.95], neg=[0.995]))
    assert rec is None, f"不存在零误配的阈值，不该返回 {rec}"
    assert "不存在零误配" in note


def test_recommend_reports_clean_separation_when_available() -> None:
    """两类区间不重叠时应明确报出分离点 —— 这是"判据可用"的证据。"""
    from eval.run_cache_threshold_eval import recommend

    rec, note = recommend(_rows(pos=[0.95, 0.92], neg=[0.40, 0.30]))
    assert rec is not None
    assert "干净分离点" in note, f"应报出分离点，实际：{note}"


# ---------------------------------------------------------------------------
# 4. 扫描表
# ---------------------------------------------------------------------------
def test_sweep_covers_range_with_two_decimals() -> None:
    rows = sweep(_rows(pos=[0.90], neg=[0.50]))
    assert rows[0]["threshold"] == SWEEP_START
    assert rows[-1]["threshold"] == round(SWEEP_END, 2)
    expected = int(round((SWEEP_END - SWEEP_START) / SWEEP_STEP)) + 1
    assert len(rows) == expected, f"扫描点数应为 {expected}，实际 {len(rows)}"


def test_sweep_counts_are_monotonic() -> None:
    """阈值越高，命中数只会减少、误配数只会减少（单调性保证表可解读）。"""
    rows = sweep(_rows(pos=[0.99, 0.80, 0.60], neg=[0.70, 0.40]))
    hits = [r["hit"] for r in rows]
    bads = [r["bad"] for r in rows]
    assert hits == sorted(hits, reverse=True), f"命中数不单调：{hits}"
    assert bads == sorted(bads, reverse=True), f"误配数不单调：{bads}"
