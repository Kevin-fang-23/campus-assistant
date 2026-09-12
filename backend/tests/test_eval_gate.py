"""检索质量门禁（#7）的测试。

门禁的价值全在"它真的会拦住劣化"。一个只会打印数字、永远退 0 的脚本
放进 CI 是**负资产** —— 它让人以为有保护，实际没有。因此这里重点钉住：

1. 指标低于基线（超出容差）→ 必须报出劣化项；
2. 指标持平或上升 → 必须通过（不能因为"与基线不一致"就失败，
   否则任何改进都要先改基线才能过 CI，反而阻碍优化）；
3. 容差内的轻微波动 → 通过（否则依赖小版本差异会让 CI 频繁假红）；
4. 基线缺失 → 明确报错，而不是静默放行。
"""
from __future__ import annotations

import json

import pytest

from eval.run_retrieval_eval import (
    BASELINE_FILE,
    GATE_TOLERANCE,
    build_baseline_payload,
    check_gate,
    load_baseline,
)

# ---------------------------------------------------------------------------
# 1. 劣化必须被拦下
# ---------------------------------------------------------------------------
def test_gate_flags_metric_below_baseline() -> None:
    """核心行为：指标掉到基线以下（超出容差）必须报出劣化。"""
    baseline = {"metrics": {"mrr@5": 0.90}}
    current = {"mrr@5": 0.80}

    failures = check_gate(baseline, current, tolerance=GATE_TOLERANCE)

    assert failures, "MRR 从 0.90 掉到 0.80 竟然通过了门禁"
    assert any("mrr@5" in f for f in failures), f"劣化项未点名指标：{failures}"


def test_gate_reports_every_degraded_metric() -> None:
    """多个指标同时劣化时都要报出来 —— 只报第一条会让人漏掉根因。"""
    baseline = {"metrics": {"mrr@5": 0.90, "recall@1": 0.80, "ndcg@5": 0.85}}
    current = {"mrr@5": 0.70, "recall@1": 0.60, "ndcg@5": 0.65}

    failures = check_gate(baseline, current, tolerance=GATE_TOLERANCE)

    assert len(failures) == 3, f"应报出 3 项劣化，实际 {len(failures)}：{failures}"


# ---------------------------------------------------------------------------
# 2. 正常情况必须通过（不能误伤）
# ---------------------------------------------------------------------------
def test_gate_passes_when_metrics_hold() -> None:
    baseline = {"metrics": {"mrr@5": 0.90, "recall@5": 1.0}}
    assert check_gate(baseline, {"mrr@5": 0.90, "recall@5": 1.0}, tolerance=GATE_TOLERANCE) == []


def test_gate_passes_when_metrics_improve() -> None:
    """指标**上升**不得触发失败。

    回归风险：若实现写成"与基线不等即失败"，那么任何人做了改进都要先改基线
    才能过 CI —— 门禁就从"防劣化"变成了"防改进"。
    """
    baseline = {"metrics": {"mrr@5": 0.90}}
    assert check_gate(baseline, {"mrr@5": 0.99}, tolerance=GATE_TOLERANCE) == []


def test_gate_tolerates_small_regression_within_tolerance() -> None:
    """容差内的波动应通过：浮点误差与依赖小版本差异不该让 CI 假红。"""
    baseline = {"metrics": {"mrr@5": 0.9000}}
    tiny = 0.9000 - GATE_TOLERANCE / 2

    assert check_gate(baseline, {"mrr@5": tiny}, tolerance=GATE_TOLERANCE) == []

    # 但一旦超出容差就必须拦下 —— 否则容差就变成了"无限放行"
    beyond = 0.9000 - GATE_TOLERANCE * 2
    assert check_gate(baseline, {"mrr@5": beyond}, tolerance=GATE_TOLERANCE)


# ---------------------------------------------------------------------------
# 3. 边界与异常
# ---------------------------------------------------------------------------
def test_gate_ignores_sample_count() -> None:
    """`n`（查询条数）不参与门禁。

    它是样本量而非质量指标。若一起卡住，"新增/删减评测用例"这种正常维护
    就会莫名触发 CI 失败，逼人为了过 CI 而不敢动测试集。
    """
    baseline = {"metrics": {"mrr@5": 0.90, "n": 129}}
    current = {"mrr@5": 0.90, "n": 100}   # 样本量变了，质量没变

    assert check_gate(baseline, current, tolerance=GATE_TOLERANCE) == []


def test_gate_flags_metric_missing_from_current() -> None:
    """本次没算出来的指标也要报 —— 静默缺失等同于静默失效。"""
    failures = check_gate({"metrics": {"mrr@5": 0.9, "ndcg@5": 0.8}}, {"mrr@5": 0.9},
                          tolerance=GATE_TOLERANCE)
    assert any("ndcg@5" in f for f in failures), failures


def test_gate_empty_baseline_metrics_passes() -> None:
    """基线里没有指标时不应崩（首次接入的过渡状态）。"""
    assert check_gate({"metrics": {}}, {"mrr@5": 0.1}, tolerance=GATE_TOLERANCE) == []


# ---------------------------------------------------------------------------
# 4. 基线文件本身
# ---------------------------------------------------------------------------
def test_baseline_file_exists_and_is_wellformed() -> None:
    """基线文件必须随仓库提交 —— 否则 CI 里的 --gate 直接因缺文件而失败。

    这条断言也防止"只在本地生成过、忘了 git add"的情况。
    """
    assert BASELINE_FILE.exists(), (
        f"缺少 {BASELINE_FILE.name}。生成："
        "python -m eval.run_retrieval_eval --embedding local_hash --update-baseline"
    )
    data = json.loads(BASELINE_FILE.read_text(encoding="utf-8"))

    for key in ("embedding", "weights", "metrics"):
        assert key in data, f"基线缺少字段 {key}"
    assert data["embedding"] == "local_hash", (
        "基线必须用确定性的 local_hash 记录：dashscope 会随服务端波动，"
        "拿它当门禁基准只会带来假红"
    )

    # 质量指标必须齐备：缺一个就等于那一项没有保护
    for metric in ("recall@1", "recall@3", "recall@5", "mrr@5", "ndcg@5"):
        assert metric in data["metrics"], f"基线缺少指标 {metric}"
        assert 0.0 <= data["metrics"][metric] <= 1.0, (
            f"{metric} 不在 [0,1]：{data['metrics'][metric]}"
        )


def test_load_baseline_raises_when_file_absent(tmp_path, monkeypatch) -> None:
    """缺基线时必须**明确报错**，不能静默放行。

    静默放行是最糟的失败模式：CI 一直绿，实际什么都没检查。
    """
    import eval.run_retrieval_eval as mod

    monkeypatch.setattr(mod, "BASELINE_FILE", tmp_path / "nope.json")
    with pytest.raises(SystemExit) as exc:
        load_baseline()
    assert "nope.json" in str(exc.value), "报错信息应指出缺的是哪个文件"


def test_build_baseline_payload_shape() -> None:
    """写回基线的结构必须含链路信息，让"换了后端/权重"可被人工核对。"""
    payload = build_baseline_payload(
        embedding="local_hash",
        metrics={"mrr@5": 0.9},
        weights={"w_vector": 0.1, "w_bm25": 0.9},
        n_corpus=55,
        n_cases=129,
    )
    assert payload["embedding"] == "local_hash"
    assert payload["weights"] == {"w_vector": 0.1, "w_bm25": 0.9}
    assert payload["corpus_size"] == 55
    assert payload["case_count"] == 129
    assert payload["metrics"] == {"mrr@5": 0.9}
    # 写回的文件要能被人看懂，因此带说明字段
    assert payload["_comment"]
