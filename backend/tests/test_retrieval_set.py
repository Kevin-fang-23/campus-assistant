"""检索评测集的完整性约束。

评测集是一份手写数据，最容易出的问题是"改了 id 却忘了改 expected"这类
静默错误 —— 它不会报错，只会让指标悄悄变差，从而把权重标定带偏。
这些用例把关键不变量固化下来。
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from eval.run_retrieval_eval import bm25_text, build_text

SET_FILE = Path(__file__).resolve().parents[1] / "eval" / "retrieval_set.json"

# 噪声查询用这个哨兵 id：语料里不存在，表示"本就没有相关文档"
NOISE_ID = 999


@pytest.fixture(scope="module")
def data() -> dict:
    return json.loads(SET_FILE.read_text(encoding="utf-8"))


def test_set_parses_and_has_minimum_scale(data: dict) -> None:
    """规模下限：扩容后不应再退回小样本（权重标定依赖样本量）。"""
    assert len(data["corpus"]) >= 50, f"语料仅 {len(data['corpus'])} 条，样本过小"
    assert len(data["cases"]) >= 100, f"查询仅 {len(data['cases'])} 条，样本过小"


def test_corpus_ids_unique_and_fields_complete(data: dict) -> None:
    ids = [c["id"] for c in data["corpus"]]
    dup = [k for k, v in Counter(ids).items() if v > 1]
    assert not dup, f"语料 id 重复：{dup}"

    required = {"id", "title", "summary", "course", "location", "tags", "text"}
    for doc in data["corpus"]:
        missing = required - set(doc)
        assert not missing, f"语料 #{doc['id']} 缺字段 {missing}"
        # 结构化字段参与向量索引，为空会把向量路的文本构造弄成空串
        assert doc["title"], f"#{doc['id']} title 为空"
        assert doc["summary"], f"#{doc['id']} summary 为空（向量路依赖它）"
        assert doc["text"], f"#{doc['id']} text 为空（BM25 路依赖它）"
        assert doc["tags"], f"#{doc['id']} tags 为空（向量路依赖它）"


def test_expected_ids_reference_existing_corpus(data: dict) -> None:
    """expected 里的每个 id 都必须存在于语料中（噪声哨兵除外）。

    这是最关键的一条：写了不存在的 id 会让该查询永远不可能命中，
    指标被静默拉低，而人看不出问题。
    """
    ids = {c["id"] for c in data["corpus"]}
    bad = [
        (case["query"], nid)
        for case in data["cases"]
        for nid in case["expected"]
        if nid not in ids and nid != NOISE_ID
    ]
    assert not bad, f"expected 引用了不存在的语料 id：{bad}"


def test_every_case_has_tags(data: dict) -> None:
    """标签用于分层诊断，缺了就没法定位"哪类查询变差了"。"""
    for case in data["cases"]:
        assert case.get("tags"), f"查询 {case['query']!r} 缺 tags"


def test_corpus_covers_all_categories(data: dict) -> None:
    """五类场景都要有语料，否则模型只在小范围内被评估。"""
    tags = {t for doc in data["corpus"] for t in doc["tags"]}
    for expected_tag in ("作业", "调课", "报修", "讲座", "考试"):
        assert expected_tag in tags, f"语料缺少「{expected_tag}」类场景"


def test_index_texts_mirror_production_split(data: dict) -> None:
    """两路索引文本必须按生产的构成方式分离。

    生产里：向量路只索引结构化字段，BM25 路额外索引正文。
    若两者相同，就退化成"评测吃全文、生产吃摘要"的失真口径，
    标出的权重不可迁移 —— 这正是本次扩容修掉的问题之一。
    """
    doc = data["corpus"][0]
    vec_text = build_text(doc)
    bm_text = bm25_text(doc)

    assert doc["title"] in vec_text
    assert doc["summary"] in vec_text
    assert doc["text"] not in vec_text, "向量路不应包含正文（生产只用结构化字段）"
    assert doc["text"] in bm_text, "BM25 路必须包含正文（精确串依赖它）"
    assert bm_text.startswith(vec_text), "BM25 文本应以结构化字段开头，再接正文"


def test_noise_cases_are_marked_with_sentinel(data: dict) -> None:
    """噪声查询必须可被识别，否则会被算进主指标、稀释差异。"""
    ids = {c["id"] for c in data["corpus"]}
    noise = [c for c in data["cases"] if not (set(c["expected"]) & ids)]
    assert noise, "应保留若干无答案查询，用于暴露『没有相关性阈值』这一局限"
    for case in noise:
        assert NOISE_ID in case["expected"], (
            f"噪声查询 {case['query']!r} 应显式标注哨兵 id {NOISE_ID}"
        )


def test_noise_sample_has_minimum_scale_and_both_categories(data: dict) -> None:
    """噪声样本量下限 + 两类分层标签齐备（防退回小样本）。

    背景：阈值的拦截结论曾建立在仅 2 条噪声样本上，扩到 30 条后才发现
    真实拦截率远低于预期（校外 6/10、校园无答案 2/20）。没有量下限，
    后续维护中删减噪声样本会让「阈值能拦多少噪声」的结论再次失去统计意义。

    两类分层缺一不可：
      · 校外话题 —— 阈值对它有效（拦下大部分），是 T=1 的价值所在；
      · 校园无答案 —— 最难拦的一类（共享通用校园词汇），是已知边界的证据。
    """
    ids = {c["id"] for c in data["corpus"]}
    noise = [c for c in data["cases"] if not (set(c["expected"]) & ids)]
    assert len(noise) >= 20, (
        f"噪声样本仅 {len(noise)} 条，阈值拦截结论的统计意义不足（应 ≥ 20）"
    )
    tags = {t for c in noise for t in c.get("tags", [])}
    for required in ("校外话题", "校园无答案"):
        assert required in tags, f"噪声样本缺「{required}」分层 —— 两类的拦截难度完全不同"
