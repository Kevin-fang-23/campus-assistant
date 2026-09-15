"""混合检索（建议 11）的测试。

分三层：
1. 分词与 BM25 —— 中文 bigram、IDF、词频饱和、长度归一化、增删一致性；
2. 融合 —— 加权 RRF 的排序性质与边界（单路命中、归一化、确定性）；
3. 端到端 —— /api/search、/api/qa 走混合检索，且降级路径可用。
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.config import settings
from app.main import app
from app.services.bm25 import BM25Index, tokenize
from app.services.hybrid import (
    fuse,
    hybrid_search,
    is_relevant,
    reset_indexes,
    shared_term_count,
)

# ---------------------------------------------------------------------------
# 1. 分词
# ---------------------------------------------------------------------------
def test_tokenize_chinese_bigrams() -> None:
    """中文串切相邻双字：无语料分词器时的经典做法。"""
    assert tokenize("线性代数") == ["线性", "性代", "代数"]


def test_tokenize_single_cjk_char_kept() -> None:
    assert tokenize("书") == ["书"]


def test_tokenize_ascii_kept_whole() -> None:
    """英文/数字串整体保留：房间号、缩写、邮箱片段不能被拆碎。"""
    assert tokenize("B203") == ["b203"]
    assert tokenize("CET-4") == ["cet", "4"]


def test_tokenize_mixed_text() -> None:
    tokens = tokenize("教一302室")
    assert "教一" in tokens
    assert "302" in tokens


def test_tokenize_empty() -> None:
    assert tokenize("") == []
    assert tokenize("！！，。") == []


# ---------------------------------------------------------------------------
# 2. BM25
# ---------------------------------------------------------------------------
@pytest.fixture
def index() -> BM25Index:
    idx = BM25Index(k1=1.5, b=0.75)
    idx.add(1, "线性代数期末复习提纲提交，截止时间9月18日。")
    idx.add(2, "数据结构实验报告要求，实现二叉搜索树。")
    idx.add(3, "关于计算机网络课程调课的通知，地点改为第三教学楼A305。")
    idx.add(4, "宿舍报修申请，卫生间水管漏水，请联系后勤。")
    return idx


def test_bm25_exact_term_match_wins(index: BM25Index) -> None:
    """精确课程名必须一次命中 —— 这正是 BM25 补向量短板的场景。"""
    hits = index.search("线性代数", top_k=3)
    assert hits, "应至少命中一条"
    assert hits[0][0] == 1


def test_bm25_room_number_match(index: BM25Index) -> None:
    hits = index.search("A305", top_k=3)
    assert hits and hits[0][0] == 3


def test_bm25_no_match_returns_empty(index: BM25Index) -> None:
    """查询词全不在词表时应返回空，而不是给所有文档一个基础分。"""
    assert index.search("量子纠缠退相干", top_k=3) == []


def test_bm25_scores_are_positive(index: BM25Index) -> None:
    assert all(score > 0 for _, score in index.search("报修 漏水", top_k=5))


def test_bm25_idf_downweights_common_terms() -> None:
    """高频词 IDF 更低：给所有文档都加分的词不该主导排序。"""
    idx = BM25Index()
    for i in range(1, 6):
        idx.add(i, "课程通知")           # 所有文档都含"课程"
    idx.add(6, "奖学金申请通知")          # 独有一条含"奖学金"
    assert idx._idf("课程") < idx._idf("奖学")


def test_bm25_term_frequency_saturates() -> None:
    """词频收益递减：把某词重复 50 次的文档，不该是重复 1 次的 50 倍分。"""
    idx = BM25Index()
    idx.add(1, "报修")
    idx.add(2, "报修 " + "报修 " * 50)
    score1 = dict(idx.search("报修", top_k=5))[1]
    score2 = dict(idx.search("报修", top_k=5))[2]
    assert score2 > score1, "词频高确实应该分更高"
    assert score2 < score1 * 5, f"但不应线性增长：{score1=} {score2=}"


def test_bm25_length_normalization() -> None:
    """长度归一化：同样含一次关键词，长文档得分应低于短文档。"""
    idx = BM25Index()
    idx.add(1, "报修")
    idx.add(2, "报修" + "其他无关内容" * 20)
    scores = dict(idx.search("报修", top_k=5))
    assert scores[1] > scores[2]


def test_bm25_add_is_idempotent() -> None:
    """重复 add 同一 id 必须先移除旧版本，否则文档频率会虚高。"""
    idx = BM25Index()
    idx.add(1, "线性代数")
    idx.add(1, "线性代数")
    assert idx.size == 1
    assert idx._df["线性"] == 1, "重复写入不该把 df 累加"


def test_bm25_remove_cleans_postings() -> None:
    idx = BM25Index()
    idx.add(1, "线性代数")
    idx.add(2, "数据结构")
    idx.remove(1)
    assert idx.size == 1
    assert "线性" not in idx._postings, "无文档引用的 term 应被清理"
    assert idx.search("线性代数", top_k=3) == []


def test_bm25_clear(index: BM25Index) -> None:
    index.clear()
    assert index.size == 0
    assert index.avgdl == 0.0
    assert index.search("线性代数") == []


def test_bm25_deterministic(index: BM25Index) -> None:
    """同分按 id 升序 → 同输入同输出，测试里的"前后一致"断言才成立。"""
    a = index.search("通知 课程 地点", top_k=5)
    b = index.search("通知 课程 地点", top_k=5)
    assert a == b


def test_bm25_respects_top_k(index: BM25Index) -> None:
    assert len(index.search("通知", top_k=2)) <= 2


# ---------------------------------------------------------------------------
# 3. 融合
# ---------------------------------------------------------------------------
def test_fuse_prefers_document_ranked_high_by_both() -> None:
    """两路都排第一的文档必须排最前。"""
    vec = [(1, 0.9), (2, 0.5)]
    bm = [(1, 12.0), (3, 8.0)]
    hits = fuse(vec, bm, k=60, w_vector=0.3, w_bm25=0.7)
    assert hits[0].notice_id == 1
    assert hits[0].match == "both"
    assert hits[0].score == pytest.approx(1.0), "两路均第一 → 归一化后应为 1.0"


def test_fuse_marks_match_source() -> None:
    vec = [(1, 0.9)]
    bm = [(2, 12.0)]
    by_id = {h.notice_id: h for h in fuse(vec, bm, k=60, w_vector=0.5, w_bm25=0.5)}
    assert by_id[1].match == "vector"
    assert by_id[2].match == "bm25"
    assert by_id[1].score_vector == 0.9 and by_id[1].score_bm25 is None
    assert by_id[2].score_bm25 == 12.0 and by_id[2].score_vector is None


def test_fuse_scores_are_normalized() -> None:
    vec = [(i, 1.0 - i * 0.1) for i in range(1, 6)]
    bm = [(i, 20.0 - i) for i in range(3, 8)]
    hits = fuse(vec, bm, k=60, w_vector=0.3, w_bm25=0.7)
    assert all(0.0 <= h.score <= 1.0 for h in hits)
    assert hits == sorted(hits, key=lambda h: (-h.score, h.notice_id))


def test_fuse_weight_moves_ranking() -> None:
    """权重必须真的影响排序 —— 否则"可调权重"就是摆设。"""
    vec = [(1, 0.99)]   # 向量排 1
    bm = [(2, 50.0)]    # BM25 排 1
    heavy_vec = fuse(vec, bm, k=60, w_vector=0.9, w_bm25=0.1)
    heavy_bm = fuse(vec, bm, k=60, w_vector=0.1, w_bm25=0.9)
    assert heavy_vec[0].notice_id == 1
    assert heavy_bm[0].notice_id == 2


def test_fuse_rrf_k_smooths_rank_advantage() -> None:
    """k 越大，名次优势越平缓 → 第 1 与第 2 名的分差应缩小。"""
    vec = [(1, 0.9), (2, 0.8)]
    small = fuse(vec, [], k=1, w_vector=1.0, w_bm25=0.0)
    large = fuse(vec, [], k=1000, w_vector=1.0, w_bm25=0.0)
    assert (small[0].score - small[1].score) > (large[0].score - large[1].score)


def test_fuse_single_side_only() -> None:
    """只有一路有结果时，该路的第一名仍应得满分。

    回归：归一化分母若按**配置权重**算（0.3+0.7），唯一可用那一路的第一名
    只能拿到 0.3/(0.3+0.7)=0.3，界面显示"检索分 30%"，看着像坏了。
    正确做法是按**实际可用路径**的权重算分母。
    """
    only_vec = fuse([(1, 0.9), (2, 0.8)], [], k=60, w_vector=0.3, w_bm25=0.7)
    assert [h.notice_id for h in only_vec] == [1, 2]
    assert only_vec[0].score == pytest.approx(1.0)
    assert only_vec[0].match == "vector"

    only_bm = fuse([], [(9, 5.0)], k=60, w_vector=0.3, w_bm25=0.7)
    assert [h.notice_id for h in only_bm] == [9]
    assert only_bm[0].score == pytest.approx(1.0)
    assert only_bm[0].match == "bm25"


def test_fuse_handles_negative_cosine() -> None:
    """余弦可能为负 —— RRF 只看排名，因此不受影响（这是选 RRF 的理由之一）。"""
    hits = fuse([(1, -0.3)], [(1, 7.0)], k=60, w_vector=0.3, w_bm25=0.7)
    assert hits[0].notice_id == 1
    assert hits[0].score == pytest.approx(1.0)
    assert hits[0].score_vector == -0.3


def test_fuse_empty_inputs() -> None:
    assert fuse([], [], k=60, w_vector=0.3, w_bm25=0.7) == []


def test_fuse_zero_weights_returns_empty() -> None:
    """两路权重都是 0 时归一化分母为 0，必须安全返回而不是 ZeroDivisionError。"""
    assert fuse([(1, 0.9)], [(1, 5.0)], k=60, w_vector=0.0, w_bm25=0.0) == []


def test_fuse_deterministic_tie_break() -> None:
    """真正的同分场景：两路名次互换，加权 RRF 后总分完全相等。

    注意同一路内部两条文档名次必然不同，因此"同分"只能由两路交叉构成 ——
    这也是为什么排序键里必须带 notice_id 兜底：否则同分时顺序取决于 set 迭代顺序。
    """
    vec = [(5, 0.9), (3, 0.8)]   # doc5 向量第 1，doc3 第 2
    bm = [(3, 10.0), (5, 8.0)]   # doc3 关键词第 1，doc5 第 2
    hits = fuse(vec, bm, k=60, w_vector=0.5, w_bm25=0.5)
    assert hits[0].score == pytest.approx(hits[1].score), "前提：两者确实同分"
    assert [h.notice_id for h in hits] == [3, 5], "同分应按 id 升序，保证确定性"


# ---------------------------------------------------------------------------
# 4. 端到端
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def _reload_indexes_from_db() -> None:
    """转发到 conftest 的共享实现（保持本文件内调用点可读）。

    收尾为什么必须重建索引、以及不重建会导致什么后果，
    详见 `tests/conftest.py::reload_indexes_from_db` 的说明。
    """
    from tests.conftest import reload_indexes_from_db

    reload_indexes_from_db()


def test_search_response_exposes_match_info(client: TestClient) -> None:
    """接口需暴露"靠哪一路命中"，让混合检索可被看见/排查。"""
    resp = client.post("/api/search", json={"query": "线性代数 作业", "top_k": 5})
    assert resp.status_code == 200
    body = resp.json()
    assert body["hits"], "应召回结果"
    hit = body["hits"][0]
    for field in ("score", "match", "score_vector", "score_bm25"):
        assert field in hit, f"缺少字段 {field}"
    assert hit["match"] in ("both", "vector", "bm25")
    assert 0.0 <= hit["score"] <= 1.0


def test_search_finds_notice_by_exact_room_number(client: TestClient) -> None:
    """精确房间号：BM25 的看家本领，向量路常常打不中。

    先入库一条含**唯一房间号**的通知再检索 —— 测试库里原本没有这个房间号，
    直接查是查不到的（写这条用例时踩过：断言写成了 `any(...) or hits`，
    结果 hits 为空时 `False or []` 返回空列表才暴露出来）。
    """
    content = "关于《离散数学》习题课教室变更的通知\n各位同学：习题课地点改为格物楼C407，请相互转告。\n"
    r = client.post(
        "/api/documents/text", json={"content": content, "filename": "hybrid_room_probe.txt"}
    )
    assert r.status_code == 200

    resp = client.post("/api/search", json={"query": "格物楼C407", "top_k": 3})
    assert resp.status_code == 200
    hits = resp.json()["hits"]
    assert hits, "精确房间号应能召回刚入库的通知"
    top = hits[0]
    assert "离散数学" in top["title"] or "C407" in (top["summary"] or "") or "C407" in top["title"], (
        f"Top1 应是习题课通知，实际：{top['title']!r} / {top['summary']!r}"
    )
    # 精确匹配靠 BM25，因此该命中至少应包含 bm25 路
    assert top["match"] in ("bm25", "both"), f"精确房间号未走关键词路：{top['match']}"
    assert top["score_bm25"] is not None


def test_qa_uses_hybrid_and_still_degrades(client: TestClient) -> None:
    """问答走混合检索；LLM 不可用时降级路径照常返回引用。"""
    resp = client.post("/api/qa", json={"query": "作业什么时候截止", "top_k": 3})
    assert resp.status_code == 200
    body = resp.json()
    assert body["citations"], "混合检索应召回引用"
    if body["degraded"]:
        assert body["answer"] == body["citations"][0]["snippet"]


def test_hybrid_disabled_falls_back_to_vector(client: TestClient, monkeypatch) -> None:
    """hybrid_enabled=False 必须退回纯向量（可回滚开关），且不报错。"""
    monkeypatch.setattr(settings, "hybrid_enabled", False)
    resp = client.post("/api/search", json={"query": "作业截止", "top_k": 3})
    assert resp.status_code == 200
    for hit in resp.json()["hits"]:
        assert hit["match"] == "vector"
        assert hit["score_bm25"] is None


def test_hybrid_empty_bm25_falls_back_gracefully(client: TestClient, monkeypatch) -> None:
    """BM25 索引为空时不得报错，应退回纯向量。"""
    from app.services import bm25 as bm25_module

    monkeypatch.setattr(bm25_module, "_bm25", BM25Index())  # 空索引
    resp = client.post("/api/search", json={"query": "作业截止", "top_k": 3})
    assert resp.status_code == 200


def test_hybrid_search_respects_top_k(client: TestClient) -> None:
    hits = hybrid_search("作业 通知 课程", top_k=2)
    assert len(hits) <= 2


def test_index_notice_writes_both_indexes(client: TestClient) -> None:
    """索引入口必须同时写入两路 —— 否则会出现"向量更新了 BM25 没更新"的静默不一致。"""
    from app.db import SessionLocal
    from app.models import Document, Notice
    from app.services.bm25 import get_bm25_index
    from app.services.hybrid import index_notice

    before_size = get_bm25_index().size
    with SessionLocal() as db:
        doc = Document(filename="hybrid_probe.txt", kind="text", sha256="sha-hybrid-probe")
        db.add(doc)
        db.flush()
        notice = Notice(
            document_id=doc.id,
            category="other",
            title="混合索引探针通知",
            summary="用于验证双索引写入的探针内容。",
        )
        db.add(notice)
        db.flush()
        index_notice(db, notice)
        db.commit()

        assert get_bm25_index().size == before_size + 1, "BM25 未同步写入"
        assert get_bm25_index().search("混合索引探针", top_k=3), "写入后应可被检索到"

        # 清理，避免污染其它用例
        db.delete(notice)
        db.delete(doc)
        db.commit()

    # 向量索引没有单条删除接口，必须按库重建 —— 否则会留下孤儿
    #（见 _reload_indexes_from_db 的说明：那是"单跑绿、全量红"的根源）
    _reload_indexes_from_db()
    assert get_bm25_index().size == before_size


def test_bm25_indexes_raw_body_not_just_extracted_fields(client: TestClient) -> None:
    """回归：BM25 必须索引**正文**，否则正文里的精确串搜不到。

    这是端到端测试抓出来的缺陷 —— 原先 BM25 只索引 build_text
    （标题/摘要/课程/地点/标签）。查询 `027-87659999` 时，这个电话从未被
    抽取进任何结构化字段，BM25 只能匹配到 `027` 这个共享前缀，
    结果被另一条同样以 027 开头的通知（更短，长度归一化更有利）抢了首位。
    而"精确串一次命中"正是 BM25 存在的理由。
    """
    from app.db import SessionLocal
    from app.models import Document, Notice
    from app.services.bm25 import get_bm25_index
    from app.services.hybrid import index_notice

    # 正文含唯一手机号，但结构化字段里一律不出现该号码
    raw = "关于《复变函数》习题课的通知\n习题课改到周三晚，有疑问请联系助教，手机：13912345678。\n"
    with SessionLocal() as db:
        doc = Document(
            filename="body_probe.txt", kind="text", sha256="sha-body-probe", raw_text=raw
        )
        db.add(doc)
        db.flush()
        notice = Notice(
            document_id=doc.id,
            category="course_notice",
            title="关于《复变函数》习题课的通知",
            summary="习题课时间调整，详情联系助教。",
            course="复变函数",
        )
        db.add(notice)
        db.flush()
        index_notice(db, notice)
        db.commit()

        idx = get_bm25_index()
        hits = idx.search("13912345678", top_k=5)
        assert hits, "正文里的手机号必须能被 BM25 搜到（需索引 raw_text）"
        assert hits[0][0] == notice.id, f"应命中该通知，实际 {hits}"

        db.delete(notice)
        db.delete(doc)
        db.commit()

    # 同 test_index_notice_writes_both_indexes：向量索引需按库重建，避免孤儿
    _reload_indexes_from_db()


def test_reset_indexes_clears_bm25() -> None:
    """reset_indexes() 必须把 BM25 单例清空。

    注意收尾：`reset_indexes()` 只是把单例置 None（下次访问才懒重建），
    因此在**会话级共享**的测试进程里，这条用例会顺手毁掉后面所有用例
    （含本次新增的相关性阈值判定，它依赖 BM25 路提供词面信号）所依赖的索引。
    这正是"单独跑绿、全量跑红"的又一来源 —— 必须在使用后按库重建。

    历史背景：本用例起初没有收尾，导致 `test_search_filters_irrelevant_query_end_to_end`
    等用例在全量运行时 BM25 为空 → hybrid 静默退回纯向量 → 阈值判定的
    BM25 信号消失 → filtered 恒为 False。
    """
    from app.services.bm25 import get_bm25_index

    get_bm25_index().add(999, "临时文档")
    reset_indexes()
    assert get_bm25_index().size == 0

    # 收尾：把索引恢复成「库里有什么就索引什么」，避免影响后续用例
    _reload_indexes_from_db()


# ---------------------------------------------------------------------------
# 5. 相关性阈值（#6b）
# ---------------------------------------------------------------------------
# 判据是「查询与文档共享的二字及以上词个数」。以下用例把这条判据的
# **反面**也钉住：单字重合必须**不**算相关 —— 那正是分数阈值失效的原因。
def test_shared_term_count_ignores_single_chars() -> None:
    """单字重合不算相关 —— 这是整个阈值方案成立的前提。

    "今天天气怎么样" 与正文里的 "明天" 共享一个 '天' 字，若把单字算进去，
    这条查询就会"看起来相关"，阈值也就形同虚设。
    """
    assert shared_term_count("今天天气怎么样", "希望后勤师傅明天上午上门处理") == 0
    # 单字查询本身没有二字词，与任何文本的重合都是 0
    assert shared_term_count("天", "明天天气") == 0


def test_shared_term_count_counts_two_char_overlap() -> None:
    assert shared_term_count("线性代数作业", "《线性代数》期末复习提纲") >= 2


def test_shared_term_count_reuses_bm25_tokenizer_ascii() -> None:
    """ASCII 串按整串比对（与 BM25 同一切词），房间号/缩写要能判定为相关。"""
    assert shared_term_count("格物楼C407", "习题课地点改为格物楼C407") >= 1


def test_is_relevant_respects_threshold(monkeypatch) -> None:
    from app.services.hybrid import RetrievalHit

    hit = RetrievalHit(
        notice_id=1, score=0.9, score_vector=0.5, score_bm25=3.0, match="both"
    )
    texts = {1: "关于开放夜间自习教室的通知\n期末考试周期间开放通宵自习教室。"}

    monkeypatch.setattr(settings, "search_min_bigram_overlap", 1)
    assert is_relevant(hit, "通宵自习教室在哪", texts) is True
    assert is_relevant(hit, "今天天气怎么样", texts) is False

    # 关掉阈值（0）时一律视为相关 —— 可回滚到旧行为
    monkeypatch.setattr(settings, "search_min_bigram_overlap", 0)
    assert is_relevant(hit, "今天天气怎么样", texts) is True


def test_is_relevant_missing_text_defaults_to_open(monkeypatch) -> None:
    """拿不到文本时必须**放开**而不是收紧：宁可多给一条，也别丢掉真实命中。"""
    from app.services.hybrid import RetrievalHit

    monkeypatch.setattr(settings, "search_min_bigram_overlap", 5)
    hit = RetrievalHit(
        notice_id=42, score=0.9, score_vector=None, score_bm25=None, match="vector"
    )
    assert is_relevant(hit, "任意查询", {}) is True


def test_search_filters_irrelevant_query_end_to_end(client: TestClient) -> None:
    """端到端：与全库零词面交集的查询应返回空列表并置 filtered=True。

    这是本次改造的产品语义 —— "没找到"就如实说，而不是硬凑 top-k。

    查询选 `红楼梦作者是谁` 而非"今天天气怎么样"：后者与测试库里的
    《人工智能前沿讲座》会共享 bigram（实测重合=1）而侥幸留下一条，
    测试就变成了"看库里有哪几条数据"，换个库状态就红。
    这里要钉的是**机制**（零交集必被拦），不是某条查询的运气。
    """
    resp = client.post("/api/search", json={"query": "红楼梦作者是谁", "top_k": 5})
    assert resp.status_code == 200
    body = resp.json()
    assert body["hits"] == [], f"零交集查询不应返回结果，实际：{body['hits']}"
    assert body["filtered"] is True, "应标记为「因阈值过滤而空」，供前端区分文案"


def test_search_keeps_real_results_after_threshold(client: TestClient) -> None:
    """阈值不得误杀真实查询 —— 开启阈值后正常的检索必须照常返回结果。"""
    resp = client.post("/api/search", json={"query": "线性代数 作业", "top_k": 5})
    assert resp.status_code == 200
    body = resp.json()
    assert body["hits"], "真实查询被阈值误杀了"


def test_filtered_flag_only_set_when_result_is_emptied(client: TestClient) -> None:
    """`filtered` 只在**结果被清空**时置位，有结果时恒为 False。

    回归：早期实现用 `len(kept) < len(hits)` 判定，而候选池是故意多召回的
    （hybrid_fetch_k=20），几乎总有边缘候选被丢 —— 于是这个标志位恒为 True、
    彻底失去信息量，前端也就没法区分两种空态文案。
    """
    # 有结果：即使候选池里有条目被丢，也不该置位
    ok = client.post("/api/search", json={"query": "线性代数 作业", "top_k": 5}).json()
    assert ok["hits"]
    assert ok["filtered"] is False

    # 无结果但库里非空：这时才置位（正是前端要区分的场景）
    empty = client.post("/api/search", json={"query": "红楼梦作者是谁", "top_k": 5}).json()
    assert empty["hits"] == []
    assert empty["filtered"] is True


def test_search_threshold_can_be_disabled(client: TestClient, monkeypatch) -> None:
    """search_min_bigram_overlap=0 时必须退回旧行为（恒返回 top-k）。"""
    monkeypatch.setattr(settings, "search_min_bigram_overlap", 0)
    resp = client.post("/api/search", json={"query": "红楼梦作者是谁", "top_k": 3})
    assert resp.status_code == 200
    body = resp.json()
    assert body["hits"], "关闭阈值后应退回「恒返回 top-k」的旧行为"
    assert body["filtered"] is False


def test_vector_index_has_no_orphans(client: TestClient) -> None:
    """向量索引条数必须等于库内通知数 —— 不许有孤儿。

    回归：早期用例（test_index_notice_writes_both_indexes /
    test_bm25_indexes_raw_body_not_just_extracted_fields）写入临时通知后
    只清 BM25 与 DB，**忘了向量索引没有单条删除**，于是留下孤儿。
    后果是可检索范围大于库内容，且随执行顺序变化 ——
    相关性阈值用例因此出现"单跑绿、全量跑红"。

    这条断言把不变量钉死：任何新增用例若再泄漏索引，会在这里立刻暴露，
    而不是变成一个隐蔽的顺序依赖。
    """
    from sqlalchemy import func, select

    from app.db import SessionLocal
    from app.models import Notice
    from app.services.vector_store import get_store

    with SessionLocal() as db:
        n_notices = db.execute(select(func.count(Notice.id))).scalar_one()
    assert get_store().size == n_notices, (
        f"向量索引 {get_store().size} 条 != 库内通知 {n_notices} 条："
        "说明有用例写入后未清理索引（见 _reload_indexes_from_db）"
    )
