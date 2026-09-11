"""句级片段选择（建议 10）的测试。

覆盖三层：
1. 分句 —— 中文标点、换行、小数、域名/邮箱、省略号、无标点长句等边界；
2. 打分 —— 意图奖励、标签位折扣、标题降权、长度先验；
3. 选片 —— 短文本直通、长文本命中答案句、确定性、预算与保底。

分句边界不是"锦上添花"：开发过程中就踩到两个真实缺陷 ——
英文句点把 `maker_club@campus.edu.cn` 切成三句；「办公地点」靠着
"大学生活动中心"里的"活动"二字，把"活动地点"顶了下去。
"""
from __future__ import annotations

import pytest

from app.services.snippet import (
    score_sentence,
    select_context,
    select_snippet,
    split_sentences,
)

LONG_POSTER = """关于开展 2026 年校园创客季·智能硬件工作坊的通知

各位同学：
为提升同学们的动手实践能力，丰富校园科技文化生活，校园创客社团联合电子工程学院
共同举办本次智能硬件工作坊。本次活动为创客季系列活动之一，旨在让同学们在动手
实践中理解电子设备的基本结构与回收价值，树立绿色环保与资源循环利用的意识。
往期活动已累计举办六期，覆盖同学 200 余人，涉及旧手机拆解、旧耳机回收、废旧
数据线再利用等主题，活动照片与回顾可查阅社团公众号「校园创客」历史推送。

活动内容涵盖旧手机拆解、电子元件识别与分类、可回收材料处理、简单电路搭建等多个
环节，并邀请电子工程学院张明教授与实验教学中心王磊工程师现场指导。活动全程配备
防护用具与专业工具，无需具备电子专业基础，欢迎各年级同学报名参加。为保证指导
质量与操作安全，本次活动严格控制人数，采用先到先得的报名方式。

一、活动基本信息
活动主题：旧手机拆解与电子元件回收
活动时间：10 月 12 日（周六）14:00 - 17:00
活动地点：图书馆南门集合，统一前往工程训练中心 B203
主办单位：校园创客社团
协办单位：电子工程学院实验教学中心

二、报名方式
报名请扫描下方二维码或登录社团小程序，填写姓名、学号、联系方式后提交。
报名截止时间：10 月 10 日 18:00，逾期不再受理。
名额有限，共 40 人，先到先得，报名成功后请留意短信通知。

三、注意事项
请自备旧手机一部（无需可用），现场提供全套拆解工具与防护手套。
参与同学需全程听从指导，不得擅自操作高压部件与破损电池。
活动结束后可领取第二课堂学分 0.5 分。

四、联系方式
如有疑问请联系社团负责人李同学，手机：138-0000-1234
邮箱：maker_club@campus.edu.cn
办公地点：大学生活动中心 305 室
"""

LONG_TITLE = "关于开展 2026 年校园创客季·智能硬件工作坊的通知"


# ---------------------------------------------------------------------------
# 1. 分句
# ---------------------------------------------------------------------------
def test_split_basic_chinese_punctuation() -> None:
    text = "第一句话。第二句话！第三句话？第四句话；第五句话"
    assert split_sentences(text) == [
        "第一句话。",
        "第二句话！",
        "第三句话？",
        "第四句话；",
        "第五句话",
    ]


def test_split_on_newlines_without_punctuation() -> None:
    """海报体常用换行分条且不加标点，换行必须作为切分点。"""
    text = "活动主题：旧手机拆解\n活动时间：10 月 12 日\n主办单位：创客社团"
    sents = split_sentences(text)
    assert len(sents) == 3
    assert sents[0] == "活动主题：旧手机拆解"
    assert sents[2] == "主办单位：创客社团"


def test_split_keeps_decimal_intact() -> None:
    """小数点不能切句：3.14 / 9.18 要留在同一句里。"""
    text = "圆周率约等于 3.14，报名截止 9.18 当天。"
    sents = split_sentences(text)
    assert len(sents) == 1, f"小数被误切：{sents}"
    assert "3.14" in sents[0] and "9.18" in sents[0]


def test_split_keeps_domain_and_email_intact() -> None:
    """回归：英文句点只在"后接空白或文末"时切句。

    修复前 `maker_club@campus.edu.cn` 被切成 `maker_club@campus.` /
    `edu.` / `cn` 三句，导致引用片段在 `campus.` 处被截断成半截邮箱。
    """
    text = "邮箱：maker_club@campus.edu.cn\n联系电话：027-87654321"
    sents = split_sentences(text)
    assert any("maker_club@campus.edu.cn" in s for s in sents), f"邮箱被切断：{sents}"
    assert not any(s.endswith("campus.") for s in sents)


def test_split_english_sentence_period() -> None:
    """真正的英文句末（句点后接空格）仍应切分。"""
    text = "Please submit on time. Late work is not accepted."
    sents = split_sentences(text)
    assert len(sents) == 2
    assert sents[0] == "Please submit on time."
    assert sents[1] == "Late work is not accepted."


def test_split_ellipsis_requires_two_marks() -> None:
    """两个及以上 `…` 才算句末；单个 `…` 视为句内停顿。"""
    assert split_sentences("还没想好…不过先这样") == ["还没想好…不过先这样"]
    assert len(split_sentences("就这样吧……下一段开始。")) == 2


def test_split_merges_repeated_enders() -> None:
    sents = split_sentences("真的吗？！我不信。。。然后呢")
    assert sents[0] == "真的吗？！"
    assert sents[1] == "我不信。。。"
    assert sents[2] == "然后呢"


def test_split_keeps_trailing_closers_with_sentence() -> None:
    """引号/括号等收尾符号属于上一句，不该孤立成句。"""
    sents = split_sentences("他说：「请准时到场。」然后就走了。")
    assert sents[0].endswith("」")
    assert "然后就走了。" in sents[1]


def test_split_empty_and_whitespace() -> None:
    assert split_sentences("") == []
    assert split_sentences("   \n\t  ") == []


def test_split_long_text_without_any_punctuation() -> None:
    """整篇无标点时至少不能崩、不能丢内容。"""
    text = "活动时间十月十二日下午两点活动地点图书馆南门集合报名截止十月十日"
    sents = split_sentences(text)
    assert sents == [text]


# ---------------------------------------------------------------------------
# 2. 打分
# ---------------------------------------------------------------------------
def test_score_zero_when_no_overlap() -> None:
    assert score_sentence("今天天气不错", "量子力学作业") == 0.0


def test_score_zero_for_empty_query() -> None:
    assert score_sentence("活动地点：图书馆", "") == 0.0


def test_time_cue_prefers_answer_sentence_over_topic_sentence() -> None:
    """核心规则：问时间时，含时间的句子必须压过复述标题的句子。

    实测问题——「《线性代数》期末复习提纲：第1-5章课后习题，提交至学习通。」
    靠堆主题词就能拿高分，但它不含答案；含答案的是「截止时间：9月18日 22:00。」。
    """
    query = "线性代数期末复习提纲什么时候截止"
    topic_sentence = "《线性代数》期末复习提纲：第1-5章课后习题，提交至学习通。"
    answer_sentence = "截止时间：9月18日 22:00。"
    assert score_sentence(answer_sentence, query) > score_sentence(topic_sentence, query)


def test_label_position_beats_incidental_match() -> None:
    """回归：问「活动在哪里举办」时，「活动地点：」必须胜过「办公地点：」。

    踩坑经过——「办公地点：大学生活动中心 305 室」里的「305 室」也匹配地点模式，
    且其前视窗口若放宽到 6 字，就会把"大学生活动中心"里的「活动」也算作
    "活动+地点"相邻，把社团办公室排到了活动场地前面。
    解法：相邻窗口收紧到 3 字，且「地点」作为字段标签归入强特征。
    """
    query = "活动在哪里举办"
    right = "活动地点：图书馆南门集合，统一前往工程训练中心 B203"
    wrong = "办公地点：大学生活动中心 305 室"
    assert score_sentence(right, query) > score_sentence(wrong, query)


def test_vague_time_word_does_not_beat_concrete_time_value() -> None:
    """回归：泛指时间词不能压过具体时间值。

    「为配合期末考试复习，图书馆决定自即日起延长开放时间。」与问题的词面重合
    更高（含"图书馆""期末""开放"），但它只说了"延长"，没说延长到几点；
    真正的答案在「…开放至 23:00」里。若两者同分，引用片段会支撑不了答案。
    """
    query = "图书馆期末开放到几点"
    vague = "为配合期末考试复习，图书馆决定自即日起延长开放时间。"
    concrete = "一楼自习区与二楼阅览室开放至 23:00；"
    assert score_sentence(concrete, query) > score_sentence(vague, query)


def test_method_cue_ignores_topic_noun() -> None:
    """回归：`报名` 是问题宾语（名词），不该被当成"方式"特征。

    「欢迎各年级同学报名参加」与问题的词面重合很高，但它没说"怎么"报名；
    真正的答案在「报名请扫描下方二维码…」里。若把 `报名` 收进方式特征，
    号召句会被误判为方式答案。
    """
    query = "怎么报名参加"
    slogan = "防护用具与专业工具，无需具备电子专业基础，欢迎各年级同学报名参加。"
    real = "报名请扫描下方二维码或登录社团小程序，填写姓名、学号、联系方式后提交。"
    assert score_sentence(real, query) > score_sentence(slogan, query)


def test_contact_cue() -> None:
    query = "有疑问联系谁"
    assert score_sentence("如有疑问请联系社团负责人李同学，手机：138-0000-1234", query) > 0
    assert score_sentence("如有疑问请联系社团负责人李同学，手机：138-0000-1234", query) > (
        score_sentence("活动主题：旧手机拆解与电子元件回收", query)
    )


def test_title_like_sentence_is_penalized() -> None:
    """与标题高度重合的句子降权：标题已在 API 的 title 字段单独展示。"""
    title = "关于《计算机网络》课程调课的通知"
    query = "计算机网络调课到什么时候"
    penalized = score_sentence(title, query, title=title)
    unpenalized = score_sentence(title, query)
    assert penalized < unpenalized


def test_short_fragment_penalized() -> None:
    """过短碎片（如「教务处」）应低于同等相关度的完整句。"""
    query = "教务处联系方式"
    short = score_sentence("教务处", query)
    full = score_sentence("教务处联系电话：027-87654321", query)
    assert short < full


# ---------------------------------------------------------------------------
# 3. select_snippet
# ---------------------------------------------------------------------------
def test_short_text_returned_unchanged() -> None:
    """短于预算的文本整篇返回，不做任何裁剪。"""
    text = "截止时间：9月18日 22:00。"
    assert select_snippet(text, "什么时候截止", max_chars=120) == text


def test_empty_text_returns_empty() -> None:
    assert select_snippet("", "问题") == ""
    assert select_snippet("   ", "问题") == ""


def test_long_text_picks_deadline_sentence() -> None:
    """长通知里挑出报名截止句，而不是开头的标题与引言。"""
    out = select_snippet(LONG_POSTER, "工作坊什么时候截止报名", title=LONG_TITLE)
    assert "报名截止时间" in out
    assert "10 月 10 日" in out
    assert len(out) <= 120


@pytest.mark.parametrize(
    "query,expected",
    [
        ("工作坊什么时候截止报名", "报名截止时间"),
        ("活动在哪里举办", "活动地点"),
        ("怎么报名参加", "报名请扫描"),
        ("有疑问联系谁", "如有疑问请联系"),
    ],
)
def test_poster_queries_select_right_sentence(query: str, expected: str) -> None:
    out = select_snippet(LONG_POSTER, query, title=LONG_TITLE)
    assert expected in out, f"问题「{query}」选到了：{out}"


def test_selection_is_deterministic() -> None:
    """确定性：同一输入必须得到同一输出（测试里有"引用前后一致"的断言）。"""
    outs = {select_snippet(LONG_POSTER, "活动在哪里举办", title=LONG_TITLE) for _ in range(5)}
    assert len(outs) == 1


def test_never_returns_empty_for_nonempty_text() -> None:
    for query in ("", "完全无关的问题 xyz", "活动"):
        assert select_snippet(LONG_POSTER, query, title=LONG_TITLE).strip()


def test_no_overlap_falls_back_to_head() -> None:
    """问题与文本没有实质交集时，退回开头（保持原行为，不做激进裁剪）。

    注意"没有实质交集"不等于"分数恰好为 0"：中文单字重合极易发生，
    例如"量子"的「子」会与正文"电子"的「子」撞上，产生极小非零分。
    真正触发保底的是「最高分低于 _MIN_RELEVANCE」这条判定。
    查询也刻意避开"时间/地点"等意图词，否则会触发意图奖励。
    """
    from app.services.snippet import _MIN_RELEVANCE

    query = "量子纠缠退相干"
    scores = [score_sentence(s, query) for s in split_sentences(LONG_POSTER)]
    assert max(scores) < _MIN_RELEVANCE, f"前提不成立，最高分 {max(scores)} 不该达到相关度下限"

    out = select_snippet(LONG_POSTER, query, title=LONG_TITLE, max_chars=60)
    assert out == LONG_POSTER[:60].strip()


def test_respects_max_chars() -> None:
    for limit in (40, 80, 120, 200):
        out = select_snippet(LONG_POSTER, "报名截止时间", title=LONG_TITLE, max_chars=limit)
        assert len(out) <= limit, f"超出预算：{len(out)} > {limit}"


def test_preserves_original_newlines() -> None:
    """输出取自原文区间，分隔符与原文一致。

    真正的不变量是「片段是原文的逐字切片」—— 满足它，原文里的换行/缩进
    自然被保留；而不满足时说明中间被重新拼接或截断过。
    单句片段本来就不含换行，因此不能简单断言"必须含 \\n"。
    """
    for query in ("怎么报名参加", "有疑问联系谁", "活动在哪里举办"):
        out = select_snippet(LONG_POSTER, query, title=LONG_TITLE)
        assert out in LONG_POSTER, f"片段不是原文的逐字切片：{out!r}"

    # 答案横跨两行（手机与邮箱分行）时，换行必须被保留
    multi = select_snippet(LONG_POSTER, "有疑问联系谁", title=LONG_TITLE)
    assert "\n" in multi, f"跨行答案的换行丢失：{multi!r}"
    assert "138-0000-1234" in multi and "maker_club@campus.edu.cn" in multi


def test_snippet_does_not_cut_mid_sentence_when_possible() -> None:
    """选中的片段应以句末标点或原文边界结束，而不是从字中间截断。"""
    out = select_snippet(LONG_POSTER, "报名截止时间", title=LONG_TITLE)
    assert out.rstrip().endswith(("。", "！", "？", "；", "…")), f"残句：{out!r}"


# ---------------------------------------------------------------------------
# 4. select_context
# ---------------------------------------------------------------------------
def test_context_short_text_unchanged() -> None:
    text = "截止时间：9月18日 22:00。"
    assert select_context(text, "什么时候截止", max_chars=400) == text


def test_context_keeps_relevant_sentence_within_budget() -> None:
    """长通知超预算时，上下文必须包含相关句（而不是只给开头 400 字）。"""
    ctx = select_context(LONG_POSTER, "工作坊什么时候截止报名", title=LONG_TITLE)
    assert "报名截止时间" in ctx
    assert len(ctx) <= 400


def test_context_bypasses_head_truncation_limitation() -> None:
    """反证：单纯取前 400 字会漏掉答案句，句级选择不会。

    这是「上下文按句选」的核心价值 —— 关键信息落在截断点之后时，
    模型原本看不到答案，只能回答「未找到」，而库里分明有。
    """
    offset = LONG_POSTER.find("报名截止时间")
    assert offset >= 400, (
        f"夹具问题：答案句在偏移 {offset}，未超出 400 字预算，该用例失去意义 —— "
        "请把测试海报的引言写长一些"
    )
    assert "报名截止时间" not in LONG_POSTER[:400], "前提：原硬截断会漏掉答案句"

    ctx = select_context(LONG_POSTER, "工作坊什么时候截止报名", title=LONG_TITLE)
    assert "报名截止时间" in ctx
    assert len(ctx) <= 400


def test_context_no_overlap_falls_back_to_head() -> None:
    query = "量子纠缠退相干"
    ctx = select_context(LONG_POSTER, query, title=LONG_TITLE, max_chars=100)
    assert ctx == LONG_POSTER[:100].strip()


def test_context_respects_budget() -> None:
    for limit in (120, 250, 400):
        ctx = select_context(LONG_POSTER, "活动地点在哪里", title=LONG_TITLE, max_chars=limit)
        assert len(ctx) <= limit


# ---------------------------------------------------------------------------
# 5. 端到端：接入 /api/qa 后是否真的生效
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def qa_client():
    import io

    from fastapi.testclient import TestClient

    from app.main import app

    with TestClient(app) as c:
        # 用真实入库链路写入长海报（会走 pipeline），作为后续问答的语料
        c.post(
            "/api/documents/upload",
            files={
                "file": (
                    "maker_workshop_poster.txt",
                    io.BytesIO(LONG_POSTER.encode("utf-8")),
                    "text/plain",
                )
            },
        )
        yield c


def _citation_for(client, query: str) -> dict | None:
    """在 top_k 结果里找出本测试长海报那条引用。

    按「智能硬件」匹配 —— 这个词只出现在本测试的长海报里，
    不会被 conftest 播种的短通知（"校园创客社团旧手机拆解工作坊"）误命中。
    """
    resp = client.post("/api/qa", json={"query": query, "top_k": 5})
    assert resp.status_code == 200, resp.text
    for c in resp.json()["citations"]:
        if "智能硬件" in c["title"]:
            return c
    return None


def test_api_citation_uses_sentence_selection(qa_client) -> None:
    """端到端：/api/qa 的引用片段应是「答案句」，而不是开头 120 字的硬截断。"""
    c = _citation_for(qa_client, "创客季智能硬件工作坊报名什么时候截止")
    assert c is not None, "未召回长海报通知，检查入库或检索"

    assert "报名截止时间" in c["snippet"], f"引用片段没选到答案句：{c['snippet']!r}"
    assert "10 月 10 日" in c["snippet"]
    # 旧的硬截断只会给出开头的标题与引言，不含答案
    assert c["snippet"] != LONG_POSTER[:120].strip()
    assert len(c["snippet"]) <= 120


def test_api_citation_snippet_is_not_truncated_midsentence(qa_client) -> None:
    """引用片段必须是原文的**逐字切片**，且结束在句边界（不是从字中间切断）。

    注意句边界不一定是标点：海报体常以换行分条，换行结尾同样算完整句子。
    """
    c = _citation_for(qa_client, "创客季智能硬件工作坊活动在哪里举办")
    assert c is not None
    snippet = c["snippet"]

    idx = LONG_POSTER.find(snippet)
    assert idx >= 0, f"片段不是原文的逐字切片（说明被字符截断过）：{snippet!r}"

    end = idx + len(snippet)
    at_boundary = (
        end == len(LONG_POSTER)
        or LONG_POSTER[end] == "\n"
        or snippet.rstrip().endswith(("。", "！", "？", "；", "…"))
    )
    assert at_boundary, f"片段结束在句子中间：{snippet!r}（后接 {LONG_POSTER[end:end+10]!r}）"


def test_api_llm_prompt_contains_answer_sentence(qa_client, monkeypatch) -> None:
    """关键回归：喂给 LLM 的上下文必须包含答案句。

    改造前 context 取原文前 400 字，而「报名截止时间」在偏移 400 之后，
    模型根本看不到，只能回答「未找到」—— 而库里分明有答案。
    这里直接抓取请求体，断言 prompt 里出现了答案句。
    """
    import json

    import httpx

    from app.api import qa as qa_module
    from app.providers.llm_client import OpenAICompatClient

    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["prompt"] = json.loads(request.content)["messages"][1]["content"]
        return httpx.Response(
            200, json={"choices": [{"message": {"content": "报名截止时间为 10 月 10 日 18:00 [1]。"}}]}
        )

    def fake_client() -> OpenAICompatClient:
        return OpenAICompatClient(
            base_url="https://mock.test/v1",
            api_key="test-key",
            transport=httpx.MockTransport(handler),
            sleep_fn=lambda _s: None,
        )

    monkeypatch.setattr(qa_module, "_build_llm_client", fake_client)
    monkeypatch.setattr(qa_module.settings, "qa_provider", "auto")
    monkeypatch.setattr(qa_module.settings, "dashscope_api_key", "test-key")

    resp = qa_client.post("/api/qa", json={"query": "创客工作坊报名什么时候截止", "top_k": 5})
    assert resp.status_code == 200
    prompt = captured.get("prompt")
    assert prompt is not None, "未捕获到 LLM 请求"
    assert "报名截止时间" in prompt, (
        "上下文里没有答案句 —— 说明仍在用硬截断，模型看不到关键信息"
    )


# ---------------------------------------------------------------------------
# 6. 端到端：接入 /api/search 结果页
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def search_client(qa_client):
    """复用 qa_client 已入库的长海报语料，避免重复 ingest。"""
    return qa_client


def _search_hit_for(client, query: str) -> dict | None:
    """在 /api/search 结果里找出本测试长海报那条命中。"""
    resp = client.post("/api/search", json={"query": query, "top_k": 5})
    assert resp.status_code == 200, resp.text
    for h in resp.json()["hits"]:
        if "智能硬件" in h["title"]:
            return h
    return None


def test_search_hit_contains_snippet(search_client) -> None:
    """端到端：/api/search 的命中应带 snippet，且是「答案句」而非开头的概括。"""
    h = _search_hit_for(search_client, "创客季智能硬件工作坊报名什么时候截止")
    assert h is not None, "未召回长海报通知，检查入库或检索"

    assert "snippet" in h, "响应缺少 snippet 字段"
    snippet = h["snippet"]
    assert snippet, f"snippet 不应为空：{h!r}"
    assert "报名截止时间" in snippet, f"片段没选到答案句：{snippet!r}"
    assert len(snippet) <= 120, f"片段超出上限：{len(snippet)}"


def test_search_snippet_beats_summary_for_long_notice(search_client) -> None:
    """snippet 必须区别于 summary —— 这正是本次接入的意义。

    summary 是抽取阶段的概括，与查询无关；搜索"地点在哪"时它答非所问。
    """
    h = _search_hit_for(search_client, "创客季智能硬件工作坊活动在哪里举办")
    assert h is not None
    assert h["snippet"] != h["summary"], (
        "snippet 与 summary 相同 —— 说明没有按查询选句，退回了概括"
    )
    assert "地点" in h["snippet"], f"片段没选到地点句：{h['snippet']!r}"


def test_search_keeps_backward_compatible_fields(search_client) -> None:
    """契约兼容：加了 snippet 后，原有字段必须一个不少（前端老逻辑仍可用）。"""
    resp = search_client.post("/api/search", json={"query": "创客工作坊", "top_k": 3})
    assert resp.status_code == 200
    body = resp.json()
    assert set(body) >= {"query", "backend", "hits"}
    for h in body["hits"]:
        for field in ("notice_id", "score", "title", "category", "summary", "deadline"):
            assert field in h, f"原有字段 {field} 丢失，破坏向后兼容"


def test_search_snippet_traces_back_to_source_text(search_client) -> None:
    """强断言：snippet 必须能逐字追溯回通知原文，而非拼接/编造的文本。

    取原文的路径：/api/search 命中 → /api/notices 拿 document_id
    → /api/documents/{id} 拿 raw_text。回溯本身就是可追溯性的证明。
    """
    notices = {n["id"]: n for n in search_client.get("/api/notices").json()}

    for query in ("宿舍报修怎么申请", "创客工作坊", "期末考试"):
        resp = search_client.post("/api/search", json={"query": query, "top_k": 5})
        assert resp.status_code == 200
        for h in resp.json()["hits"]:
            snippet = h.get("snippet")
            assert snippet, f"查询 {query!r} 的命中未返回 snippet"

            notice = notices.get(h["notice_id"])
            if notice is None:
                continue
            detail = search_client.get(f"/api/documents/{notice['document_id']}")
            if detail.status_code != 200:
                continue
            raw = (detail.json().get("raw_text") or "").strip()
            if not raw:
                continue
            # 单句超长时后端会截断并补省略号，比对时去掉它
            assert snippet in raw or snippet.rstrip("…") in raw, (
                f"snippet 无法在原文中找到，可能被拼接或改动：\n"
                f"  snippet={snippet!r}\n  raw={raw[:200]!r}"
            )
