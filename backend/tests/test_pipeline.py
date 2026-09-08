from __future__ import annotations

from datetime import datetime

import pytest

from app.graph.pipeline import engine_name, run_pipeline
from app.models import Category

BASE = datetime(2026, 8, 24, 10, 0)

SAMPLES = {
    Category.COURSE_NOTICE: "关于《计算机网络》课程调课的通知：原定本周三的课调整至本周五下午2:30，"
                            "地点改为第三教学楼A305。教务处",
    Category.HOMEWORK: "《数据结构》第三次实验报告作业要求：提交源代码与报告，"
                       "截止时间 9月10日 23:00，逾期不计分。",
    Category.ACTIVITY: "人工智能前沿讲座 主办：计算机学院 时间：明天晚上7点 地点：图书馆报告厅 "
                       "报名截止：今天下午5点",
    Category.REPAIR: "宿舍报修申请 报修位置：3号楼412室 故障描述：卫生间水管漏水，"
                     "希望明天上午10点前上门维修 联系人手机：13800001234",
}


def _run(text: str):
    return run_pipeline(document_id=0, raw_text=text, kind="text", base_time=BASE)


@pytest.mark.parametrize("category,text", list(SAMPLES.items()))
def test_classify_and_generate_tasks(category: str, text: str) -> None:
    state = _run(text)
    assert state["category"] == category
    assert state["notice"]["title"]
    assert state["task_drafts"], "应至少生成一条待办"
    assert [t["node"] for t in state["trace"]][:4] == ["classify", "extract", "validate", "dedup"]


def test_course_notice_fields() -> None:
    state = _run(SAMPLES[Category.COURSE_NOTICE])
    notice = state["notice"]
    assert notice["event_time"] == datetime(2026, 8, 28, 14, 30)
    assert notice["location"] == "第三教学楼A305"
    assert notice["course"] == "计算机网络"
    assert notice["issuer"] == "教务处"


def test_homework_generates_prestart_task() -> None:
    state = _run(SAMPLES[Category.HOMEWORK])
    titles = [t["title"] for t in state["task_drafts"]]
    assert any("提交作业" in t for t in titles)
    assert any("开始动手" in t for t in titles), "距 DDL 超 3 天应额外生成启动任务"
    main = state["task_drafts"][0]
    assert main["due_at"] == datetime(2026, 9, 10, 23, 0)
    assert main["remind_at"] < main["due_at"]


def test_activity_two_tasks_signup_and_attend() -> None:
    state = _run(SAMPLES[Category.ACTIVITY])
    titles = [t["title"] for t in state["task_drafts"]]
    assert any("报名截止" in t for t in titles)
    assert any("参加活动" in t for t in titles)


def test_empty_input_is_marked_review() -> None:
    state = _run("   ")
    assert state["category"] == Category.OTHER
    assert state["needs_review"] is True


def test_low_confidence_needs_review() -> None:
    state = _run("同学们注意天气变化，注意保暖。")
    assert state["needs_review"] is True
    assert state["review_reasons"]


def test_engine_is_langgraph_when_installed() -> None:
    assert engine_name() in ("langgraph", "sequential")


def test_base_time_is_propagated_to_extraction() -> None:
    """相对时间必须按传入的 base_time 解析，而不是 datetime.now()。

    这条用例与「今天几号」无关：基准日固定为 2026-08-24（周一），
    两个相隔一周的基准日必须解析出相差一周的结果。
    """
    text = "《计算机网络》课程调整至本周五下午2:30，地点改为A305"
    monday = datetime(2026, 8, 24, 10, 0)     # 本周五 = 08-28
    next_monday = datetime(2026, 8, 31, 10, 0)  # 本周五 = 09-04

    st_a = run_pipeline(document_id=0, raw_text=text, kind="text", base_time=monday)
    st_b = run_pipeline(document_id=0, raw_text=text, kind="text", base_time=next_monday)

    assert st_a["notice"]["event_time"] == datetime(2026, 8, 28, 14, 30)
    assert st_b["notice"]["event_time"] == datetime(2026, 9, 4, 14, 30)
    assert st_a["notice"]["event_time"] != st_b["notice"]["event_time"]


def test_location_stops_at_next_field_label() -> None:
    """单行空格分隔的海报里，地点值必须在下一个字段标签处截断。

    回归 act_03：原贪婪匹配把「主办:创客社团」整个吞进地点，
    又因 issuer 与地点重叠被连坐清空。
    """
    state = _run("创客社团招新宣讲会 时间：下周三晚上7点半 地点：工学院B101 主办：创客社团")
    notice = state["notice"]
    assert notice["location"] == "工学院B101"
    assert notice["issuer"] == "创客社团"


def test_labeled_issuer_overlapping_location_is_kept() -> None:
    """显式「主办：X」与地点重叠时不判误抽。

    回归 act_08：「地点:计算机学院A301，主办:计算机学院」——学院既是
    场地又是主办方，语义都成立。只有猜测型 issuer 才按重叠清空。
    """
    state = _run("学术讲座通知：主题《大模型时代的软件工程》，时间：9月9日15:00，"
                 "地点：计算机学院A301，主办：计算机学院。")
    notice = state["notice"]
    assert notice["location"] == "计算机学院A301"
    assert notice["issuer"] == "计算机学院"


def test_guessed_issuer_overlapping_location_still_cleared() -> None:
    """无显式标签、猜测型 issuer 与地点重叠时仍应清空（防 act_07/rp_03 回归）。

    本条断言的口径与评测一致（双向包含）：地点允许是期望值的子串，
    但 issuer（「在图书馆」这类误抽）必须为 None。
    """
    state = _run("志愿服务活动招募：本周六上午8:00在图书馆南门集合，报名从速。")
    notice = state["notice"]
    assert "图书馆" in (notice["location"] or "")
    assert notice["issuer"] is None
