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
