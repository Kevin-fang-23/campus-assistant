from __future__ import annotations

from datetime import datetime

import pytest

from app.services.datetime_utils import extract_datetimes, pick_times

BASE = datetime(2026, 8, 24, 10, 0)  # 周一


@pytest.mark.parametrize(
    "text,expected",
    [
        ("考试时间 2026年9月3日 14:30", datetime(2026, 9, 3, 14, 30)),
        ("9月3日下午2点开始", datetime(2026, 9, 3, 14, 0)),
        ("明天上午8:00集合", datetime(2026, 8, 25, 8, 0)),
        ("活动时间：本周五晚上七点", datetime(2026, 8, 28, 19, 0)),
        ("下周一上课", datetime(2026, 8, 31, 9, 0)),
        ("3天后交", datetime(2026, 8, 27, 23, 59)),
        ("讲座 9/5 15:00", datetime(2026, 9, 5, 15, 0)),
        ("提交截止时间 9月10日", datetime(2026, 9, 10, 23, 59)),
        ("集合时间 3点", datetime(2026, 8, 24, 15, 0)),
    ],
)
def test_single_expression(text: str, expected: datetime) -> None:
    hits = extract_datetimes(text, base=BASE)
    assert hits, f"未解析出时间: {text}"
    assert hits[0].dt == expected


def test_deadline_kind_and_default_time() -> None:
    hits = extract_datetimes("作业务必于9月10日之前提交", base=BASE)
    assert hits[0].kind == "deadline"
    assert hits[0].dt == datetime(2026, 9, 10, 23, 59)


def test_reschedule_wins_over_original() -> None:
    text = "原定本周三下午的课程调整至本周五下午2:30，地点改为A305"
    event, deadline, hits = pick_times(text, category="course_notice", base=BASE)
    assert event == datetime(2026, 8, 28, 14, 30)
    assert {h.kind for h in hits} >= {"stale", "reschedule"}
    assert deadline is None


def test_homework_picks_latest_as_deadline() -> None:
    event, deadline, _ = pick_times("第三次作业，9月5日布置，9月12日 20:00 收齐",
                                   category="homework", base=BASE)
    assert deadline == datetime(2026, 9, 12, 20, 0)


def test_past_date_rolls_to_next_year() -> None:
    hits = extract_datetimes("报名时间 1月5日", base=datetime(2026, 8, 24, 10, 0))
    assert hits[0].dt.year == 2027


def test_no_datetime_returns_empty() -> None:
    assert extract_datetimes("请同学们注意教室卫生") == []
