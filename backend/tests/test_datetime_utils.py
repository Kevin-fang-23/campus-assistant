from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from app.services.datetime_utils import extract_datetimes, pick_times, to_naive_local

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


# ---------------------------------------------------------------------------
# to_naive_local：客户端 tz-aware 时间的归一化
# ---------------------------------------------------------------------------


def test_to_naive_local_passes_through_naive_and_none() -> None:
    """naive 输入必须原样返回（同一对象），保证既有抽取链路零开销、零影响。"""
    v = datetime(2026, 10, 1, 16, 12)
    assert to_naive_local(v) is v
    assert to_naive_local(None) is None


def test_to_naive_local_converts_aware_to_local_wall_clock() -> None:
    """aware 输入 → 同一瞬时在**本地时区**的墙钟（tmzinfo 去掉）。"""
    offset = datetime.now().astimezone().utcoffset()
    aware = datetime(2026, 10, 1, 8, 12, tzinfo=timezone.utc)

    out = to_naive_local(aware)

    assert out is not None and out.tzinfo is None
    assert out == (aware + offset).replace(tzinfo=None)


def test_to_naive_local_does_not_fall_back_to_naive_utc() -> None:
    """回归护栏：不得用 `astimezone(utc)` 降级。

    naive 语义在本项目里是**本地墙钟**（相对时间解析、截止补 23:59 都以
    `datetime.now()` 为基准）。若误降级为 naive-UTC，在 GMT+8 等非 UTC 时区下
    用户填的 16:12 会被静默存成 08:12 —— 不报错，比崩溃更难发现。
    """
    offset = datetime.now().astimezone().utcoffset()
    if not offset:
        pytest.skip("本机为 UTC，无法区分 naive-UTC 与 naive-本地")

    aware = datetime(2026, 10, 1, 8, 12, tzinfo=timezone.utc)
    assert to_naive_local(aware) != aware.replace(tzinfo=None)


def test_roundtrip_aware_input_matches_user_wall_clock() -> None:
    """端到端语义：浏览器把本地 16:12 转成 08:12Z 提交，后端必须还原成 16:12。"""
    offset = datetime.now().astimezone().utcoffset() or timedelta(0)
    user_wall = datetime(2026, 10, 1, 16, 12)          # 用户在表单里选的
    as_utc_instant = user_wall - offset                # new Date(...).toISOString()
    aware = as_utc_instant.replace(tzinfo=timezone.utc)

    assert to_naive_local(aware) == user_wall


def test_no_datetime_returns_empty() -> None:
    assert extract_datetimes("请同学们注意教室卫生") == []
