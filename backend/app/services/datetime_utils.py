"""中文时间表达归一化。

支持：2026年9月3日 / 9月3日14:30 / 9/3 / 明天下午三点 / 本周五 / 下周一 8:00 /
3天后 / 截止时间 9月10日（自动补 23:59）等常见校园通知写法。
输出统一为 naive datetime（本地时区语义），并按上下文关键词区分「活动时间」与「截止时间」。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta

__all__ = ["DateHit", "extract_datetimes", "normalize_text", "pick_times"]

_CN_DIGITS = {"零": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5,
              "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}

DEADLINE_KWS = ("截止", "截至", "最晚", "不迟于", "deadline", "ddl", "提交时间", "提交截止",
                "前提交", "上交", "交至", "报名截止", "务必于", "之前")
# 调课/改期语境：这类时间点优先级最高（是最终生效时间）
RESCHEDULE_KWS = ("调整至", "调整到", "调整为", "调至", "调到", "改为", "改到", "改至",
                  "变更为", "变更至", "顺延至", "延期至", "推迟到", "推迟至", "提前到", "提前至")
# 失效语境：原定时间，不应作为最终时间
# 「原」单独收录：覆盖「由原9月1日推迟到9月8日」这类省略「定」的写法
STALE_KWS = ("原定", "原为", "原计划", "原安排", "原课表", "取消", "原")
EVENT_KWS = ("时间", "举办", "开始", "开讲", "举行", "上课", "考试", "集合", "报到",
             "签到", "地点", "活动", "讲座", "于")

_FULL_HALF = str.maketrans(
    "０１２３４５６７８９：－／．（）　", "0123456789:-/.() "
)


def normalize_text(text: str) -> str:
    """全角转半角 + 压缩空白，避免正则漏匹配。"""
    if not text:
        return ""
    text = text.translate(_FULL_HALF)
    text = text.replace("\u00a0", " ")
    return re.sub(r"[ \t]+", " ", text)


def _cn_to_int(token: str) -> int | None:
    """一二三…三十一 → int；已是阿拉伯数字则直接转。"""
    token = token.strip()
    if token.isdigit():
        return int(token)
    if not token or any(ch not in _CN_DIGITS for ch in token):
        return None
    if token == "十":
        return 10
    if len(token) == 1:
        return _CN_DIGITS[token]
    if token[0] == "十":  # 十一 ~ 十九
        return 10 + _CN_DIGITS[token[1]]
    if "十" in token:  # 二十 / 二十三 / 三十一
        head, _, tail = token.partition("十")
        return _CN_DIGITS[head] * 10 + (_CN_DIGITS[tail] if tail else 0)
    return None


@dataclass
class DateHit:
    raw: str
    dt: datetime
    kind: str = "unknown"          # deadline | event | unknown
    has_time: bool = False
    start: int = 0
    meta: dict = field(default_factory=dict)


# ---------- 正则 ----------
_NUM = r"\d{1,2}|[一二三四五六七八九十]{1,3}"
RE_DATE_YMD = re.compile(
    r"(?:(\d{4})\s*[年\-/])?\s*(\d{1,2})\s*[月\-/]\s*(\d{1,2})\s*[日号]?"
)
RE_DATE_CN = re.compile(rf"({_NUM})\s*月\s*({_NUM})\s*[日号]")
RE_RELDAY = re.compile(r"(今天|今日|今晚|明晚|明天|明日|后天|大后天|当天)")
RE_WEEKDAY = re.compile(
    r"(本周|这周|这个?星期|本星期|下周|下个?星期|周|星期)\s*([一二三四五六日天1-7])"
)
RE_OFFSET = re.compile(rf"({_NUM})\s*(天|小时|周|个?工作日)\s*(?:之)?后")
RE_TIME_COLON = re.compile(r"(上午|下午|晚上|中午|早上|凌晨|傍晚)?\s*(\d{1,2})\s*[:]\s*(\d{2})")
RE_TIME_CN = re.compile(
    rf"(上午|下午|晚上|中午|早上|凌晨|傍晚)?\s*({_NUM})\s*[点时](?:\s*(半|{_NUM})\s*分?)?"
)
# 「3天后交 / 两周后提交」这类结构等价于截止时间
RE_DEADLINE_TAIL = re.compile(r"后\s*(?:交|提交|上交|完成|截止|结束)(?!流)")
# 「9月30日前提交」「本周五前修好」：日期/星期后紧跟「前」= 截止边界而非事件时间。
# 注意 _classify 的 right 从命中起点开始，本身含日期文本，故用 search 而非 match。
RE_DEADLINE_FRONT = re.compile(r"[日号天一二三四五六]\s*前")
# 纯时间（无日期）只有在时间语境里才采信，避免把「共有4点要求」误判为时间
RE_TIME_CONTEXT = re.compile(
    r"(?:时间|上午|下午|晚上|中午|早上|凌晨|傍晚|晚|开始|集合|报到|签到|截止|之前|至|到|于|前)\s*$"
)


def _classify(text: str, pos: int) -> str:
    """按命中位置前后的关键词判定时间语义：reschedule > deadline > stale > event。"""
    left = text[max(0, pos - 8):pos].lower()
    right = text[pos:pos + 14].lower()
    if any(kw in left for kw in RESCHEDULE_KWS):
        return "reschedule"
    if (any(kw in left for kw in DEADLINE_KWS) or any(kw in right for kw in DEADLINE_KWS)
            or RE_DEADLINE_TAIL.search(right) or RE_DEADLINE_FRONT.search(right)):
        return "deadline"
    if any(kw in left for kw in STALE_KWS):
        return "stale"
    if any(kw in left for kw in EVENT_KWS):
        return "event"
    return "unknown"


def _time_from_match(m: re.Match[str], colon: bool) -> tuple[int, int] | None:
    if colon:
        return _apply_meridiem(int(m.group(2)), m.group(1)), int(m.group(3))
    hour = _cn_to_int(m.group(2))
    if hour is None or hour > 24:
        return None
    raw_min = m.group(3)
    minute = 30 if raw_min == "半" else ((_cn_to_int(raw_min) or 0) if raw_min else 0)
    return _apply_meridiem(hour, m.group(1)), min(minute, 59)


def _parse_time(window: str, max_gap: int = 5) -> tuple[int, int, int] | None:
    """在日期后的窗口里找紧邻的时间，返回 (hour, minute, 匹配结束偏移)。

    max_gap 限制时间必须紧跟日期（只允许「的」「空格」等少量间隔字符），
    否则「9月5日布置，9月12日 20:00」会把 20:00 错挂到 9月5日 上。
    """
    best: tuple[int, int, int] | None = None
    for pattern, colon in ((RE_TIME_COLON, True), (RE_TIME_CN, False)):
        m = pattern.search(window)
        if not m or m.start() > max_gap:
            continue
        parsed = _time_from_match(m, colon)
        if parsed and (best is None or m.end() < best[2]):
            best = (parsed[0], parsed[1], m.end())
    return best


def _apply_meridiem(hour: int, marker: str | None) -> int:
    if marker in ("下午", "晚上", "傍晚") and hour < 12:
        hour += 12
    elif marker == "中午" and hour < 12:
        hour = 12 if hour in (12, 0) else hour + 12
    elif marker in ("上午", "早上", "凌晨") and hour == 12:
        hour = 0
    elif marker is None and 1 <= hour <= 7:
        hour += 12  # 校园通知里的「3点」基本指下午
    return hour % 24


def _roll_year(dt: datetime, base: datetime, explicit_year: bool) -> datetime:
    """未写年份且日期已过去 60 天以上时顺延到次年。"""
    if explicit_year:
        return dt
    if (base - dt).days > 60:
        try:
            return dt.replace(year=dt.year + 1)
        except ValueError:  # 2/29
            return dt + timedelta(days=365)
    return dt


def _weekday_index(token: str) -> int | None:
    if token in ("日", "天", "7"):
        return 6
    v = _cn_to_int(token)
    if v is None or not 1 <= v <= 6:
        return None
    return v - 1


def _finalize(dt: datetime, has_time: bool, kind: str) -> datetime:
    """日期级精度补默认时间：截止 23:59，其他 09:00。"""
    if has_time:
        return dt.replace(second=0, microsecond=0)
    if kind == "deadline":
        return dt.replace(hour=23, minute=59, second=0, microsecond=0)
    return dt.replace(hour=9, minute=0, second=0, microsecond=0)


def extract_datetimes(text: str, base: datetime | None = None) -> list[DateHit]:
    """抽取文本中的全部时间点，按出现顺序返回并去重。"""
    if not text:
        return []
    base = (base or datetime.now()).replace(second=0, microsecond=0)
    src = normalize_text(text)
    hits: list[DateHit] = []
    consumed: list[tuple[int, int]] = []

    def overlaps(s: int, e: int) -> bool:
        return any(s < ce and cs < e for cs, ce in consumed)

    def add(raw: str, day: datetime, span: tuple[int, int], explicit_year: bool = True) -> None:
        s, e = span
        if overlaps(s, e):
            return
        kind = _classify(src, s)
        window = src[e:e + 14].split("\n")[0]
        tm = _parse_time(window)
        has_time = tm is not None
        if tm:
            day = day.replace(hour=tm[0], minute=tm[1])
        day = _roll_year(day, base, explicit_year)
        consumed.append((s, e + tm[2] if tm else e))  # 连带吃掉时间片段，避免被后续重复命中
        hits.append(
            DateHit(raw=raw.strip(), dt=_finalize(day, has_time, kind), kind=kind,
                    has_time=has_time, start=s)
        )

    # 1) 2026年9月3日 / 9月3日 / 9/3
    for m in RE_DATE_YMD.finditer(src):
        year, month, dayv = m.group(1), int(m.group(2)), int(m.group(3))
        if not (1 <= month <= 12 and 1 <= dayv <= 31):
            continue
        # 排除 "1.5倍" "3/4 的学生" 之类：分隔符为 / 且紧邻数字
        try:
            day = datetime(int(year) if year else base.year, month, dayv)
        except ValueError:
            continue
        add(m.group(0), day, m.span(), explicit_year=bool(year))

    # 2) 九月三日
    for m in RE_DATE_CN.finditer(src):
        month, dayv = _cn_to_int(m.group(1)), _cn_to_int(m.group(2))
        if not month or not dayv or not (1 <= month <= 12 and 1 <= dayv <= 31):
            continue
        try:
            day = datetime(base.year, month, dayv)
        except ValueError:
            continue
        add(m.group(0), day, m.span(), explicit_year=False)

    # 3) 今天 / 明天 / 后天
    rel_map = {"今天": 0, "今日": 0, "当天": 0, "明天": 1, "明日": 1, "后天": 2, "大后天": 3,
               "今晚": 0, "明晚": 1}
    # 只说「今晚」没给具体时刻时，默认 19:00 而不是 00:00
    rel_hour = {"今晚": 19, "明晚": 19}
    for m in RE_RELDAY.finditer(src):
        token = m.group(1)
        day = (base + timedelta(days=rel_map[token])).replace(hour=0, minute=0)
        if token in rel_hour:
            day = day.replace(hour=rel_hour[token])
        add(m.group(0), day, m.span())

    # 4) 本周五 / 下周一 / 周三
    for m in RE_WEEKDAY.finditer(src):
        idx = _weekday_index(m.group(2))
        if idx is None:
            continue
        prefix = m.group(1)
        monday = base - timedelta(days=base.weekday())
        target = monday + timedelta(days=idx)
        if prefix in ("下周", "下星期", "下个星期"):
            target += timedelta(days=7)
        elif prefix in ("周", "星期") and target.date() < base.date():
            target += timedelta(days=7)  # 裸「周三」指最近的将来
        add(m.group(0), target.replace(hour=0, minute=0), m.span())

    # 5) 3天后 / 2小时后 / 1周后
    for m in RE_OFFSET.finditer(src):
        n = _cn_to_int(m.group(1))
        if not n:
            continue
        unit = m.group(2)
        delta = (timedelta(hours=n) if unit == "小时"
                 else timedelta(weeks=n) if unit == "周" else timedelta(days=n))
        target = base + delta
        hit_time = unit == "小时"
        s, e = m.span()
        if overlaps(s, e):
            continue
        kind = _classify(src, s)
        consumed.append((s, e))
        hits.append(DateHit(raw=m.group(0), dt=_finalize(target, hit_time, kind), kind=kind,
                            has_time=hit_time, start=s))

    # 6) 纯时间（无日期）：如「集合时间 3点」「截止 17:00」，按 base 当天处理
    for pattern, colon in ((RE_TIME_COLON, True), (RE_TIME_CN, False)):
        for m in pattern.finditer(src):
            s, e = m.span()
            if overlaps(s, e):
                continue
            left = src[max(0, s - 8):s]
            if not (m.group(1) or colon or RE_TIME_CONTEXT.search(left)):
                continue  # 无时段词、无冒号、无时间语境 → 大概率不是时间
            parsed = _time_from_match(m, colon)
            if not parsed:
                continue
            kind = _classify(src, s)
            day = base.replace(hour=parsed[0], minute=parsed[1])
            consumed.append((s, e))
            hits.append(DateHit(raw=m.group(0).strip(), dt=_finalize(day, True, kind),
                                kind=kind, has_time=True, start=s,
                                meta={"date_source": "base_day"}))

    hits.sort(key=lambda h: h.start)
    unique: list[DateHit] = []
    seen: set[tuple[datetime, str]] = set()
    for h in hits:
        key = (h.dt, h.kind)
        if key in seen:
            continue
        seen.add(key)
        unique.append(h)
    return unique


def pick_times(
    text: str, category: str = "other", base: datetime | None = None
) -> tuple[datetime | None, datetime | None, list[DateHit]]:
    """返回 (event_time, deadline, all_hits)。作业/报修类默认把时间点当截止时间。"""
    hits = extract_datetimes(text, base=base)
    if not hits:
        return None, None, []

    live = [h for h in hits if h.kind != "stale"] or hits  # 「原定X改为Y」只认 Y
    deadline = next((h.dt for h in live if h.kind == "deadline"), None)
    rescheduled = next((h.dt for h in live if h.kind == "reschedule"), None)
    event_hits = [h for h in live if h.kind == "event"]
    # 多个事件时间时优先取带明确时刻的那个，避免选到只有日期、被默认成 09:00 的条目
    event = next((h.dt for h in event_hits if h.has_time), None) or next(
        (h.dt for h in event_hits), None
    )
    unknown = [h for h in live if h.kind == "unknown"]

    if category in ("homework", "repair"):
        # 改期后的时间即新的截止（「由原9月1日推迟到9月8日」）
        if rescheduled is not None:
            deadline = rescheduled
        elif deadline is None:
            candidates = [h for h in live if h.kind != "event"] or live
            deadline = max(c.dt for c in candidates)  # 无关键词时取最晚时间点作截止
        # 作业/报修以截止为主，只有明确的 event 语义才填 event_time（避免与 deadline 双填）
    else:
        event = rescheduled or event or (unknown[0].dt if unknown else None)
        if event is None and deadline is None:
            event = live[0].dt

    return event, deadline, hits
