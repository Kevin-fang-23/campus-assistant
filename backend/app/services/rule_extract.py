"""规则版分类与信息抽取。

作用：1) 无 VLM Key 时作为 Mock 实现，保证全链路可跑通；
      2) 有 VLM 时作为兜底与交叉校验（模型漏字段时用规则补齐）。
"""
from __future__ import annotations

import re
from datetime import datetime

from ..models import Category
from .datetime_utils import normalize_text, pick_times

CATEGORY_KEYWORDS: dict[str, tuple[tuple[str, float], ...]] = {
    Category.HOMEWORK: (
        ("作业", 3.0), ("实验报告", 3.0), ("课程设计", 2.5), ("论文", 2.0), ("提交", 1.5),
        ("上交", 1.5), ("ddl", 2.0), ("deadline", 2.0), ("截止", 1.2), ("习题", 2.0),
        ("大作业", 3.0), ("答辩", 1.5), ("查重", 2.0), ("提交格式", 2.0),
    ),
    Category.COURSE_NOTICE: (
        ("调课", 3.0), ("停课", 3.0), ("补课", 3.0), ("上课", 2.0), ("教室", 1.5),
        ("教务", 2.5), ("考试", 2.5), ("课表", 2.5), ("选课", 2.5), ("期末", 1.5),
        ("授课", 2.0), ("学分", 1.5), ("请假", 1.0), ("串课", 2.5),
    ),
    Category.ACTIVITY: (
        ("讲座", 3.0), ("活动", 2.0), ("报名", 2.0), ("社团", 2.5), ("比赛", 2.5),
        ("招新", 3.0), ("沙龙", 2.5), ("音乐会", 3.0), ("嘉宾", 2.0), ("主办", 1.5),
        ("志愿", 2.0), ("宣讲会", 3.0), ("海报", 1.5), ("免费入场", 2.0),
    ),
    Category.REPAIR: (
        ("报修", 3.5), ("维修", 3.0), ("故障", 2.5), ("漏水", 3.0), ("空调", 2.0),
        ("电路", 2.0), ("后勤", 2.0), ("宿舍", 1.2), ("水管", 3.0), ("灯管", 2.5),
        ("门锁", 2.5), ("停水", 2.5), ("停电", 2.5), ("热水器", 2.5),
    ),
}

RE_PHONE = re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)")
RE_TEL = re.compile(r"(?<!\d)(?:0\d{2,3}-)?\d{7,8}(?:-\d{1,4})?(?!\d)")
RE_EMAIL = re.compile(r"[\w.\-+]+@[\w\-]+\.[\w.\-]+")
RE_QQ = re.compile(r"(?:QQ群?|群号)[:：\s]*(\d{5,12})", re.I)
RE_WECHAT = re.compile(r"(?:微信|wechat)[:：\s]*([A-Za-z0-9_\-]{4,20})", re.I)
RE_LOC_LABEL = re.compile(
    r"(?:地点|地址|举办地点|集合地点|教室|房间|位置)\s*"
    r"(?:[:：]|改为|调整为|调整到|调至|变更为|改到|为|在|是)\s*([^\n，。;；]{2,30})"
)
RE_LOC_GUESS = re.compile(
    r"([\u4e00-\u9fa5A-Za-z0-9]{0,8}?(?:教学楼|报告厅|体育馆|图书馆|礼堂|实验室|活动室|机房|"
    r"教室|会议室|操场|食堂|号楼|楼)[A-Za-z0-9\-]{0,8})"
)
# 位置字段前常混入的动词/标签，需剥离
RE_LOC_NOISE = re.compile(
    r"^(?:地点|地址|位置|举办地点|集合地点|房间)?(?:改为|调整为|调整到|调至|变更为|改到|为|在|于|是)?"
)
RE_ISSUER_LABEL = re.compile(
    r"(?:主办|承办|发布单位|发布方|通知单位|来自|发布人)[:：]?\s*([^\n，。;；]{2,24})"
)
RE_ISSUER_GUESS = re.compile(
    r"([\u4e00-\u9fa5]{0,10}(?:教务处|学生处|后勤集团|后勤处|保卫处|团委|学生会|"
    r"研究生院|图书馆|社团联合会|学工办|辅导员|学院|书院))"
)
RE_ISSUER_DEPT = re.compile(r"([\u4e00-\u9fa5]{2,10}系)(?![统列数别])")
RE_COURSE_BOOK = re.compile(r"《([^》]{2,30})》")
RE_COURSE_LABEL = re.compile(r"(?:课程|科目|课程名称)[:：]\s*([^\n，。;；]{2,24})")
RE_COURSE_GUESS = re.compile(
    r"([\u4e00-\u9fa5]{2,14}(?:导论|原理|基础|设计|实验|语言程序设计|数据结构|"
    r"操作系统|计算机网络|数据库|编译原理|机器学习|离散数学|线性代数|概率论))"
)
RE_DORM = re.compile(r"([\u4e00-\u9fa5A-Za-z0-9]{0,8}\d{1,2}号?楼[A-Za-z]?\d{2,4}(?:室|房)?)")
_NOISE = re.compile(r"^[\s\-—=*#>·•\d.、]+|[\s\-—=*#>·•]+$")


def _clean_line(line: str) -> str:
    return _NOISE.sub("", line).strip()


def classify(text: str) -> tuple[str, float, dict[str, float]]:
    """关键词加权打分分类，返回 (category, confidence, 各类得分)。"""
    low = normalize_text(text).lower()
    scores: dict[str, float] = {}
    for cat, kws in CATEGORY_KEYWORDS.items():
        scores[cat] = round(sum(w for kw, w in kws if kw.lower() in low), 2)
    best = max(scores, key=lambda k: scores[k])
    top = scores[best]
    if top <= 0:
        return Category.OTHER, 0.2, scores
    second = sorted(scores.values(), reverse=True)[1] if len(scores) > 1 else 0.0
    margin = (top - second) / max(top, 1e-6)
    conf = min(0.92, 0.45 + min(top, 8.0) / 8.0 * 0.3 + margin * 0.2)
    return best, round(conf, 3), scores


def _guess_title(text: str, category: str) -> str:
    lines = [_clean_line(ln) for ln in normalize_text(text).splitlines()]
    lines = [ln for ln in lines if len(ln) >= 4]
    if not lines:
        flat = normalize_text(text).strip()
        return flat[:60] or "未命名通知"
    for ln in lines[:6]:
        if any(k in ln for k in ("通知", "公告", "海报", "报名", "作业", "报修", "讲座", "安排")):
            return ln[:80]
    return lines[0][:80]


def _first(pattern: re.Pattern[str], text: str, group: int = 1) -> str | None:
    m = pattern.search(text)
    if not m:
        return None
    val = (m.group(group) or "").strip(" ：:，,。.")
    return val or None


def _clean_location(value: str | None) -> str | None:
    """剥离「地点改为」「地址：」等前缀，只留真正的位置。"""
    if not value:
        return None
    cleaned = RE_LOC_NOISE.sub("", value.strip()).strip(" ：:，,。.、")
    return cleaned or None


def _collect_contacts(text: str) -> list[str]:
    out: list[str] = []
    for m in RE_PHONE.finditer(text):
        out.append(f"电话 {m.group(0)}")
    for m in RE_EMAIL.finditer(text):
        out.append(f"邮箱 {m.group(0)}")
    for m in RE_QQ.finditer(text):
        out.append(f"QQ群 {m.group(1)}")
    for m in RE_WECHAT.finditer(text):
        out.append(f"微信 {m.group(1)}")
    if not any(c.startswith("电话") for c in out):
        for m in RE_TEL.finditer(text):
            token = m.group(0)
            if "-" in token or len(token) in (7, 8):
                out.append(f"电话 {token}")
                break
    seen, uniq = set(), []
    for c in out:
        if c not in seen:
            seen.add(c)
            uniq.append(c)
    return uniq[:6]


def _summarize(text: str, limit: int = 140) -> str:
    flat = re.sub(r"\s+", " ", normalize_text(text)).strip()
    if len(flat) <= limit:
        return flat
    cut = flat[:limit]
    for sep in ("。", "；", ";", "，"):
        idx = cut.rfind(sep)
        if idx > limit * 0.5:
            return cut[: idx + 1]
    return cut + "…"


def extract(text: str, category: str, base: datetime | None = None) -> dict:
    """规则抽取结构化字段。"""
    src = normalize_text(text)
    event_time, deadline, hits = pick_times(src, category=category, base=base)

    location = _clean_location(_first(RE_LOC_LABEL, src) or _first(RE_LOC_GUESS, src))
    issuer = (_first(RE_ISSUER_LABEL, src) or _first(RE_ISSUER_GUESS, src)
              or _first(RE_ISSUER_DEPT, src))
    course = _first(RE_COURSE_LABEL, src) or _first(RE_COURSE_BOOK, src) or _first(
        RE_COURSE_GUESS, src
    )
    if category == Category.REPAIR:
        location = _first(RE_DORM, src) or location

    tags = sorted({kw for kw, _ in CATEGORY_KEYWORDS.get(category, ()) if kw in src.lower()})[:6]

    conf = 0.35
    conf += 0.15 if (event_time or deadline) else 0.0
    conf += 0.1 if location else 0.0
    conf += 0.1 if issuer else 0.0
    conf += 0.08 if course else 0.0
    conf += 0.07 if len(src) > 60 else 0.0
    if category == Category.OTHER:
        conf -= 0.15

    return {
        "category": category,
        "title": _guess_title(src, category),
        "summary": _summarize(src),
        "issuer": issuer,
        "location": location,
        "course": course,
        "event_time": event_time,
        "deadline": deadline,
        "contacts": _collect_contacts(src),
        "tags": tags,
        "extra": {
            "datetime_hits": [
                {"raw": h.raw, "dt": h.dt.isoformat(), "kind": h.kind, "has_time": h.has_time}
                for h in hits
            ],
            "extractor": "rule",
        },
        "confidence": round(min(conf, 0.9), 3),
    }
