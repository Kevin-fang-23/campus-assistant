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
        # 「教室」已移出：它是地点词而非课程信号，会把「注意教室卫生」误判为教务通知
        ("调课", 3.0), ("停课", 3.0), ("补课", 3.0), ("上课", 2.0),
        ("教务", 2.5), ("考试", 2.5), ("课表", 2.5), ("选课", 2.5), ("期末", 1.5),
        ("授课", 2.0), ("学分", 1.5), ("请假", 1.0), ("串课", 2.5), ("答疑", 2.0),
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
# 字段值边界前瞻：值在「下一个字段标签+冒号」（如「地点:X 主办:Y」）、
# 硬分隔符或串尾处结束。单行海报常为空格分隔的字段序列，贪婪匹配会把
# 后续字段（如「主办:创客社团」）整个吞进地点里。
_FIELD_BOUND = (
    r"(?=\s*(?:主办|承办|发布单位|发布方|通知单位|发布人|来自|"
    r"时间|地点|地址|报名|截止|联系|电话|主讲|嘉宾)\s*[:：]|[\n，。;；]|$)"
)
RE_LOC_LABEL = re.compile(
    r"(?:地点|地址|举办地点|集合地点|活动地点|上课地点|考场|教室|房间|位置)\s*"
    r"(?:[:：]|改为|调整为|调整到|调至|变更为|改到|为|在|是)\s*"
    r"([^\n，。;；]{2,30}?)" + _FIELD_BOUND
)
# 前导仅允许中文且最多 6 字：原先允许 A-Za-z0-9 会把 "00在图书馆"、
# "由A201调整为图书馆" 这类噪声一并吞进地点里
RE_LOC_GUESS = re.compile(
    r"([\u4e00-\u9fa5]{0,6}?(?:教学楼|报告厅|体育馆|图书馆|礼堂|实验室|活动室|机房|"
    r"教室|会议室|操场|食堂|号楼|楼|体育场)[A-Za-z0-9\-]{0,8})"
)
# 兜底：裸房间号（A101 / B201），仅在上述两者都未命中时使用
RE_ROOM = re.compile(r"(?<![A-Za-z0-9])([A-Za-z]{1,2}\d{3,4}(?:室)?)(?![A-Za-z0-9])")
# 位置字段前常混入的动词/标签，需剥离
RE_LOC_NOISE = re.compile(
    r"^(?:地点|地址|位置|举办地点|集合地点|活动地点|上课地点|考场|房间)?"
    r"(?:由|从)?(?:调整[为到至]|变更[为到至]|改[为到至]|调至|定于|位于|设在)?"
    r"(?:请|在|于|是|为|到|有同学|有)?"
)
# 含这些词基本可判定为噪声文本误匹配，而非真实地点
RE_LOC_STOPWORDS = re.compile(r"(?:请|注意|爱护|保持|节约|关闭|严禁)")
RE_ISSUER_LABEL = re.compile(
    r"(?:主办|承办|发布单位|发布方|通知单位|来自|发布人)[:：]?\s*"
    r"([^\n，。;；]{2,24}?)" + _FIELD_BOUND
)
RE_ISSUER_GUESS = re.compile(
    r"([\u4e00-\u9fa5]{0,10}(?:教务处|学生处|后勤集团|后勤处|保卫处|团委|学生会|"
    r"研究生院|图书馆|社团联合会|学工办|辅导员|学院|书院))"
)
RE_ISSUER_DEPT = re.compile(r"([\u4e00-\u9fa5]{2,10}系)(?![统列数别])")
RE_COURSE_BOOK = re.compile(r"《([^》]{2,30})》")
RE_COURSE_LABEL = re.compile(r"(?:课程|科目|课程名称)[:：]\s*([^\n，。;；]{2,24})")
# 前缀改为 {0,14}：原先要求至少 2 个前导汉字，导致句首的「操作系统第2次作业」
# 「概率论作业」这类裸课程名抽不到
RE_COURSE_GUESS = re.compile(
    r"([\u4e00-\u9fa5]{0,14}(?:导论|原理|基础|设计|实验|语言程序设计|数据结构|"
    r"操作系统|计算机网络|数据库|编译原理|机器学习|离散数学|线性代数|概率论))"
)
RE_DORM = re.compile(r"([\u4e00-\u9fa5A-Za-z0-9]{0,8}\d{1,2}号?楼[A-Za-z]?\d{2,4}(?:室|房)?)")
_NOISE = re.compile(r"^[\s\-—=*#>·•\d.、]+|[\s\-—=*#>·•]+$")


def _clean_line(line: str) -> str:
    return _NOISE.sub("", line).strip()


# 否定语境：「关空调」「关闭电源…和空调」里的「空调」是节能提醒，不是报修信号。
# 用 10 字窗口而非紧邻字符，才能覆盖「关闭电源、水源和空调」这种隔词修饰。
_NEGATION_MARKERS = ("关", "闭", "爱护", "保持", "节约", "停用")
# 这些词里的「关」不是否定用法
_NOT_NEGATION = ("关于", "相关", "有关", "机关", "海关", "难关", "关照")


def _keyword_present(low: str, kw: str) -> bool:
    """关键词是否以肯定语义出现（存在至少一处未被否定语境修饰的命中）。"""
    start = 0
    while True:
        i = low.find(kw, start)
        if i < 0:
            return False
        window = low[max(0, i - 10):i]
        # 先剔除「关于/相关」这类含「关」却非否定的词，再判定，
        # 否则「关于《软件工程》助教答疑…」里的「关于」会误伤后面的关键词
        for noise in _NOT_NEGATION:
            window = window.replace(noise, "")
        negated = any(m in window for m in _NEGATION_MARKERS)
        if not negated:
            return True
        start = i + len(kw)


def classify(text: str) -> tuple[str, float, dict[str, float]]:
    """关键词加权打分分类，返回 (category, confidence, 各类得分)。"""
    low = normalize_text(text).lower()
    scores: dict[str, float] = {}
    for cat, kws in CATEGORY_KEYWORDS.items():
        scores[cat] = round(sum(w for kw, w in kws if _keyword_present(low, kw.lower())), 2)
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
    raw = value.strip()
    # 噪声文本（"请同学们注意教室卫生"）不应产出地点
    if RE_LOC_STOPWORDS.search(raw):
        return None
    cleaned = RE_LOC_NOISE.sub("", raw).strip(" ：:，,。.、")
    return cleaned or None


def _overlaps(a: str | None, b: str | None) -> bool:
    """两个短串是否互相包含或高度重叠。

    用于剔除「issuer == location」的伪字段：常见的是把发布单位
    （如「图书馆」「教务处」）误同时认作地点，需要二选一保留。
    """
    if not a or not b:
        return False
    if a == b:
        return True
    if a in b or b in a:
        # 更短的一方长度>=2，避免「一」字被另一方误吃
        return min(len(a), len(b)) >= 2
    return False


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

    location = _clean_location(
        _first(RE_LOC_LABEL, src) or _first(RE_LOC_GUESS, src) or _first(RE_ROOM, src)
    )
    issuer = _first(RE_ISSUER_LABEL, src)
    issuer_labeled = bool(issuer)
    if not issuer:
        issuer = _first(RE_ISSUER_GUESS, src) or _first(RE_ISSUER_DEPT, src)
    # 活动类材料里的《》通常是讲座主题或作品名（如《大模型时代的软件工程》《流浪地球3》），
    # 不是课程名，故活动类不走书名号抽取。
    if category == Category.ACTIVITY:
        course = _first(RE_COURSE_LABEL, src) or _first(RE_COURSE_GUESS, src)
    else:
        course = (_first(RE_COURSE_LABEL, src) or _first(RE_COURSE_BOOK, src)
                  or _first(RE_COURSE_GUESS, src))
    if category == Category.REPAIR:
        location = _first(RE_DORM, src) or location

    # 发布方不应是地点的一部分：「图书馆四楼自习室」里的「图书馆」是地点，
    # 不是发布单位。但显式写了「主办：计算机学院」时，学院既是场地
    # （计算机学院A301）又是主办方，语义上都成立——只有「猜」出来的
    # issuer（无显式标签）才按重叠判误抽，显式标签信任文本作者。
    if issuer and location and not issuer_labeled and _overlaps(issuer, location):
        issuer = None

    # OTHER = 未能归类到任何已知通知场景。这类材料（随手提醒、失物招领、安全须知）
    # 里的"地点/发布方/课程"多半是噪声，与其猜错不如不抽——精度优先于召回。
    if category == Category.OTHER:
        location = issuer = course = None

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
