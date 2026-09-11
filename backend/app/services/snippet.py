"""句级片段选择：从通知原文中挑出与问题最相关的句子。

## 为什么需要

原先两处都用「按字符硬截断」：

```python
snippet = text[:120]        # 前端引用溯源展示
sources.append(text[:400])  # 喂给 LLM 的上下文
```

硬截断有两个实际问题：

1. **可能从句子中间切断** —— 引用读起来是残句（"…提交至学习通。截止时"），
   招聘官/用户看到的是被劈开的文本，可信度打折。
2. **更要命的是上下文截断** —— 通知的海报体常见「标题 + 详细说明 + 落款」结构，
   关键信息（截止时间、地点、联系方式）常落在 400 字之后。
   一旦被截掉，LLM 看不到就只会回答「未找到」，而库里其实有答案。

改成句级选择后：引用片段是**完整且与问题相关的句子**，
LLM 上下文在超预算时**优先保留相关句子**而不是盲目前 400 字。

## 设计约束

- 纯函数、无 I/O、无框架依赖 —— 便于单测与复用；
- **确定性**：同一 (text, query) 必得同一结果（同分按原序取先出现者），
  否则测试里的「引用前后一致」断言会随机失败；
- 永不返回空串：文本非空时至少回退到截断结果，保证前端不会拿到空白引用。
"""
from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 分句
# ---------------------------------------------------------------------------
# 句末标点。中文通知里 ； 常用来分隔并列要点，按句处理更利于精准选片。
_END_CHARS = "。！？!?；;"
# 紧随句末标点的收尾符号（引号/括号等）应留在句内，不该被切到下一句开头
_TAIL_CHARS = "”’\"'）」』】》)]）"
# 句首常见的无意义前缀（行首装饰符），清理后可提升分句质量
_LEAD_JUNK = "·•-—*#>\t "


def _split_with_spans(text: str) -> list[tuple[str, int, int]]:
    """切句并返回每句在原文中的字符区间 (sentence, start, end)。

    返回区间而非字符串，是为了最后能直接切片 `text[start:end]` 取回片段 ——
    这样**原文的换行与缩进被完整保留**（海报体常用换行分条，丢掉换行会读成一坨）。
    """
    if not text:
        return []

    out: list[tuple[str, int, int]] = []
    start = 0
    i = 0
    n = len(text)

    def flush(end: int) -> None:
        """把 text[start:end] 收尾成一句（去掉两端空白与行首装饰符）。"""
        nonlocal start
        s, e = start, end
        while s < e and text[s] in " \t\r\n":
            s += 1
        while s < e and text[s] in _LEAD_JUNK:
            s += 1
        while e > s and text[e - 1] in " \t\r\n":
            e -= 1
        if e > s:
            out.append((text[s:e], s, e))

    while i < n:
        ch = text[i]

        if ch == "\n":
            flush(i)
            start = i + 1
            i += 1
            continue

        is_end = ch in _END_CHARS
        if ch == "…":
            # 单个 … 是句内停顿，连续两个及以上才当句末
            is_end = i + 1 < n and text[i + 1] == "…"
        elif ch == ".":
            # 英文句点：仅当"后面是空白或文末"且"前面不是数字"时才算句末。
            # 两条都要有：前者保护 campus.edu.cn / Mr.Smith 这类域名与缩写，
            # 后者保护 3.14 / 9.18 这类小数。
            prev = text[i - 1] if i > 0 else ""
            nxt = text[i + 1] if i + 1 < n else ""
            is_end = (nxt == "" or nxt.isspace()) and not prev.isdigit()

        if is_end:
            # 吞掉连续的句末标点（！？ / 。。。 / ……）
            while i + 1 < n and (text[i + 1] in _END_CHARS or text[i + 1] == "…"):
                i += 1
            # 吞掉收尾符号（引号、括号），它们属于上一句
            while i + 1 < n and text[i + 1] in _TAIL_CHARS:
                i += 1
            flush(i + 1)
            start = i + 1

        i += 1

    flush(n)
    return out


def split_sentences(text: str) -> list[str]:
    """把通知正文切成句子列表（保留句末标点）。见 _split_with_spans 的切分规则。"""
    return [s for s, _, _ in _split_with_spans(text)]


# ---------------------------------------------------------------------------
# 打分
# ---------------------------------------------------------------------------
# 只保留有信息量的字符：中日韩文字、英文字母、数字
_KEEP_CHAR = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbfa-zA-Z0-9]")

# 问题意图 → 正文中的「答案特征」。
# 每项为 (问题侧正则, 具体值正则, 泛指词正则, 名称)：
#   命中"具体值"给全额奖励，"泛指词"只给部分奖励。
#
# 为什么要区分具体与泛指：问"开放到几点"时，引言句里的"延长开放时间"只含
# 泛指词，正文句里的"开放至 23:00"才含具体值。若两者同分，引言句会靠与问题
# 更高的词面重合胜出，结果是**引用片段支撑不了答案**（用户看到"延长开放时间"，
# 却看不到到底几点）。实测踩到过这个坑。
_INTENTS: tuple[tuple[re.Pattern[str], re.Pattern[str], re.Pattern[str], str], ...] = (
    (
        re.compile(r"什么时候|何时|截止|时间|几点|几号|日期|多久|期限|deadline"),
        re.compile(
            r"\d{1,4}\s*[年月日号]|\d{1,2}\s*[:：]\s*\d{2}|\d{1,2}\s*点|"
            r"周[一二三四五六日末]"
        ),
        re.compile(r"[上下]午|晚上|中午|本周|下周|今天|明天|后天|即日起|前提交|截止"),
        "time",
    ),
    (
        re.compile(r"在哪|在哪里|地点|哪里|哪个教室|哪个考场|考场|教室|场馆"),
        re.compile(
            r"地点|教学楼|实验[室楼]|图书馆|礼堂|体育馆|训练中心|"
            r"[A-Za-z]?\d{3}\s*室|[一二三四五六七八九]教\s*[A-Za-z0-9]"
        ),
        re.compile(r"教室|场馆|楼|室|馆|场"),
        "location",
    ),
    (
        re.compile(r"怎么|如何|怎样|方式|流程|步骤|怎么办|怎么申请"),
        # 只收"过程动作"词。刻意**不收 `报名`** —— 它是问题的宾语（名词），
        # 不是过程特征；否则"欢迎各年级同学报名参加"这类号召句会被误判为方式答案。
        # 宾语部分由词面覆盖率负责匹配，不需要在这里重复。
        re.compile(r"扫描|登录|填写|提交至|预约|请携带|二维码|小程序|链接|关注|发送"),
        re.compile(r"方式|流程|步骤|须|需|通过"),
        "method",
    ),
    (
        re.compile(r"联系|电话|邮箱|咨询|找谁|谁能问"),
        re.compile(r"\d{11}|\d{3,4}-\d{7,8}|@[\w.]+|微信"),
        re.compile(r"联系|电话|邮箱|手机|咨询"),
        "contact",
    ),
)

# 意图命中奖励。量级必须**显著高于词面重叠**：实测中复述标题的句子
# （如「《线性代数》期末复习提纲：第1-5章课后习题，提交至学习通。」）
# 靠堆主题词就能拿到 20+ 分，而真正的答案句（「截止时间：9月18日 22:00。」）
# 主题词很少。若不把意图奖励做强，"答案句"永远顶不过"主题句"。
_CUE_BONUS = 26.0
# 泛指词（无具体值）的折扣：是候选，但不如含具体值的句子
_CUE_VAGUE_FACTOR = 0.5
# 词面重叠权重。**用覆盖率而非绝对计数**：绝对计数会让长句靠堆词获胜，
# 覆盖率对句子长度不敏感，衡量的是"问题里的信息被这句话覆盖了多少"。
_W_BIGRAM = 2.0
_W_UNIGRAM = 0.6
_TOPIC_SCALE = 10.0
# 与通知标题高度重合的句子降权：标题在 API 里已由 title 字段单独展示，
# 再占用引用片段预算就是浪费（用户想看的是「证据」而不是「标题」）
_TITLE_SIM_THRESHOLD = 0.85
_TITLE_PENALTY = 0.15
# 长度先验：过短的碎片（如「教务处」「特此通知」）降权
_MIN_GOOD_CHARS = 10
_SHORT_PENALTY = 0.4
# 相关度下限：最高分低于此值就认为"没有真正相关的句子"，退回取开头。
#
# 为什么需要：中文里单字重合极易发生（问「量子纠缠」会与正文的「电子」共享一个「子」），
# 若不加下限，任何问题都会"选出一句"，实际是噪声，还不如老实取开头。
# 量级参考：单词重合约 1 分；真正的答案句因命中意图奖励通常在 25 分以上。
_MIN_RELEVANCE = 3.0


def _tokens(text: str) -> tuple[set[str], set[str]]:
    """返回 (unigram 集合, bigram 集合)，仅保留有信息量的字符。"""
    chars = _KEEP_CHAR.findall(text or "")
    lowered = [c.lower() for c in chars]
    unigrams = set(lowered)
    bigrams = {lowered[i] + lowered[i + 1] for i in range(len(lowered) - 1)}
    return unigrams, bigrams


def _similarity(a: str, b: str) -> float:
    """基于 bigram 的 Jaccard 相似度，用于判断某句是否等同于标题。"""
    if not a or not b:
        return 0.0
    _, ba = _tokens(a)
    _, bb = _tokens(b)
    if not ba or not bb:
        return 0.0
    inter = len(ba & bb)
    union = len(ba | bb)
    return inter / union if union else 0.0


def _cue_types(query: str) -> set[str]:
    return {name for q_re, _, _, name in _INTENTS if q_re.search(query or "")}


# 答案类型词前视窗口长度：看它前面几个字，判断这段是不是在回答"问题的那个主题"。
# 取 3 字是刻意的"紧邻"语义，窗口放宽会误伤：实测「办公地点：大学生活动中心
# 305 室」中的「305 室」匹配地点模式，窗口放到 6 字就把"大学生活动中心"里的
# 「活动」也算作"活动+地点"相邻，于是社团办公室压过了活动场地。
_CUE_WINDOW = 3
# 「问题里的词 + 答案类型词」紧邻出现的奖励。
# 用于区分同为标签的两句：「活动地点：图书馆南门…」与「办公地点：…」。
_CUE_ADJACENT_BONUS = 8.0


def _cue_bonus(sentence: str, cues: set[str], query: str) -> float:
    """计算句子的意图奖励。

    - 命中"具体值"（如 23:00、教一 302、138-0000-1234）给全额奖励；
    - 只命中"泛指词"（如"延长开放时间""截止"）给折扣奖励；
    - 若类型词前面紧邻问题里的词（问题问「活动…地点」对应正文「活动地点：」），
      再加相邻奖励 —— 这是区分"同为主题标签但主题不同"的关键信号。
    """
    if not cues:
        return 0.0
    _, q_bi = _tokens(query)
    bonus = 0.0
    for q_re, strong_re, vague_re, name in _INTENTS:
        if name not in cues:
            continue
        for m in strong_re.finditer(sentence):
            candidate = _CUE_BONUS
            window = sentence[max(0, m.start() - _CUE_WINDOW) : m.start()]
            if window and q_bi and any(bi in window for bi in q_bi):
                candidate += _CUE_ADJACENT_BONUS
            bonus = max(bonus, candidate)
        # 泛指词也可能紧邻问题词（如问题问"报名截止"、正文写"报名截止时间"）
        for m in vague_re.finditer(sentence):
            candidate = _CUE_BONUS * _CUE_VAGUE_FACTOR
            window = sentence[max(0, m.start() - _CUE_WINDOW) : m.start()]
            if window and q_bi and any(bi in window for bi in q_bi):
                candidate += _CUE_ADJACENT_BONUS
            bonus = max(bonus, candidate)
    return bonus


def score_sentence(sentence: str, query: str, *, title: str | None = None) -> float:
    """给单个句子打分。分数越高越可能是「问题的答案所在句」。

    构成（详见各权重常量处的说明）：
        1. 词面覆盖率 —— 问题里的信息被这句话覆盖多少（长度不敏感）；
        2. 意图奖励   —— 问题问时间/地点/方式/联系方式时，含对应特征才加分；
                        若问题里的词紧邻该特征词（如「活动」+「地点」），再加分；
        3. 标题降权   —— 与标题高度重合的句子降权（标题另有字段展示）；
        4. 长度先验   —— 过短碎片降权。
    """
    q_uni, q_bi = _tokens(query)
    if not q_uni:
        return 0.0

    s_uni, s_bi = _tokens(sentence)
    if not s_uni:
        return 0.0

    cov_bi = len(s_bi & q_bi) / len(q_bi) if q_bi else 0.0
    cov_uni = len(s_uni & q_uni) / len(q_uni)
    score = (_W_BIGRAM * cov_bi + _W_UNIGRAM * cov_uni) * _TOPIC_SCALE

    score += _cue_bonus(sentence, _cue_types(query), query)

    if title and _similarity(sentence, title) >= _TITLE_SIM_THRESHOLD:
        score *= _TITLE_PENALTY

    if len(_KEEP_CHAR.findall(sentence)) < _MIN_GOOD_CHARS:
        score *= _SHORT_PENALTY

    return score


def select_snippet(
    text: str,
    query: str,
    *,
    title: str | None = None,
    max_chars: int = 120,
    max_sentences: int = 2,
) -> str:
    """从 text 中挑出与 query 最相关的片段。

    输入：
        text          通知原文（或摘要）
        query         用户问题
        title         通知标题，用于给「与标题重合的句子」降权（可选）
        max_chars     片段字符上限
        max_sentences 最多拼接几个句子（相邻句会作为上下文一起带回）

    输出：
        一段连续文本（保留原文换行）。若最优句超长，按 max_chars 截断并加省略号；
        若选不出任何相关句，回退为原文本的前 max_chars 字符（保证非空）。

    选择规则（按优先级）：
        1. 逐句打分（词面覆盖率 + 意图奖励 + 标题降权 + 长度先验）；
        2. 最高分低于 _MIN_RELEVANCE 视为"无相关句"，退回取开头；
        3. 取最高分句为主体（同分取先出现者，保证确定性）；
        4. 向**左右相邻**句扩展（必须是连续句块，避免把中间未选中的句子一并带出），
           条件是相邻句分数 ≥ 最高分 × 0.6，且总长不超 max_chars、句数不超 max_sentences；
        5. 输出取所选句的**原文区间**，因此换行/缩进与原文一致。
    """
    text = (text or "").strip()
    if not text:
        return ""
    if len(text) <= max_chars:
        # 短通知整篇就是片段：不做裁剪，避免把本来完整的上下文切碎
        return text

    spans = _split_with_spans(text)
    if not spans:
        return text[:max_chars].strip()

    sentences = [s for s, _, _ in spans]
    scores = [score_sentence(s, query, title=title) for s in sentences]
    best = max(scores)

    if best < _MIN_RELEVANCE:
        # 问的问题与文本没有实质交集：无从判断相关性，保持原行为（取开头）
        return text[:max_chars].strip()

    best_idx = scores.index(best)
    threshold = best * 0.6
    lo = hi = best_idx

    # 左右交替扩展，优先扩展分数更高的一侧；放不下就试另一侧，都不行则停
    while (hi - lo + 1) < max_sentences:
        options: list[tuple[float, int, int]] = []
        if lo - 1 >= 0 and scores[lo - 1] >= threshold:
            options.append((scores[lo - 1], lo - 1, hi))
        if hi + 1 < len(sentences) and scores[hi + 1] >= threshold:
            options.append((scores[hi + 1], lo, hi + 1))
        if not options:
            break
        options.sort(key=lambda t: -t[0])
        for _, c_lo, c_hi in options:
            if spans[c_hi][2] - spans[c_lo][1] <= max_chars:
                lo, hi = c_lo, c_hi
                break
        else:
            break  # 两侧都超出预算

    out = text[spans[lo][1] : spans[hi][2]].strip()
    if len(out) > max_chars:
        out = out[: max_chars - 1].rstrip() + "…"
    return out


def select_context(
    text: str,
    query: str,
    *,
    title: str | None = None,
    max_chars: int = 400,
    max_sentences: int = 8,
) -> str:
    """为 LLM 挑选上下文：超预算时优先保留相关句子，而非盲目前 N 字。

    与 select_snippet 的区别在**目标**：引用片段追求「短而准」（给用户看），
    上下文追求「不漏关键信息」（给模型看）。因此句数上限更宽。

    选法：在**连续句块**中滑动窗口，取总分最高且不超预算的一段。
    必须是连续句块 —— 直接对选中句取区间会把中间未选中的句子也带出来，
    结果可能远超预算（实测踩到过 696 字 > 400 预算）。

    安全保底：最高分低于 _MIN_RELEVANCE、或整篇只有一句时，退回前 max_chars 字。
    宁可多给模型一些无关内容，也不要因裁剪丢掉唯一线索。
    """
    text = (text or "").strip()
    if not text or len(text) <= max_chars:
        return text

    spans = _split_with_spans(text)
    if len(spans) <= 1:
        return text[:max_chars].strip()

    sentences = [s for s, _, _ in spans]
    scores = [score_sentence(s, query, title=title) for s in sentences]
    if max(scores) < _MIN_RELEVANCE:
        return text[:max_chars].strip()

    n = len(sentences)
    best_lo = best_hi = None
    best_total = -1.0
    for start in range(n):
        total = 0.0
        for end in range(start, min(start + max_sentences, n)):
            if spans[end][2] - spans[start][1] > max_chars:
                break
            total += scores[end]
            if total > best_total:
                best_total, best_lo, best_hi = total, start, end

    if best_lo is None:
        return text[:max_chars].strip()

    return text[spans[best_lo][1] : spans[best_hi][2]].strip()
