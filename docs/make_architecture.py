"""生成系统架构图 SVG（矢量），供 docs/architecture.svg 与 README 用 PNG 使用。

设计取向：
- 分层横向带状（前端 → 接入 → 编排/检索 → 能力 → 存储），层间竖直箭头表示数据流向；
- 每层左侧色条 + 层名 + 该层职责一句话；
- 层内每个模块标注「模块名 + 关键职责」，包含实际配置值（模型名/维度/权重），
  避免出现"看起来像架构图但读完不知道用了什么"的空图。
"""
from __future__ import annotations

from pathlib import Path
from xml.sax.saxutils import escape

W = 1560
M = 40
INNER = W - 2 * M

FONT = "Microsoft YaHei, PingFang SC, Segoe UI, Helvetica, Arial, sans-serif"
MONO = "Cascadia Mono, Consolas, SFMono-Regular, Menlo, monospace"

INK = "#0f172a"       # 标题
BODY = "#475569"      # 正文
MUTED = "#64748b"     # 次要
PANEL = "#f8fafc"     # 层底板
PANEL_LINE = "#e2e8f0"
CARD = "#ffffff"

C_FE = "#4f46e5"      # 前端
C_API = "#0284c7"     # 接入
C_ORCH = "#d97706"    # 编排
C_RET = "#059669"     # 检索
C_PROV = "#7c3aed"    # 能力
C_STORE = "#475569"   # 存储
C_SAFE = "#dc2626"    # 护栏

out: list[str] = []
y = 0

# ---------------------------------------------------------------- 基础绘制

def rect(x, y_, w, h, fill, stroke=None, r=12, sw=1):
    s = f'<rect x="{x:.1f}" y="{y_:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{r}" fill="{fill}"'
    if stroke:
        s += f' stroke="{stroke}" stroke-width="{sw}"'
    out.append(s + "/>")


def text(x, y_, s, size=12, fill=BODY, weight="400", anchor="start", family=FONT, opacity=None):
    o = f' opacity="{opacity}"' if opacity else ""
    out.append(
        f'<text x="{x:.1f}" y="{y_:.1f}" font-family="{family}" font-size="{size}" '
        f'font-weight="{weight}" fill="{fill}" text-anchor="{anchor}"{o}>{escape(s)}</text>'
    )


def arrow(x1, y1, x2, y2, color="#94a3b8", sw=2, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    out.append(
        f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
        f'stroke="{color}" stroke-width="{sw}" marker-end="url(#ah)"{d}/>'
    )


def band(y_, h, accent, name, duty):
    """一层：底板 + 左侧色条 + 层名 + 职责。返回模块区起始 y。"""
    rect(M, y_, INNER, h, PANEL, PANEL_LINE, r=14)
    rect(M, y_ + 16, 5, h - 32, accent, r=2.5)
    text(M + 20, y_ + 30, name, 15, accent, "700")
    text(M + 20 + len(name) * 16 + 16, y_ + 30, duty, 12, MUTED)
    return y_ + 48


def card(x, y_, w, h, accent, title, lines, tsize=13):
    rect(x, y_, w, h, CARD, "#e2e8f0", r=10)
    rect(x, y_ + 10, 3.5, h - 20, accent, r=1.75)
    text(x + 14, y_ + 24, title, tsize, INK, "600")
    yy = y_ + 44
    for ln in lines:
        text(x + 14, yy, ln, 11, BODY, "400")
        yy += 16
    return y_ + h


# ---------------------------------------------------------------- 标题

out.append(
    f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="1190" '
    f'viewBox="0 0 {W} 1190">'
)
out.append(
    '<defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" '
    'markerHeight="7" orient="auto-start-reverse">'
    '<path d="M 0 0 L 10 5 L 0 10 z" fill="#94a3b8"/></marker></defs>'
)
rect(0, 0, W, 1190, "#ffffff", r=0)

y = 40
text(M, y, "校园事务智能助手 · 系统架构", 27, INK, "700")
text(M, y + 26, "多模态抽取 → 混合检索 → RAG 问答   |   FastAPI · LangGraph · FAISS · BM25 · React · SQLite", 13, MUTED)

# ---------------------------------------------------------------- 1 前端

y = 92
inner = band(y, 116, C_FE, "① 前端 SPA", "职责：五视图交互与呈现；句级片段、引用溯源、命中来源可视化")
c1y = inner + 4
cw = (INNER - 40 - 4 * 18) / 5
titles = [
    ("概览", "统计卡 + 类别分布"),
    ("导入", "文件 / 粘贴文本"),
    ("通知复核", "人工修正与筛选"),
    ("待办看板", "四态流转 · 逾期标红"),
    ("知识库问答", "语义检索 / 智能问答"),
]
for i, (t, d) in enumerate(titles):
    cx = M + 20 + i * (cw + 18)
    rect(cx, c1y, cw, 46, CARD, "#e2e8f0", r=9)
    text(cx + cw / 2, c1y + 21, t, 12.5, C_FE, "600", anchor="middle")
    text(cx + cw / 2, c1y + 37, d, 10.5, MUTED, anchor="middle")

# 箭头：前端 → 接入
arrow(W / 2, 208, W / 2, 240, sw=2.2)
text(W / 2 + 12, 228, "HTTP / JSON   /api/* · /health", 11.5, MUTED)

# ---------------------------------------------------------------- 2 接入层

y = 244
inner = band(
    y, 158, C_API, "② 接入层 · FastAPI",
    "职责：请求校验、路由分发、限流护栏、统一错误语义（429 带跨域头）",
)
c2y = inner + 4
eps = [
    ("/api/documents", "导入：图片 / PDF / 文本"),
    ("/api/notices", "通知列表与人工修正"),
    ("/api/tasks", "待办状态流转"),
    ("/api/search", "混合检索 + 片段"),
    ("/api/qa", "RAG 问答 + 引用"),
    ("/health", "运行时能力与降级态"),
]
ew = (INNER - 40 - 5 * 14) / 6
for i, (t, d) in enumerate(eps):
    cx = M + 20 + i * (ew + 14)
    rect(cx, c2y, ew, 44, CARD, "#e2e8f0", r=9)
    text(cx + 10, c2y + 19, t, 11.5, C_API, "600", family=MONO)
    text(cx + 10, c2y + 35, d, 10.5, BODY)
text(
    M + 20, c2y + 74,
    "中间件：RateLimitMiddleware —— L1 每 IP/分钟（令牌桶）+ L2 每 IP/日 + L3 全局/日（固定窗口，计数落库）"
    "   ·   CORSMiddleware 置于外层，保证 429 可被前端读取",
    11.5, MUTED,
)

# 箭头：接入 → 编排/检索（分叉）
arrow(W * 0.28, 406, W * 0.28, 438, sw=2.2)
arrow(W * 0.72, 406, W * 0.72, 438, sw=2.2)
text(W * 0.28 + 12, 426, "写入链路（导入）", 11.5, MUTED)
text(W * 0.72 + 12, 426, "读取链路（检索 / 问答）", 11.5, MUTED)

# ---------------------------------------------------------------- 3 编排层 / 检索层

y = 442
half = (INNER - 22) / 2
bh = 222

# 左：编排层
rect(M, y, half, bh, PANEL, PANEL_LINE, r=14)
rect(M, y + 16, 5, bh - 32, C_ORCH, r=2.5)
text(M + 20, y + 30, "③ 编排层 · LangGraph", 15, C_ORCH, "700")
text(M + 20, y + 52, "职责：把「原始文件」变成「结构化通知 + 待办」", 12, MUTED)
rect(M + 20, y + 64, half - 40, 34, CARD, "#e2e8f0", r=9)
text(
    M + 34, y + 85,
    "parse → extract → normalize → dedup → persist → tasks",
    11.5, C_ORCH, "600", family=MONO,
)
for i, ln in enumerate([
    "· 多模态统一入口：图片 / PDF / 文本各走对应 parser",
    "· 文件级去重：sha256 命中已解析文档 → 直接复用，不重复计费",
    "· VLM 抽取失败 → 自动降级为规则抽取（正则 + 时间解析）",
    "· 通知级去重：向量余弦 ≥ 0.90 判为重复，跳过建库",
    "· 无 langgraph 时退化为顺序执行，逻辑等价（离线可跑）",
]):
    text(M + 24, y + 120 + i * 17, ln, 11.5, BODY)

# 右：检索层
rx = M + half + 22
rect(rx, y, half, bh, PANEL, PANEL_LINE, r=14)
rect(rx, y + 16, 5, bh - 32, C_RET, r=2.5)
text(rx + 20, y + 30, "④ 检索层 · 混合检索", 15, C_RET, "700")
text(rx + 20, y + 52, "职责：精确串与语义表达都能召回，并给出可解释的命中", 12, MUTED)
rect(rx + 20, y + 64, half - 40, 34, CARD, "#e2e8f0", r=9)
text(
    rx + 34, y + 85,
    "向量召回 + BM25 召回 → 加权 RRF → 相关性过滤 → 句级选片",
    11.5, C_RET, "600", family=MONO,
)
for i, ln in enumerate([
    "· 两路互补：课程名/房间号/电话靠 BM25，近义表述靠向量",
    "· 加权 RRF（只用名次，免疫两路分数尺度差异），权重 0.1 : 0.9",
    "· 相关性阈值过滤 → filtered 标志，区分「不相关」与「库内确实没有」",
    "· 句级选片：按查询挑出最相关句子，避免从字中间硬切出残句",
    "· 索引 FAISS IndexFlatIP；不可用时退化为 numpy 余弦（结果等价）",
]):
    text(rx + 24, y + 120 + i * 17, ln, 11.5, BODY)

# 箭头 → 能力层
arrow(W * 0.28, y + bh + 2, W * 0.28, y + bh + 34, sw=2.2)
arrow(W * 0.72, y + bh + 2, W * 0.72, y + bh + 34, sw=2.2)
text(W * 0.28 + 12, y + bh + 24, "调用抽取能力", 11.5, MUTED)
text(W * 0.72 + 12, y + bh + 24, "查询向量 / 生成答案", 11.5, MUTED)

# ---------------------------------------------------------------- 4 能力层

y = 702
inner = band(
    y, 150, C_PROV, "⑤ 能力层 · Providers",
    "职责：外部能力统一封装；无 Key 也能跑通全链路（每项均可降级）",
)
c4y = inner + 4
provs = [
    ("VLM · 视觉抽取", ["qwen3-vl-plus @ dashscope", "海报 / 截图 → 结构化字段", "失败 → 规则抽取兜底"]),
    ("OCR · 文字识别", ["RapidOCR / PaddleOCR", "扫描件与图片文字层", "未安装 → stub（不阻断）"]),
    ("Embedding · 向量化", ["text-embedding-v4（1024 维）", "无 Key → local_hash 兜底", "降级态由 /health 暴露"]),
    ("LLM Client · 生成", ["OpenAI 兼容协议", "进程级连接复用（省握手）", "超时 / 重试 / 错误分级"]),
]
pw = (INNER - 40 - 3 * 16) / 4
for i, (t, lines) in enumerate(provs):
    card(M + 20 + i * (pw + 16), c4y, pw, 86, C_PROV, t, lines)

# 箭头 → 存储层
arrow(W / 2, 856, W / 2, 890, sw=2.2)
text(W / 2 + 12, 878, "SQLAlchemy 2.0 读写", 11.5, MUTED)

# ---------------------------------------------------------------- 5 存储层

y = 894
inner = band(
    y, 152, C_STORE, "⑥ 存储层 · SQLAlchemy 2.0",
    "默认 SQLite（演示零依赖），可切 PostgreSQL —— 仅需改 DATABASE_URL",
)
c5y = inner + 4
stores = [
    ("notices / documents", ["抽取结果与原始文档", "置信度 · 复核标记 · 去重关系"]),
    ("tasks / task_events", ["待办与状态流转事件", "截止/提醒时间 · 逾期判定"]),
    ("notice_embeddings", ["向量持久化（BLOB + 维度）", "启动加载进索引，维度不符跳过"]),
    ("rate_limit_counters", ["限流日计数（原子占位）", "多 worker 共享 · 重启不清零"]),
    ("storage/", ["文件名为 {sha前12位}_{原名}", "仅文件类落盘，粘贴文本不落盘"]),
]
sw_ = (INNER - 40 - 4 * 14) / 5
for i, (t, lines) in enumerate(stores):
    card(M + 20 + i * (sw_ + 14), c5y, sw_, 84, C_STORE, t, lines, tsize=12)

# ---------------------------------------------------------------- 6 降级与护栏

y = 1060
inner = band(
    y, 84, C_SAFE, "⑦ 横切 · 降级与护栏",
    "职责：任何外部依赖失效都不返回 500 —— 可用性优先，且降级态可被观测",
)
c6y = inner + 0
guards = [
    "VLM 失败 → 规则抽取",
    "LLM 失败 → 抽取式回答（答案不中断）",
    "Embedding 降级 → local_hash（/health 暴露）",
    "限流存储异常 → 放行 + 告警",
]
gw = (INNER - 40 - 3 * 16) / 4
for i, g in enumerate(guards):
    gx = M + 20 + i * (gw + 16)
    rect(gx, c6y, gw, 26, "#fef2f2", "#fecaca", r=7)
    text(gx + gw / 2, c6y + 17.5, g, 11, "#b91c1c", "500", anchor="middle")

out.append("</svg>")

svg = "\n".join(out)

# 输出路径相对脚本自身，因此从任意 cwd 运行都写对位置
HERE = Path(__file__).resolve().parent
(HERE / "architecture.svg").write_text(svg, encoding="utf-8")

# PNG 渲染用的包装页写到系统临时目录：不进仓库，避免污染
import tempfile

wrap = Path(tempfile.gettempdir()) / "_campus_arch_wrap.html"
wrap.write_text(
    "<!doctype html><html><head><meta charset='utf-8'>"
    "<style>html,body{margin:0;padding:0;background:#fff;}</style></head><body>"
    + svg
    + "</body></html>",
    encoding="utf-8",
)

print("svg bytes:", len(svg.encode("utf-8")))
print("height used:", y + 84)
print("written:", HERE / "architecture.svg")
print("wrap page (用于渲染 PNG):", wrap.as_uri())
