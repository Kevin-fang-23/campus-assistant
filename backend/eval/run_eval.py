"""抽取质量评测。

用法：
    python -m eval.run_eval                    # 跑全部样本
    python -m eval.run_eval --tag 改期         # 只跑带某个 tag 的样本
    python -m eval.run_eval --verbose          # 打印逐条明细

设计要点：
1. 每条样本必须自带 base_time，相对时间（明天/本周五）才有唯一正确答案。
2. 直接调 run_pipeline，绕过数据库与文件落盘，只评估「抽取」这一环。
3. 分类用精确匹配；时间用容差匹配（默认 ±60 秒）；
   文本字段用归一化后的包含匹配（避免"计算机学院" vs "学院计算机"这类误判为错）。
4. 期望值只标能从原文确切推出的字段，标 null 表示"原文里没有，抽出来就算错"。
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

from app.graph.pipeline import run_pipeline  # noqa: E402

CASES_FILE = _HERE / "cases.jsonl"
TIME_FIELDS = ("event_time", "deadline")
TEXT_FIELDS = ("course", "location", "issuer")
SCORED_FIELDS = ("category", "course", "location", "issuer", "event_time", "deadline")

_PUNCT = re.compile(r"[\s，。、；：！？,.;:!?\-—_（）()【】\[\]《》\"'`~]")


def norm_text(s: str | None) -> str:
    """归一化：去空白与标点、全角转半角、小写。"""
    if not s:
        return ""
    table = str.maketrans("０１２３４５６７８９ＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺ：",
                          "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ:")
    return _PUNCT.sub("", str(s).translate(table)).lower()


def parse_dt(v: Any) -> datetime | None:
    if v is None or v == "":
        return None
    if isinstance(v, datetime):
        return v
    return datetime.fromisoformat(str(v).replace("Z", "").replace("/", "-"))


def match_field(field: str, exp: Any, pred: Any, tol_sec: int) -> bool:
    if field in TIME_FIELDS:
        e, p = parse_dt(exp), parse_dt(pred)
        if e is None and p is None:
            return True
        if e is None or p is None:
            return False
        return abs((p - e).total_seconds()) <= tol_sec
    e, p = norm_text(exp), norm_text(pred)
    if not e and not p:
        return True
    if not e or not p:
        return False
    if field == "category":
        return e == p
    return e in p or p in e  # 包含匹配


def score(case: dict, pred: dict, tol_sec: int) -> dict[str, str]:
    """对单条样本逐字段判定：tp / fp / fn / tn / ok。"""
    out: dict[str, str] = {}
    for f in SCORED_FIELDS:
        exp, got = case.get("expect", {}).get(f), pred.get(f)
        out[f] = "ok" if match_field(f, exp, got, tol_sec) else "bad"
    return out


def confusion(case: dict, pred: dict, tol_sec: int) -> dict[str, tuple[int, int, int]]:
    """返回每字段的 (tp, fp, fn)。"""
    res: dict[str, tuple[int, int, int]] = {}
    for f in SCORED_FIELDS:
        exp, got = case.get("expect", {}).get(f), pred.get(f)
        has_exp = exp not in (None, "")
        has_pred = got not in (None, "")
        ok = match_field(f, exp, got, tol_sec)
        if has_exp and has_pred:
            res[f] = (1, 0, 0) if ok else (0, 1, 1)
        elif has_exp and not has_pred:
            res[f] = (0, 0, 1)
        elif not has_exp and has_pred:
            res[f] = (0, 1, 0)
        else:
            res[f] = (0, 0, 0)
    return res


def prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def run_case(case: dict) -> dict[str, Any]:
    base = parse_dt(case.get("base_time")) or datetime.now()
    path = case.get("file")
    if path:
        from app.parsers import parse_bytes

        data = (_HERE / path).read_bytes()
        parsed = parse_bytes(data, Path(path).name)
        text, kind, images = parsed.text, parsed.kind, parsed.images
    else:
        text, kind, images = case["content"], "text", []
    state = run_pipeline(document_id=0, raw_text=text, kind=kind, images=images, base_time=base)
    return state.get("notice") or {}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default=None, help="只跑带该 tag 的样本")
    ap.add_argument("--tol", type=int, default=60, help="时间容差（秒）")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    cases = [
        json.loads(line)
        for line in CASES_FILE.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if args.tag:
        cases = [c for c in cases if args.tag in (c.get("tags") or [])]
    if not cases:
        print("没有匹配的样本。")
        return 1

    total: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
    cat_ok = 0
    failures: list[tuple[str, str, Any, Any]] = []

    for c in cases:
        pred = run_case(c)
        for f, (tp, fp, fn) in confusion(c, pred, args.tol).items():
            for i, v in enumerate((tp, fp, fn)):
                total[f][i] += v
        if match_field("category", c["expect"].get("category"), pred.get("category"), args.tol):
            cat_ok += 1
        for f in SCORED_FIELDS:
            if not match_field(f, c["expect"].get(f), pred.get(f), args.tol):
                failures.append((c["id"], f, c["expect"].get(f), pred.get(f)))

    n = len(cases)
    print(f"\n样本数：{n}   时间容差：±{args.tol}s")

    cat_cnt = Counter(c["expect"].get("category") or "-" for c in cases)
    dif_cnt = Counter(c.get("difficulty") or "-" for c in cases)
    tag_cnt: Counter[str] = Counter()
    for c in cases:
        tag_cnt.update(c.get("tags") or [])
    print("类别分布：" + "  ".join(f"{k}={v}" for k, v in sorted(cat_cnt.items())))
    print("难度分布：" + "  ".join(f"{k}={v}" for k, v in sorted(dif_cnt.items())))
    print("标签覆盖：" + "  ".join(f"{k}={v}" for k, v in sorted(tag_cnt.items())))

    print(f"\n分类准确率：{cat_ok}/{n} = {cat_ok / n:.1%}\n")
    print(f"{'字段':<12}{'P':>8}{'R':>8}{'F1':>8}{'TP':>6}{'FP':>6}{'FN':>6}")
    print("-" * 54)
    micro = [0, 0, 0]
    for f in SCORED_FIELDS:
        tp, fp, fn = total[f]
        for i, v in enumerate((tp, fp, fn)):
            micro[i] += v
        p, r, f1 = prf(tp, fp, fn)
        print(f"{f:<12}{p:>8.2f}{r:>8.2f}{f1:>8.2f}{tp:>6}{fp:>6}{fn:>6}")
    p, r, f1 = prf(*micro)
    print("-" * 54)
    print(f"{'微平均':<12}{p:>8.2f}{r:>8.2f}{f1:>8.2f}{micro[0]:>6}{micro[1]:>6}{micro[2]:>6}")

    if failures:
        print(f"\n未命中明细（{len(failures)} 项）：")
        for cid, f, exp, got in failures[:80]:
            print(f"  [{cid}] {f}: 期望={exp!r}  实际={got!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
