from __future__ import annotations

from .base import ParsedDoc


def parse(data: bytes, filename: str = "input.txt") -> ParsedDoc:
    text, encoding = "", "utf-8"
    for enc in ("utf-8", "utf-8-sig", "gb18030", "big5", "latin-1"):
        try:
            text = data.decode(enc)
            encoding = enc
            break
        except UnicodeDecodeError:
            continue
    doc = ParsedDoc(kind="text", text=text.strip(), meta={"encoding": encoding, "source": filename})
    if not doc.text:
        doc.warnings.append("文本内容为空")
    return doc
