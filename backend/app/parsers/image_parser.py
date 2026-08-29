from __future__ import annotations

import logging

from ..providers.ocr import get_ocr
from .base import ParsedDoc

logger = logging.getLogger(__name__)


def parse(data: bytes, filename: str = "input.png", mime: str = "image/png") -> ParsedDoc:
    ocr = get_ocr()
    try:
        text, meta = ocr.image_to_text(data, mime=mime)
    except Exception as exc:  # noqa: BLE001
        logger.exception("OCR 失败: %s", filename)
        return ParsedDoc(
            kind="image",
            images=[(mime, data)],
            meta={"engine": ocr.name, "error": str(exc)},
            warnings=[f"OCR 失败：{exc}"],
        )

    doc = ParsedDoc(
        kind="image",
        text=(text or "").strip(),
        images=[(mime, data)],
        meta={"source": filename, **meta},
    )
    if not doc.text:
        doc.warnings.append(meta.get("warning") or "图片未识别出文字，可手动补充文本后重试")
    return doc
