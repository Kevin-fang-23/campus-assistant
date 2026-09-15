"""PDF 解析：优先取文本层，扫描页转图走 OCR。"""
from __future__ import annotations

import logging

from ..config import settings
from ..providers.ocr import get_ocr
from .base import ParsedDoc

logger = logging.getLogger(__name__)
MAX_VLM_PAGE_IMAGES = 2


def parse(data: bytes, filename: str = "input.pdf") -> ParsedDoc:
    try:
        import fitz  # PyMuPDF
    except ImportError as exc:  # pragma: no cover
        return ParsedDoc(kind="pdf", meta={"error": str(exc)}, warnings=["未安装 PyMuPDF"])

    doc = ParsedDoc(kind="pdf", meta={"source": filename})
    page_texts: list[str] = []
    ocr_pages: list[int] = []
    ocr_engine = None

    try:
        with fitz.open(stream=data, filetype="pdf") as pdf:
            doc.pages = pdf.page_count
            for pno in range(pdf.page_count):
                page = pdf[pno]
                text = (page.get_text("text") or "").strip()
                if len(text) < settings.pdf_text_min_chars:  # 判定为扫描页
                    ocr_engine = ocr_engine or get_ocr()
                    zoom = settings.pdf_ocr_dpi / 72
                    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
                    png = pix.tobytes("png")
                    if len(doc.images) < MAX_VLM_PAGE_IMAGES:
                        doc.images.append(("image/png", png))
                    try:
                        ocr_text, meta = ocr_engine.image_to_text(png, mime="image/png")
                        text = (ocr_text or "").strip()
                        ocr_pages.append(pno + 1)
                        doc.meta.setdefault("ocr", []).append({"page": pno + 1, **meta})
                    except Exception as exc:  # noqa: BLE001
                        doc.warnings.append(f"第 {pno + 1} 页 OCR 失败：{exc}")
                page_texts.append(text)
    except Exception as exc:
        logger.exception("PDF 解析失败: %s", filename)
        return ParsedDoc(kind="pdf", meta={"error": str(exc)}, warnings=[f"PDF 解析失败：{exc}"])

    doc.text = "\n\n".join(t for t in page_texts if t).strip()
    doc.meta.update({"ocr_pages": ocr_pages, "page_count": doc.pages})
    if not doc.text:
        doc.warnings.append("PDF 未提取到文字（可能是纯图扫描件且 OCR 不可用）")
    return doc
