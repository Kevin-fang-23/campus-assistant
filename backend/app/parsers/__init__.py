"""统一解析入口：按后缀 / MIME 路由到具体解析器。"""
from __future__ import annotations

from pathlib import Path

from ..models import DocKind
from . import image_parser, pdf_parser, text_parser
from .base import ParsedDoc

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff", ".gif"}
TEXT_EXTS = {".txt", ".md", ".csv", ".log", ".json"}

__all__ = ["ParsedDoc", "detect_kind", "parse_bytes"]


def detect_kind(filename: str, mime: str | None = None) -> str:
    ext = Path(filename or "").suffix.lower()
    mime = (mime or "").lower()
    if ext == ".pdf" or "pdf" in mime:
        return DocKind.PDF
    if ext in IMAGE_EXTS or mime.startswith("image/"):
        return DocKind.IMAGE
    if ext in TEXT_EXTS or mime.startswith("text/"):
        return DocKind.TEXT
    return DocKind.TEXT


def parse_bytes(data: bytes, filename: str, mime: str | None = None) -> ParsedDoc:
    kind = detect_kind(filename, mime)
    if kind == DocKind.PDF:
        return pdf_parser.parse(data, filename)
    if kind == DocKind.IMAGE:
        return image_parser.parse(data, filename, mime=mime or "image/png")
    return text_parser.parse(data, filename)
