"""OCR Provider：paddleocr -> rapidocr -> VLM -> stub 自动探测，任一可用即启用。"""
from __future__ import annotations

import io
import logging
from abc import ABC, abstractmethod
from functools import lru_cache
from typing import Any

from ..config import settings

logger = logging.getLogger(__name__)


def _to_ndarray(image_bytes: bytes):
    import numpy as np
    from PIL import Image

    img = Image.open(io.BytesIO(image_bytes))
    if img.mode != "RGB":
        img = img.convert("RGB")
    return np.array(img)


class BaseOCR(ABC):
    name: str = "base"
    is_stub: bool = False

    @abstractmethod
    def image_to_text(self, image_bytes: bytes, mime: str = "image/png") -> tuple[str, dict[str, Any]]:
        ...


class PaddleOCREngine(BaseOCR):
    name = "paddleocr"

    def __init__(self) -> None:
        from paddleocr import PaddleOCR

        self._engine = PaddleOCR(use_angle_cls=True, lang=settings.ocr_lang, show_log=False)

    def image_to_text(self, image_bytes: bytes, mime: str = "image/png") -> tuple[str, dict]:
        arr = _to_ndarray(image_bytes)
        lines: list[str] = []
        scores: list[float] = []
        raw = None
        if hasattr(self._engine, "predict"):      # PaddleOCR 3.x
            try:
                raw = self._engine.predict(arr)
                for page in raw or []:
                    texts = page.get("rec_texts") if isinstance(page, dict) else None
                    confs = page.get("rec_scores") if isinstance(page, dict) else None
                    if texts:
                        lines.extend(texts)
                        scores.extend(confs or [])
            except Exception as exc:  # noqa: BLE001
                logger.debug("paddle predict 不可用，改用 ocr(): %s", exc)
        if not lines:                              # PaddleOCR 2.x
            raw = self._engine.ocr(arr, cls=True)
            for page in raw or []:
                for item in page or []:
                    if isinstance(item, (list, tuple)) and len(item) >= 2:
                        text, score = item[1][0], float(item[1][1])
                        lines.append(text)
                        scores.append(score)
        avg = sum(scores) / len(scores) if scores else 0.0
        return "\n".join(lines), {"engine": self.name, "lines": len(lines), "avg_confidence": round(avg, 4)}


class RapidOCREngine(BaseOCR):
    name = "rapidocr"

    def __init__(self) -> None:
        from rapidocr_onnxruntime import RapidOCR

        self._engine = RapidOCR()

    def image_to_text(self, image_bytes: bytes, mime: str = "image/png") -> tuple[str, dict]:
        arr = _to_ndarray(image_bytes)
        result, _ = self._engine(arr)
        lines = [row[1] for row in (result or [])]
        scores = [float(row[2]) for row in (result or []) if len(row) > 2]
        avg = sum(scores) / len(scores) if scores else 0.0
        return "\n".join(lines), {"engine": self.name, "lines": len(lines), "avg_confidence": round(avg, 4)}


class VLMOCREngine(BaseOCR):
    """用视觉大模型直接读图，适合海报这类版面复杂的材料。"""

    name = "vlm"

    def image_to_text(self, image_bytes: bytes, mime: str = "image/png") -> tuple[str, dict]:
        from .vlm import get_vlm

        vlm = get_vlm()
        text = vlm.ocr_image((mime, image_bytes))
        return text, {"engine": f"{self.name}:{vlm.name}", "lines": text.count("\n") + 1}


class StubOCR(BaseOCR):
    """无任何 OCR 引擎时的占位实现：不抛错，但明确告知需要补充文本。"""

    name = "stub"
    is_stub = True

    def image_to_text(self, image_bytes: bytes, mime: str = "image/png") -> tuple[str, dict]:
        return "", {
            "engine": self.name,
            "warning": "未安装 OCR 引擎且未配置视觉模型，图片文字未识别。"
                       "可 pip install rapidocr-onnxruntime 或设置 VLM_PROVIDER=dashscope。",
        }


@lru_cache
def get_ocr() -> BaseOCR:
    want = settings.ocr_provider.lower()
    candidates: list[type[BaseOCR]]
    if want == "paddle":
        candidates = [PaddleOCREngine]
    elif want == "rapid":
        candidates = [RapidOCREngine]
    elif want == "vlm":
        candidates = [VLMOCREngine]
    elif want == "stub":
        candidates = [StubOCR]
    else:
        candidates = [PaddleOCREngine, RapidOCREngine]

    for cls in candidates:
        try:
            engine = cls()
            logger.info("OCR 引擎启用: %s", engine.name)
            return engine
        except Exception as exc:  # noqa: BLE001
            logger.info("OCR 引擎 %s 不可用: %s", cls.name, exc)

    from .vlm import get_vlm

    if want in ("auto", "vlm") and not get_vlm().is_mock:
        logger.info("OCR 引擎启用: vlm")
        return VLMOCREngine()
    logger.warning("无可用 OCR 引擎，使用 stub")
    return StubOCR()
