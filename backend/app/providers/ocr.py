"""OCR Provider：paddleocr -> rapidocr -> VLM -> stub 自动探测，任一可用即启用。"""
from __future__ import annotations

import io
import logging
import os
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


def effective_ocr_threads() -> int:
    """把 settings.ocr_cpu_threads 解析成实际 intra-op 线程数。

    返回值语义：**0 = 不干预（用引擎默认）**，>0 = 限制到该线程数。
      · 配置 <0  → 0（显式不限制，吃满全核）
      · 配置 >0  → 原值
      · 配置 =0  → 自动 max(2, cpu_count // 2)，给系统留一半核
    """
    cfg = int(settings.ocr_cpu_threads)
    if cfg < 0:
        return 0
    if cfg > 0:
        return cfg
    cores = os.cpu_count() or 4
    return max(2, cores // 2)


def limit_rapidocr_threads(n: int) -> None:
    """把 onnxruntime 的 intra-op 线程上限注入 rapidocr（幂等，n<=0 时不干预）。

    为什么需要：rapidocr_onnxruntime 的 OrtInferSession 构造 SessionOptions()
    后**从不设置 intra_op_num_threads**，onnxruntime 便按物理核数起线程，且
    det/cls/rec 三个会话各自都想吃满全核。屏幕共享/录制时采集+编码本就抢 CPU，
    OCR 一起来会把所有核占满，饿死代理转发与浏览器渲染（表现为解析极慢甚至
    加载不出来）。该库无公开参数可传线程数，故在其模块命名空间里把
    SessionOptions 换成一个会预设 intra_op_num_threads 的工厂——只影响本进程内
    rapidocr 之后创建的会话，可通过配置 <0 关闭（回到引擎默认）。
    """
    if n <= 0:
        return
    import rapidocr_onnxruntime.utils as rocr_utils

    # 幂等：已打过补丁就不再重复包裹（避免 _orig 指向上一轮的工厂）。
    if getattr(rocr_utils.SessionOptions, "_campus_thread_capped", False):
        return
    original = rocr_utils.SessionOptions

    def _capped_session_options():
        so = original()
        so.intra_op_num_threads = n
        return so

    _capped_session_options._campus_thread_capped = True  # type: ignore[attr-defined]
    rocr_utils.SessionOptions = _capped_session_options
    logger.info(
        "RapidOCR onnxruntime intra-op 线程上限设为 %d（物理核 %s）；"
        "屏幕共享等高负载场景可显著降低 CPU 争用",
        n,
        os.cpu_count(),
    )


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

        # 必须在构造 RapidOCR 之前打补丁：会话在其 __init__ 内即创建。
        limit_rapidocr_threads(effective_ocr_threads())
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
