"""视觉大模型 Provider。

- mock      : 规则实现，无需 Key，全链路可跑通（默认）
- dashscope : OpenAI 兼容接口（阿里云百炼 / 自托管 vLLM 同协议）
              环境变量：VLM_PROVIDER=dashscope, DASHSCOPE_API_KEY=sk-xxx,
                        VLM_MODEL=qwen3-vl-plus（可换 qwen3.5-vl-plus / 本地模型名）,
                        VLM_BASE_URL=http://localhost:8000/v1（自托管时）
"""
from __future__ import annotations

import base64
import json
import logging
import re
from abc import ABC, abstractmethod
from datetime import datetime
from functools import lru_cache
from typing import Any

from ..config import settings
from ..models import Category
from ..services import rule_extract
from ..services.datetime_utils import pick_times
from .llm_client import build_client_from_settings

logger = logging.getLogger(__name__)

ImagePart = tuple[str, bytes]  # (mime_type, raw_bytes)

_FIELD_SPEC = """{
  "category": "course_notice | activity_poster | homework | repair | other",
  "title": "不超过 40 字的标题",
  "summary": "80 字内摘要",
  "issuer": "发布方，未知填 null",
  "location": "地点，未知填 null",
  "course": "课程名，未知填 null",
  "event_time": "活动/上课时间 ISO8601，未知填 null",
  "deadline": "截止时间 ISO8601，未知填 null",
  "contacts": ["联系方式字符串数组"],
  "tags": ["关键词数组"],
  "confidence": 0.0
}"""


class BaseVLM(ABC):
    name: str = "base"
    is_mock: bool = True

    @abstractmethod
    def classify(self, text: str, images: list[ImagePart] | None = None) -> tuple[str, float, dict]:
        ...

    @abstractmethod
    def extract(
        self,
        text: str,
        category: str,
        images: list[ImagePart] | None = None,
        base: datetime | None = None,
    ) -> dict[str, Any]:
        """base 为时间基准，用于把「明天/本周五」等相对时间解析成绝对时间。

        调用方应把 state["base_time"] 透传进来；留空则回退到 datetime.now()，
        会导致评测不可复现（同一份样本在不同日期跑出不同结果）。
        """

    def ocr_image(self, image: ImagePart) -> str:  # 供 OCR 兜底使用
        raise NotImplementedError


class MockVLM(BaseVLM):
    name = "mock"
    is_mock = True

    def classify(self, text: str, images: list[ImagePart] | None = None) -> tuple[str, float, dict]:
        cat, conf, scores = rule_extract.classify(text)
        return cat, conf, {"scores": scores, "provider": self.name}

    def extract(
        self,
        text: str,
        category: str,
        images: list[ImagePart] | None = None,
        base: datetime | None = None,
    ) -> dict[str, Any]:
        data = rule_extract.extract(text, category, base=base)
        data["extra"]["provider"] = self.name
        return data


class OpenAICompatVLM(BaseVLM):
    """走 /chat/completions 的视觉模型，兼容百炼与自托管 vLLM。"""

    name = "dashscope"
    is_mock = False

    def __init__(self) -> None:
        # 鉴权、超时、重试统一由 llm_client 负责
        self._client = build_client_from_settings()

    # ---- 内部 ----
    def _content(self, prompt: str, images: list[ImagePart] | None) -> list[dict]:
        parts: list[dict] = []
        for mime, raw in (images or [])[:4]:
            b64 = base64.b64encode(raw).decode()
            parts.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}})
        parts.append({"type": "text", "text": prompt})
        return parts

    def _chat(self, prompt: str, images: list[ImagePart] | None, max_tokens: int = 1200) -> str:
        messages = [
            {"role": "system", "content": "你是校园事务信息抽取助手，只输出 JSON，不要解释。"},
            {"role": "user", "content": self._content(prompt, images)},
        ]
        return self._client.chat(messages, model=settings.vlm_model, max_tokens=max_tokens)

    @staticmethod
    def _parse_json(raw: str) -> dict:
        text = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.M).strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            start, depth = text.find("{"), 0
            for i in range(start, len(text)) if start >= 0 else []:
                depth += 1 if text[i] == "{" else -1 if text[i] == "}" else 0
                if depth == 0:
                    return json.loads(text[start : i + 1])
        raise ValueError("模型未返回可解析 JSON")

    @staticmethod
    def _to_dt(
        value: Any,
        fallback_text: str,
        kind: str,
        category: str,
        base: datetime | None = None,
    ) -> datetime | None:
        if isinstance(value, str) and value.strip():
            token = value.strip().replace("Z", "").replace("/", "-")
            for fmt in (None, "%Y-%m-%d %H:%M", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
                try:
                    return datetime.fromisoformat(token) if fmt is None else datetime.strptime(token, fmt)
                except ValueError:
                    continue
            event, deadline, _ = pick_times(value, category=category, base=base)
            return deadline if kind == "deadline" else event
        return None

    # ---- 对外 ----
    def classify(self, text: str, images: list[ImagePart] | None = None) -> tuple[str, float, dict]:
        prompt = (
            "判断这份校园材料属于哪一类，只输出 JSON："
            '{"category":"course_notice|activity_poster|homework|repair|other","confidence":0.0}\n\n'
            f"材料文本：\n{text[:3000]}"
        )
        try:
            data = self._parse_json(self._chat(prompt, images, max_tokens=120))
            cat = data.get("category", Category.OTHER)
            cat = cat if cat in Category.ALL else Category.OTHER
            return cat, float(data.get("confidence") or 0.6), {"provider": self.name}
        except Exception as exc:  # noqa: BLE001
            logger.warning("VLM 分类失败，回退规则分类: %s", exc)
            cat, conf, scores = rule_extract.classify(text)
            return cat, conf * 0.9, {"provider": "rule-fallback", "error": str(exc), "scores": scores}

    def extract(
        self,
        text: str,
        category: str,
        images: list[ImagePart] | None = None,
        base: datetime | None = None,
    ) -> dict[str, Any]:
        today = (base or datetime.now()).strftime("%Y-%m-%d %A")
        prompt = (
            f"今天是 {today}。这是一份校园{Category.LABELS.get(category, '')}材料，"
            f"请抽取关键信息，相对时间（明天/本周五）要换算成绝对时间。"
            f"严格按以下 JSON 结构输出：\n{_FIELD_SPEC}\n\n材料文本：\n{text[:6000]}"
        )
        rule_data = rule_extract.extract(text, category, base=base)
        try:
            data = self._parse_json(self._chat(prompt, images))
        except Exception as exc:  # noqa: BLE001
            logger.warning("VLM 抽取失败，回退规则抽取: %s", exc)
            rule_data["extra"].update({"provider": "rule-fallback", "vlm_error": str(exc)})
            return rule_data

        merged = dict(rule_data)
        for key in ("title", "summary", "issuer", "location", "course"):
            val = data.get(key)
            if isinstance(val, str) and val.strip() and val.strip().lower() != "null":
                merged[key] = val.strip()
        merged["category"] = data.get("category") if data.get("category") in Category.ALL else category
        merged["event_time"] = (
            self._to_dt(data.get("event_time"), text, "event", category, base=base)
            or rule_data["event_time"]
        )
        merged["deadline"] = (
            self._to_dt(data.get("deadline"), text, "deadline", category, base=base)
            or rule_data["deadline"]
        )
        for key in ("contacts", "tags"):
            val = data.get(key)
            if isinstance(val, list) and val:
                merged[key] = [str(x)[:60] for x in val][:8]
        merged["confidence"] = max(float(data.get("confidence") or 0.0), rule_data["confidence"])
        merged["extra"] = {**rule_data["extra"], "provider": self.name, "model": settings.vlm_model}
        return merged

    def ocr_image(self, image: ImagePart) -> str:
        prompt = "请完整输出图片中的所有文字，保留换行，不要总结、不要添加说明。"
        payload_text = self._chat(prompt, [image], max_tokens=2000)
        return payload_text.strip()


@lru_cache
def get_vlm() -> BaseVLM:
    provider = settings.vlm_provider.lower()
    if provider in ("dashscope", "openai", "vllm", "compat"):
        if not settings.dashscope_api_key and "localhost" not in settings.vlm_base_url:
            logger.warning("未配置 DASHSCOPE_API_KEY，自动回退 mock provider")
            return MockVLM()
        try:
            return OpenAICompatVLM()
        except Exception as exc:  # noqa: BLE001
            logger.error("VLM 初始化失败，回退 mock: %s", exc)
            return MockVLM()
    return MockVLM()
