from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

ImagePart = tuple[str, bytes]


@dataclass
class ParsedDoc:
    """多模态输入归一化后的统一结构，下游只认这一个契约。"""

    kind: str                      # image | pdf | text
    text: str = ""
    pages: int = 1
    images: list[ImagePart] = field(default_factory=list)  # 交给视觉模型的原图/页图
    meta: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    @property
    def char_count(self) -> int:
        return len(self.text.strip())
