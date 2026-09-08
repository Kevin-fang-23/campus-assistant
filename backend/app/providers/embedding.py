"""向量化 Provider：dashscope(text-embedding-v4) 或本地哈希向量（零依赖兜底）。"""
from __future__ import annotations

import hashlib
import logging
import re
from abc import ABC, abstractmethod
from functools import lru_cache

import numpy as np

from ..config import settings
from .llm_client import build_client_from_settings

logger = logging.getLogger(__name__)

_TOKEN_SPLIT = re.compile(r"[\s,。，；;：:！!？?、\"'（）()\[\]【】/\\|_\-]+")


class BaseEmbedding(ABC):
    name: str = "base"
    dim: int = 0

    @abstractmethod
    def embed(self, texts: list[str]) -> np.ndarray:
        ...

    def embed_one(self, text: str) -> np.ndarray:
        return self.embed([text])[0]


class LocalHashEmbedding(BaseEmbedding):
    """字符 uni/bi-gram 哈希 + TF 权重 + L2 归一化。

    确定性、零网络依赖，对「近似重复通知」判定和粗粒度语义检索足够；
    需要更强语义时把 EMBEDDING_PROVIDER 切到 dashscope。
    """

    name = "local_hash"

    def __init__(self, dim: int | None = None) -> None:
        self.dim = int(dim or settings.embedding_dim)

    @staticmethod
    def _tokens(text: str) -> list[str]:
        clean = _TOKEN_SPLIT.sub(" ", (text or "").lower())
        chars = [c for c in clean if not c.isspace()]
        grams = list(chars)
        grams += ["".join(pair) for pair in zip(chars, chars[1:])]
        grams += [w for w in clean.split() if len(w) > 1]
        return grams

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for gram in self._tokens(text):
                h = int.from_bytes(hashlib.md5(gram.encode()).digest()[:8], "little")
                idx = h % self.dim
                sign = 1.0 if (h >> 63) & 1 == 0 else -1.0
                out[row, idx] += sign
            norm = float(np.linalg.norm(out[row]))
            if norm > 0:
                out[row] /= norm
        return out


class DashScopeEmbedding(BaseEmbedding):
    name = "dashscope"

    def __init__(self) -> None:
        self.dim = 1024
        # 与 VLM 共用同一套鉴权 / 超时 / 重试策略
        self._client = build_client_from_settings()
        self._fallback = LocalHashEmbedding()

    def embed(self, texts: list[str]) -> np.ndarray:
        try:
            vecs = np.array(
                self._client.embed([t[:2000] for t in texts], model=settings.embedding_model),
                dtype=np.float32,
            )
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            self.dim = vecs.shape[1]
            return vecs / np.maximum(norms, 1e-9)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Embedding 接口失败，回退本地哈希向量: %s", exc)
            return self._fallback.embed(texts)


@lru_cache
def get_embedding() -> BaseEmbedding:
    want = settings.embedding_provider.lower()
    if want in ("dashscope", "auto") and settings.dashscope_api_key:
        try:
            return DashScopeEmbedding()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Embedding provider 初始化失败: %s", exc)
    return LocalHashEmbedding()
