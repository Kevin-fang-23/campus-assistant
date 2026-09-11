"""向量化 Provider：dashscope(text-embedding-v4) 或本地哈希向量（零依赖兜底）。"""
from __future__ import annotations

import hashlib
import logging
import re
from abc import ABC, abstractmethod
from functools import lru_cache

import numpy as np

from ..config import settings
from .llm_client import SharedClientRef, get_shared_client

logger = logging.getLogger(__name__)

_TOKEN_SPLIT = re.compile(r"[\s,。，；;：:！!？?、\"'（）()\[\]【】/\\|_\-]+")

# text-embedding-v4 的输出维度。集中成常量，供向量库在维度告警中引用，
# 避免在多处硬编码魔数导致告警信息与实现脱节。
DASHSCOPE_DIM = 1024


class BaseEmbedding(ABC):
    name: str = "base"
    dim: int = 0

    @abstractmethod
    def embed(self, texts: list[str]) -> np.ndarray:
        ...

    def embed_one(self, text: str) -> np.ndarray:
        return self.embed([text])[0]

    @property
    def active_name(self) -> str:
        """**实际**产出向量的后端名。

        与 name（配置的后端）区分：DashScopeEmbedding 在接口不可用时会回退到
        本地哈希向量，此时必须报 local_hash，否则 /health、/api/search、
        /api/qa 的 backend 字段会谎报，前端与排查都会被误导。
        """
        return self.name

    @property
    def degraded(self) -> bool:
        """是否发生了运行时回退。默认（无回退机制的 provider）恒为 False。"""
        return False


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

    # 按需解析共享客户端，而非在 __init__ 里存一份失效引用
    # （共享实例被关停/重建后仍能自动拿到可用的那个，见 SharedClientRef 说明）
    _client = SharedClientRef()

    def __init__(self) -> None:
        self.dim = DASHSCOPE_DIM
        self._fallback = LocalHashEmbedding()
        self._degraded = False
        # 提前构造一次以校验 API Key：缺 Key 时抛 AuthError，
        # 由 get_embedding() 捕获后回退到 LocalHashEmbedding（保持原有行为）。
        get_shared_client()

    @property
    def degraded(self) -> bool:
        return self._degraded

    @property
    def active_name(self) -> str:
        return self._fallback.name if self._degraded else self.name

    def _degrade(self, exc: Exception) -> None:
        """切到本地哈希并**粘滞**（本进程内不再尝试上游）。

        为什么必须粘滞：local_hash 维度（256）与 dashscope（1024）不同，
        若允许后端来回切换，向量维度就会反复翻转，进而污染向量索引
        （详见 VectorStore._append 的维度守卫）。
        宁可确定性地降级 + 如实上报，也不要静默混用两种维度的向量。
        """
        self._degraded = True
        self.dim = self._fallback.dim
        logger.error(
            "Embedding 接口不可用，本进程后续全部改用本地哈希向量（%s，维度 %d）；"
            "backend 将如实标记为 %s。原因: %s",
            self._fallback.name,
            self._fallback.dim,
            self.active_name,
            exc,
        )

    def embed(self, texts: list[str]) -> np.ndarray:
        if self._degraded:  # 已判定上游不可用，直接走本地，不再白耗额度
            return self._fallback.embed(texts)
        try:
            vecs = np.array(
                self._client.embed([t[:2000] for t in texts], model=settings.embedding_model),
                dtype=np.float32,
            )
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            self.dim = vecs.shape[1]
            return vecs / np.maximum(norms, 1e-9)
        except Exception as exc:  # noqa: BLE001
            self._degrade(exc)
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
