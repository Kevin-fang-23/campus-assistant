"""全局配置：所有外部依赖均可通过环境变量切换实现。"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, EnvSettingsSource, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent


class CorsEnvSource(EnvSettingsSource):
    """环境变量里的 cors_origins 可能是逗号分隔字符串，也可能是 JSON 数组。
    pydantic-settings 默认会把复杂类型按 JSON 解析，这里跳过 JSON 预解析，
    交给 Settings 的 field_validator 统一处理。"""

    def prepare_field_value(self, field_name, field, value, value_is_complex):
        if field_name == "cors_origins":
            return value
        return super().prepare_field_value(field_name, field, value, value_is_complex)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(BASE_DIR / ".env", BASE_DIR.parent / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @classmethod
    def settings_customise_sources(
        cls, settings_cls, init_settings, env_settings, dotenv_settings, file_secret_settings
    ):
        # 用自定义环境变量源替换默认 env 源，避免 cors_origins 被强制 JSON 解析
        return init_settings, CorsEnvSource(settings_cls), dotenv_settings, file_secret_settings

    app_name: str = "校园事务智能助手"
    debug: bool = True

    # 开发默认 SQLite，生产改成 postgresql+psycopg://user:pwd@host:5432/campus
    database_url: str = f"sqlite:///{(BASE_DIR / 'campus.db').as_posix()}"
    storage_dir: Path = BASE_DIR / "storage"

    # ---- 视觉大模型 / LLM ----
    # mock: 内置规则实现（无需 Key，全链路可跑通）；dashscope: OpenAI 兼容接口
    vlm_provider: str = "mock"
    vlm_model: str = "qwen3-vl-plus"          # 可改 qwen3.5-vl-plus / qwen3-vl-flash / 自托管模型名
    vlm_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    # 阿里云百炼密钥：优先 ALIYUN_API_KEY，DASHSCOPE_API_KEY 为历史别名（见下方校验器）
    aliyun_api_key: str = ""
    dashscope_api_key: str = ""
    vlm_timeout: float = 60.0
    vlm_max_retries: int = 2
    # 重试退避：第 n 次等待 = min(backoff_max, backoff_base * 2**n) * 抖动
    vlm_backoff_base: float = 0.8
    vlm_backoff_max: float = 8.0

    # ---- 检索增强问答（/api/qa）----
    # auto: 有 Key 走 LLM 生成，无 Key 降级为抽取式回答；mock: 强制降级（测试用）
    qa_provider: str = "auto"
    qa_model: str = "qwen-plus"              # 纯文本问答模型，无需视觉能力，比 VL 模型便宜

    # ---- OCR ----
    # auto: paddleocr -> rapidocr -> vlm -> stub 依次探测
    ocr_provider: str = "auto"
    ocr_lang: str = "ch"
    pdf_ocr_dpi: int = 180
    pdf_text_min_chars: int = 20              # 单页文本少于该值判定为扫描页，转图走 OCR

    # ---- 向量检索 ----
    embedding_provider: str = "auto"          # auto: dashscope(有 Key) -> local_hash
    embedding_model: str = "text-embedding-v4"
    embedding_dim: int = 256                  # local_hash 维度
    dedup_threshold: float = 0.90             # 余弦相似度高于此值判定重复通知
    search_top_k: int = 5

    # ---- 待办生成 ----
    remind_lead_hours: int = 24               # 截止前多久提醒
    review_confidence_threshold: float = 0.55  # 低于此置信度进人工复核

    cors_origins: list[str] = ["http://localhost:5173", "http://127.0.0.1:5173"]

    @model_validator(mode="after")
    def _merge_api_key(self) -> Settings:
        # 两个变量名任一有值即生效，避免改名后旧配置静默失效
        if not self.dashscope_api_key:
            object.__setattr__(self, "dashscope_api_key", self.aliyun_api_key)
        if not self.aliyun_api_key:
            object.__setattr__(self, "aliyun_api_key", self.dashscope_api_key)
        return self

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _parse_cors(cls, v: object) -> object:
        # 支持 .env 写 JSON 数组，也支持平台环境变量写逗号分隔字符串
        if isinstance(v, str):
            s = v.strip()
            if s.startswith("["):
                return json.loads(s)
            return [o.strip() for o in s.split(",") if o.strip()]
        return v


@lru_cache
def get_settings() -> Settings:
    s = Settings()
    s.storage_dir.mkdir(parents=True, exist_ok=True)
    return s


settings = get_settings()
