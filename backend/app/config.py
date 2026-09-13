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

    # ---- 请求限流（保护上游 LLM 额度与单进程资源）----
    # 三层：每 IP 每分钟（突发控制）→ 每 IP 每日（单源公平）→ 全局每日（总额度护栏）
    # 0 表示该层不启用。改这些值即可整体放宽/收紧，无需改代码。
    rate_limit_enabled: bool = True
    # 信任 X-Forwarded-For / X-Real-IP：置于 ngrok 等反代之后必须开启，
    # 否则所有请求会被视作同一来源；直连公网时保持 False，防止伪造头部绕过限流。
    rate_limit_trust_proxy: bool = False

    # 成本端点：/api/qa 每次请求消耗 1 次 LLM + 1 次 embedding
    rate_limit_qa_per_min: int = 6
    rate_limit_qa_per_ip_day: int = 60
    rate_limit_qa_per_day: int = 300

    # 成本端点：/api/documents/* 每次请求消耗 VLM + OCR 解析
    rate_limit_ingest_per_min: int = 10
    rate_limit_ingest_per_ip_day: int = 40
    rate_limit_ingest_per_day: int = 200

    # 其余 /api/* 只读接口：不花钱，仅防高频锤击
    rate_limit_default_per_min: int = 120

    # 日计数是否持久化（解决清单 #8 多 worker 各算一份 / #9 重启清零）。
    # sqlite: 写入当前数据库的 rate_limit_counters 表（零新依赖，默认）；
    # none:   只留进程内内存（等价改造前行为，便于回滚；**不跨进程/重启**）。
    # 多实例（多台机器）部署时 SQLite 文件无法共享，需换 Redis —— 见
    # app/services/rate_limit_store.py 的 CountStore 协议。
    rate_limit_store: str = "sqlite"
    # ⚠️ 已废弃，保留仅为兼容旧 .env（改了不影响任何行为）。
    # 日计数现在是「判定即读库、记账即写库」，不存在批量刷盘，
    # 因此没有「刷盘间隔」这个可调项，也就没有「重启丢一个间隔」的窗口。
    rate_limit_flush_interval: float = 1.0

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
    # 启动时若发现库内向量维度与当前后端不符（历史遗留/后端切换），
    # 自动用当前后端重算全部向量。关掉则只告警、由人工触发 reindex。
    reindex_on_dim_mismatch: bool = True

    # ---- 混合检索（向量 + BM25）----
    # 关掉即退回纯向量检索（与改造前行为一致），便于出问题时快速回滚。
    hybrid_enabled: bool = True
    # RRF 平滑常数 k：越大则靠前名次的优势越平缓。60 取自 RRF 原论文。
    hybrid_rrf_k: int = 60
    # 两路权重。默认 0.1:0.9（BM25 主导），由 eval/run_retrieval_eval.py
    # 在 **55 条语料 / 129 条查询**上按 0.05 步长扫描实测：
    #   · 纯向量 MRR@5=0.933、纯 BM25=0.980、混合各权重档均 ≤0.974；
    #   · 曲线**单调下降**（0.00→0.980、0.05→0.974、0.10→0.972、
    #     0.20→0.963），且互补性分析显示**向量路从未成为唯一赢家**；
    #   · 即：本语料上 BM25 已足够强，加大向量权重只会扰动排序。
    #
    # 为什么保留 0.1 而不是取实测最优的 0.0：
    #   · 语料仅 55 条，字面碰撞少，对 BM25 最有利；语料扩大后 BM25 的
    #     区分度会下降、向量路的相对价值上升（见 RETRIEVAL_BASELINE.md 局限说明）；
    #   · 「零字面重叠的改写查询」是向量路唯一不可替代的场景，
    #     0.1 的代价仅 0.8 个百分点，属可接受的保险。
    #   · 若确定只服务小语料、追求实测最高分，可直接设 0.0 —— 即纯 BM25。
    #
    # ⚠️ 换语料后务必重跑 --sweep 重新定标，不要沿用本值。
    hybrid_weight_vector: float = 0.1
    hybrid_weight_bm25: float = 0.9
    # 每路候选池大小。融合前多召回一些，才能让"向量排 12、BM25 排 2"
    # 的文档有机会被顶上来；等于 top_k 就失去了融合的意义。
    hybrid_fetch_k: int = 20
    # BM25 参数：k1 控制词频饱和，b 控制文档长度归一化强度。1.5/0.75 为文献常用值。
    bm25_k1: float = 1.5
    bm25_b: float = 0.75

    # ---- 检索相关性阈值（提示「未找到相关内容」）----
    # 语义：**候选文档与查询至少要有几个「共享二字词」才算相关**。
    # 小于该值的候选一律丢弃；若全部被丢弃，接口返回空列表并置 filtered=True。
    #
    # 为什么用「共享 bigram 个数」这一判据，而不是分数阈值：
    #   eval 里实测三种分数判据（BM25 绝对分 / 余弦 / 句级选片分）**全部区间重叠**，
    #   无法分离「无关」与「相关」——
    #     · BM25：噪声最高 3.773 >= 真实最低 2.872
    #     · 余弦：噪声最高 0.154 >= 真实最低 0.043
    #     · 选片：噪声最高 26.000 >= 真实最低 3.422
    #   根因是**单字重合**在中文里必然发生（'今天天气怎么样' 撞 '明天' 里的 '天'）。
    #   改数**二字及以上的重合**（bigram）即可过滤掉偶然碰撞：
    #     今天天气怎么样 → 0 个共享二字词（只剩单字 '天'）
    #   而 129 条真实查询的下界恰好是 1 个，因此 T=1 能零成本拦下噪声。
    #
    # 为什么是 1（而非更严的 2）：
    #   实测 T=2 虽能再多拦下 1 条噪声（"量子纠缠退相干周期"），
    #   代价是误杀 9/129 条真实查询（"怎么加入社团""快递站在哪"…）—— 不可接受。
    #   T=1 的真实误杀为 **0/129**。取舍见 RETRIEVAL_BASELINE.md 第五节。
    #
    # ⚠️ 该结论建立在**仅 2 条噪声查询**之上，样本极小：
    #   "0/129 误杀"这个数字是可靠的，但"能拦掉多少真实噪声"远未被充分验证。
    #   要评估需先扩一批真实无答案查询进 retrieval_set.json。
    #
    # 0 表示关闭判定（退回"恒返回 top-k"的旧行为，可随时回滚）。
    search_min_bigram_overlap: int = 1
    # 阈值判定失败时是否仍然返回原始结果。True = 只标记不丢弃（便于灰度观察），
    # False = 真正丢弃（默认，产品语义是"没找到就别硬凑"）。
    search_keep_if_filtered: bool = False

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
