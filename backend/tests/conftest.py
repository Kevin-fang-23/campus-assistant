from __future__ import annotations

import os
import tempfile
from pathlib import Path

# 必须在导入 app.config 之前设置，避免污染开发库
_TMP = Path(tempfile.mkdtemp(prefix="campus_test_"))
os.environ["DATABASE_URL"] = f"sqlite:///{(_TMP / 'test.db').as_posix()}"
os.environ["STORAGE_DIR"] = str(_TMP / "storage")
os.environ["VLM_PROVIDER"] = "mock"
os.environ["OCR_PROVIDER"] = "stub"
# QA 强制走降级路径（抽取式回答），测试不发起真实 LLM 调用；
# LLM 路径由专门用例注入 MockTransport 验证。
os.environ["QA_PROVIDER"] = "mock"
# 限流默认关闭：既有用例共用同一个 TestClient，反复请求会互相干扰阈值判定。
# test_rate_limit.py 会自行开启并设定小阈值来验证限流本身。
os.environ["RATE_LIMIT_ENABLED"] = "false"
# /api/qa 缓存默认关闭：既有的降级/限流测试假定"每次都走完整流程"，
# 与新加缓存层不耦合。test_qa_cache.py 自行开启并验证缓存本身。
os.environ["QA_CACHE_ENABLED"] = "false"

# Embedding 固定为本地哈希 —— **测试必须与真实上游解耦**。
#
# 不固定会出两类问题，且都已在本地（配置了真实 Key 的 .env）实测复现：
#
# 1) **不确定性**：dashscope 是远端服务，向量结果随版本/服务端状态变化。
#    而检索类断言（阈值判定、top-k 内容、nDCG）都建立在"向量可复现"之上，
#    挂真实上游等于把测试结果交给第三方抖动。
# 2) **进程级粘性降级污染后续用例**：DashScopeEmbedding 一旦因上游故障降级到
#    local_hash 就粘滞（dim 1024 → 256），而 get_embedding() 是 lru_cache 单例。
#    test_embedding_backend_truth.py 会**故意**触发一次降级来验证该设计，
#    于是之后所有 load_from_db 都用「期望 256 维」筛「实际 1024 维」的历史向量，
#    **全部被跳过**、向量索引坍缩到接近空（实测 store=1 而库内 5 条通知），
#    表现为 test_hybrid 的孤儿断言"单文件绿、全量红"。
#
# 固定为 local_hash 后：维度恒定 256、纯本地计算（无网络、毫秒级）、
# 结果完全可复现。这也正是 **CI 的实际运行条件**（CI 无 Key，
# embedding_provider=auto 自然落到 local_hash），因此本地与 CI 行为一致。
#
# 需要验证 dashscope 相关行为的用例（test_embedding_backend_truth.py）
# 自行构造 DashScopeEmbedding 实例，不依赖这里的全局单例。
os.environ["EMBEDDING_PROVIDER"] = "local_hash"

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.db import SessionLocal, init_db  # noqa: E402
from app.main import app  # noqa: E402
from app.services.qa_cache import reset_cache_for_tests  # noqa: E402
from app.services.vector_store import get_store  # noqa: E402

# 检索类用例（/api/search、/api/qa）必须有语料才能召回。
# 这些语料此前是靠 test_api.py 里"先跑的入库用例"顺带写进去的——
# 一旦单独运行某个检索用例（pytest tests/test_api.py::test_qa_degraded_path）
# 向量库为空，用例就会假失败。这里显式播种，消除执行顺序依赖。
#
# 注意：内容必须与 test_api.py 的 HOMEWORK **不同**。用同内容播种会命中
# 内容级缓存，导致 test_text_ingest_flow 的 trace 首节点从 classify 变成 cache。
_SEED_NOTICES = [
    (
        "《操作系统》第三次小班课通知\n本周五 15:00 在教三 201 讲解进程调度实验。\n"
        "预习材料已上传学习通，课前请完成阅读。\n",
        "os_seminar_seed.txt",
    ),
    (
        "关于举办校园创客社团旧手机拆解工作坊的通知\n时间：10月12日 14:00\n"
        "地点：图书馆南门集合\n主办：创客社团\n报名截止：10月10日 18:00\n",
        "maker_seed.txt",
    ),
    (
        "《线性代数》期末考试安排\n考试时间：10月20日 09:00-11:00\n考场：教三 405\n"
        "请携带学生证按时参加，迟到 15 分钟不得入场。\n",
        "linalg_seed.txt",
    ),
]


@pytest.fixture(scope="session", autouse=True)
def _prepare_db():
    init_db()
    with SessionLocal() as db:
        get_store().load_from_db(db)
    yield


@pytest.fixture(scope="session", autouse=True)
def _seed_corpus(_prepare_db):
    """播种检索语料，使检索类用例不依赖文件内/文件间的执行顺序。

    入库走真实 HTTP 接口（而非直接写库），保证走的是完整 pipeline，
    与生产路径一致；重复入库由 sha256 去重兜底，不会污染断言。
    """
    with TestClient(app) as c:
        for content, filename in _SEED_NOTICES:
            c.post("/api/documents/text", json={"content": content, "filename": filename})
    yield


@pytest.fixture(autouse=True)
def _qa_cache_clean():
    """每个用例前后清空 /api/qa 的缓存单例。

    缓存是**进程级单例**（与 `RateLimiter` / `VectorStore` 同处），
    而 TestClient 是 module 级 fixture 跨用例共享 —— 不显式清空的话，
    同一 query 的前序用例写入的响应会让后续用例命中缓存、跳过 LLM/embedding
    调用，破坏"每个用例独立验证某条路径"的契约。

    autouse=True 让所有测试自动应用，不需每个文件显式声明。
    """
    reset_cache_for_tests()
    yield
    reset_cache_for_tests()


@pytest.fixture
def db():
    with SessionLocal() as session:
        yield session


def reload_indexes_from_db() -> None:
    """按库内现状重建**两路**索引 —— 用例写完临时数据后的统一收尾手段。

    为什么必须这样收尾：`index_notice()` 会同时写向量索引与 BM25 索引，
    而两路都**没有单条删除接口**（VectorStore 只能 `load_from_db` 全量重建，
    BM25 只能 `load_documents` 全量重建；`reset_indexes()` 更是只把 BM25
    单例置 None）。因此用例删除临时通知后，索引里必然残留指向已删通知的条目。

    残留的后果不是"多召回一条"这么轻 —— 它会让**相关性阈值判定**失去依据：
    阈值要看"查询与文档共享的二字词个数"，而孤儿文档已从库里删掉、取不到文本，
    `is_relevant` 按约定对取不到文本的情况**放开**（宁可多给一条也不误杀），
    于是幽灵条目反而绕过了阈值。表现为"单跑绿、全量跑红"的顺序依赖缺陷。

    统一在收尾处调用本函数，让索引与库保持一致。
    """
    from app.services.hybrid import rebuild_bm25
    from app.services.vector_store import get_store

    with SessionLocal() as db:
        get_store().load_from_db(db)
        rebuild_bm25(db)
