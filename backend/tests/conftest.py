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

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.db import SessionLocal, init_db  # noqa: E402
from app.main import app  # noqa: E402
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


@pytest.fixture
def db():
    with SessionLocal() as session:
        yield session
