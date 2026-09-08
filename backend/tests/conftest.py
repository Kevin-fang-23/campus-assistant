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

import pytest  # noqa: E402

from app.db import SessionLocal, init_db  # noqa: E402
from app.services.vector_store import get_store  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def _prepare_db():
    init_db()
    with SessionLocal() as db:
        get_store().load_from_db(db)
    yield


@pytest.fixture
def db():
    with SessionLocal() as session:
        yield session
