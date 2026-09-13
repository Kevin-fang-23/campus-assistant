from __future__ import annotations

import io
import json
from datetime import datetime

import pytest
from fastapi.testclient import TestClient

from app.main import app

HOMEWORK = """《操作系统》第二次大作业要求
实现一个简单的进程调度模拟器，提交源代码与实验报告。
截止时间：9月18日 22:00，提交至学习通。
助教联系邮箱：ta_os@campus.edu.cn
"""


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health(client: TestClient) -> None:
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["pipeline_engine"] in ("langgraph", "sequential")
    assert "provider" in body["vlm"]


def test_text_ingest_flow(client: TestClient) -> None:
    resp = client.post("/api/documents/text", json={"content": HOMEWORK, "filename": "os_hw.txt"})
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["notice"]["category"] == "homework"
    assert data["notice"]["deadline"].startswith("2026-09-18")
    assert data["tasks"], "应生成待办"
    assert [t["node"] for t in data["trace"]][0] == "classify"


def test_upload_file_and_dedup(client: TestClient) -> None:
    files = {"file": ("poster.txt", io.BytesIO(HOMEWORK.encode()), "text/plain")}
    first = client.post("/api/documents/upload", files=files)
    assert first.status_code == 200
    # 同内容再传一次 → 命中 sha256 文件级去重
    files = {"file": ("poster_copy.txt", io.BytesIO(HOMEWORK.encode()), "text/plain")}
    second = client.post("/api/documents/upload", files=files)
    assert second.json()["duplicate"] is True


def test_task_status_flow_and_timeline(client: TestClient) -> None:
    tasks = client.get("/api/tasks", params={"category": "homework"}).json()
    assert tasks
    task_id = tasks[0]["id"]

    doing = client.patch(f"/api/tasks/{task_id}", json={"status": "doing", "note": "开始写代码"})
    assert doing.status_code == 200
    done = client.patch(f"/api/tasks/{task_id}", json={"status": "done"})
    body = done.json()
    assert body["status"] == "done"
    assert body["completed_at"]
    flows = [(e["from_status"], e["to_status"]) for e in body["events"]]
    assert (None, "todo") in flows and ("todo", "doing") in flows and ("doing", "done") in flows


def test_manual_task_and_stats(client: TestClient) -> None:
    created = client.post(
        "/api/tasks",
        json={"title": "去图书馆还书", "category": "other", "priority": 2},
    )
    assert created.status_code == 201
    stats = client.get("/api/tasks/stats").json()
    assert stats["tasks_total"] >= 1
    assert stats["notices"] >= 1


def test_notice_review_and_regenerate(client: TestClient) -> None:
    notices = client.get("/api/notices", params={"category": "homework"}).json()
    assert notices
    nid = notices[0]["id"]
    resp = client.patch(
        f"/api/notices/{nid}",
        json={"title": "操作系统大作业（人工修正）", "deadline": "2026-09-20T20:00:00"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["needs_review"] is False and body["reviewed"] is True
    assert body["title"] == "操作系统大作业（人工修正）"
    assert any(t["due_at"].startswith("2026-09-20") for t in body["tasks"])


def test_semantic_search(client: TestClient) -> None:
    resp = client.post("/api/search", json={"query": "操作系统 作业 截止", "top_k": 3})
    assert resp.status_code == 200
    body = resp.json()
    assert body["hits"], "向量库应能召回已入库通知"
    assert body["hits"][0]["score"] > 0


def test_qa_degraded_path(client: TestClient) -> None:
    """QA_PROVIDER=mock 时走降级路径：抽取式回答 + 引用，响应结构不变。"""
    resp = client.post("/api/qa", json={"query": "操作系统作业什么时候截止", "top_k": 3})
    assert resp.status_code == 200
    body = resp.json()
    assert body["degraded"] is True
    assert body["query"] == "操作系统作业什么时候截止"
    assert body["answer"], "降级路径也应给出 top1 抽取式回答"
    assert body["citations"], "降级路径引用列表不能为空"
    for c in body["citations"]:
        assert c["notice_id"] and c["title"] and c["snippet"]
        assert c["score"] > 0
    assert body["backend"]


def test_qa_llm_path_with_mock_transport(client: TestClient, monkeypatch) -> None:
    """LLM 路径：注入 MockTransport 的 OpenAICompatClient，验证真实调用链。

    必须复用 build_client_from_settings 同款客户端（含鉴权头构造与响应解析），
    只把网络层换成 MockTransport——保证「客户端封装零新增」可被测试证明。
    """
    import httpx

    from app.api import qa as qa_module
    from app.providers.llm_client import OpenAICompatClient

    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["auth"] = request.headers.get("Authorization")
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={
            "choices": [{"message": {"content": "根据[1]，截止时间是9月18日22:00。"}}]
        })

    def fake_client() -> OpenAICompatClient:
        return OpenAICompatClient(
            base_url="https://mock.test/v1",
            api_key="test-key",
            transport=httpx.MockTransport(handler),
            sleep_fn=lambda s: None,
        )

    monkeypatch.setattr(qa_module, "_build_llm_client", fake_client)
    monkeypatch.setattr(qa_module.settings, "qa_provider", "auto")
    monkeypatch.setattr(qa_module.settings, "dashscope_api_key", "test-key")

    resp = client.post("/api/qa", json={"query": "作业截止时间", "top_k": 3})
    assert resp.status_code == 200
    body = resp.json()
    assert body["degraded"] is False
    assert "9月18日" in body["answer"]
    assert body["citations"]
    # 走的是 OpenAICompatClient 的标准调用链：Bearer 鉴权头 + chat/completions
    assert captured["auth"] == "Bearer test-key"
    assert captured["body"]["model"] == qa_module.settings.qa_model
    assert captured["body"]["messages"][0]["role"] == "system"


def test_qa_validation_errors(client: TestClient) -> None:
    """入参校验与 /api/search 同构：空 query 422，top_k 越界 422。"""
    assert client.post("/api/qa", json={"query": ""}).status_code == 422
    assert client.post("/api/qa", json={"query": "x", "top_k": 0}).status_code == 422
    assert client.post("/api/qa", json={"query": "x", "top_k": 99}).status_code == 422


def test_validation_errors(client: TestClient) -> None:
    assert client.post("/api/documents/text", json={"content": ""}).status_code == 422
    assert client.get("/api/tasks", params={"status": "unknown"}).status_code == 400
    assert client.get("/api/tasks/999999").status_code == 404


# ---------------------------------------------------------------------------
# 时区回归：客户端提交 tz-aware 时间（`...Z`）
#
# 背景：浏览器端用 `new Date(v).toISOString()` 提交 datetime-local 的值，
# 产出的是**带 Z 的 tz-aware** 时间串；而项目内部（services/datetime_utils、
# graph/nodes）一律用 naive 本地时间。两者相减会抛
#   TypeError: can't subtract offset-naive and offset-aware datetimes
# 于是「通知复核 → 保存并重置待办」稳定 500（崩溃点在 _priority_by_due）。
#
# 这几条用例存在的理由：修复前**整个测试套件只有 naive 时间字面量**，
# 从未覆盖浏览器实际发出的形式，所以 CI 全绿却漏掉了这个真实缺陷。
# 断言刻意写成「等于同一瞬时的本地墙钟」，而不是写死某个小时数，
# 这样在任何时区的机器上都能同时抓住「崩溃」和「静默偏移」两类问题。
# ---------------------------------------------------------------------------


def _local_wall_clock(iso_utc: str) -> str:
    """把 `...Z` 时间换算成本地墙钟字符串（与 to_naive_local 的语义一致）。"""
    aware = datetime.fromisoformat(iso_utc.replace("Z", "+00:00"))
    return aware.astimezone().replace(tzinfo=None).isoformat()


def test_notice_update_accepts_timezone_aware_datetime(client: TestClient) -> None:
    """通知人工复核：提交带 Z 的时间不得 500，且落库值不得偏移。"""
    notices = client.get("/api/notices").json()
    assert notices, "需要至少一条通知"
    nid = notices[0]["id"]

    aware = "2026-10-01T08:12:00.000Z"
    resp = client.patch(
        f"/api/notices/{nid}",
        json={"title": "时区回归：通知", "deadline": aware, "event_time": aware},
    )
    assert resp.status_code == 200, resp.text

    body = resp.json()
    expected = _local_wall_clock(aware)
    assert body["deadline"] == expected, f"deadline 偏移了：{body['deadline']} != {expected}"
    assert body["event_time"] == expected
    # 待办被重建 —— 说明确实走过了崩溃点 _priority_by_due
    assert body["tasks"], "应生成待办（崩溃点就在优先级计算里）"


def test_notice_update_with_naive_datetime_still_works(client: TestClient) -> None:
    """兼容性：naive 入参（旧客户端 / 测试）行为完全不变。"""
    notices = client.get("/api/notices").json()
    nid = notices[0]["id"]
    resp = client.patch(
        f"/api/notices/{nid}",
        json={"title": "时区回归：naive 不变", "deadline": "2026-11-05T09:30:00"},
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["deadline"] == "2026-11-05T09:30:00"


def test_task_create_accepts_timezone_aware_datetime(client: TestClient) -> None:
    """同一缺陷在待办接口上的回归（Tasks.tsx 此前也用 toISOString()）。"""
    aware = "2026-10-01T08:12:00.000Z"
    resp = client.post(
        "/api/tasks",
        json={"title": "时区回归：待办", "category": "other", "due_at": aware},
    )
    assert resp.status_code == 201, resp.text
    expected = _local_wall_clock(aware)
    assert resp.json()["due_at"] == expected, "due_at 偏移了"


def test_task_update_accepts_timezone_aware_datetime(client: TestClient) -> None:
    """待办编辑同样收 tz-aware 时间。"""
    created = client.post(
        "/api/tasks", json={"title": "时区回归：待办编辑", "category": "other"}
    ).json()
    aware = "2026-12-31T15:00:00.000Z"
    resp = client.patch(f"/api/tasks/{created['id']}", json={"due_at": aware})
    assert resp.status_code == 200, resp.text
    assert resp.json()["due_at"] == _local_wall_clock(aware)
