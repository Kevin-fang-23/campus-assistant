from __future__ import annotations

import io

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


def test_validation_errors(client: TestClient) -> None:
    assert client.post("/api/documents/text", json={"content": ""}).status_code == 422
    assert client.get("/api/tasks", params={"status": "unknown"}).status_code == 400
    assert client.get("/api/tasks/999999").status_code == 404
