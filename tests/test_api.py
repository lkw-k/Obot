import shutil
import threading
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from obot.api.app import create_app
from obot.config import BotConfig, Secrets, Settings
from obot.llm.base import LLMError

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


class BlockingEmbedder:
    """Holds builds until released, so tests can observe the building status."""

    def __init__(self, inner):
        self.inner = inner
        self.release = threading.Event()

    def encode(self, texts):
        self.release.wait(timeout=10)
        return self.inner.encode(texts)


class FailingLLM:
    def chat(self, system, messages, json_mode=False):
        raise LLMError("provider down")


@pytest.fixture
def bots(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    shutil.copy(EXAMPLES / "orders.csv", data / "orders.csv")
    return [BotConfig(name="orders", files=[data / "orders.csv"])]


def _client(tmp_path, bots, embedder, llm, api_key=None):
    settings = Settings(secrets=Secrets(_env_file=None, obot_api_key=api_key))
    app = create_app(settings, bots, llm, lambda: embedder, tmp_path / "storage")
    return TestClient(app)


def _wait_status(client, name, expected, headers=None):
    for _ in range(200):
        status = client.get(f"/bots/{name}", headers=headers).json()["status"]
        if status == expected:
            return
        time.sleep(0.05)
    raise AssertionError(f"{name} never became {expected} (last: {status})")


@pytest.fixture
def client(tmp_path, bots, fake_embedder, fake_llm):
    with _client(tmp_path, bots, fake_embedder, fake_llm) as c:
        _wait_status(c, "orders", "ready")
        yield c


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_list_bots(client):
    assert client.get("/bots").json() == [{"name": "orders", "status": "ready"}]


def test_bot_detail_has_record_count_and_schema(client):
    body = client.get("/bots/orders").json()

    assert body["status"] == "ready"
    assert body["record_count"] == 28
    assert body["schema"]["id_field"]
    assert body["error"] is None


def test_chat_returns_answer_with_sources(client):
    response = client.post("/bots/orders/chat", json={"message": "late orders?"})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "fake answer"
    assert body["route"] == "search"
    assert body["sql"] is None
    assert len(body["sources"]) == 5
    assert {"id", "score", "record"} <= body["sources"][0].keys()


def test_chat_passes_history_and_top_k(client, fake_llm):
    history = [{"role": "user", "content": "hi"}]

    response = client.post(
        "/bots/orders/chat", json={"message": "q", "history": history, "top_k": 2}
    )

    assert len(response.json()["sources"]) == 2
    assert fake_llm.calls[-1]["messages"][:-1] == history


@pytest.mark.parametrize(
    "body", [{}, {"message": ""}, {"message": "q", "top_k": 0}, {"message": 1}]
)
def test_invalid_chat_request_is_422(client, body):
    response = client.post("/bots/orders/chat", json=body)

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "INVALID_REQUEST"


def test_undecodable_body_is_422(client):
    response = client.post(
        "/bots/orders/chat",
        content='{"message": "서울"}'.encode("cp949"),
        headers={"content-type": "application/json"},
    )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "INVALID_REQUEST"


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("get", "/bots/nope"),
        ("post", "/bots/nope/chat"),
        ("post", "/bots/nope/rebuild"),
    ],
)
def test_unknown_bot_is_404(client, method, path):
    response = client.request(method, path, json={"message": "q"})

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "BOT_NOT_FOUND"


def test_rebuild_returns_202_and_becomes_ready(client):
    response = client.post("/bots/orders/rebuild")

    assert response.status_code == 202
    assert response.json() == {"name": "orders", "status": "building"}
    _wait_status(client, "orders", "ready")


def test_building_bot_rejects_chat_and_rebuild_with_409(
    tmp_path, bots, fake_embedder, fake_llm
):
    embedder = BlockingEmbedder(fake_embedder)
    with _client(tmp_path, bots, embedder, fake_llm) as c:
        try:
            assert c.get("/bots/orders").json()["status"] == "building"
            chat = c.post("/bots/orders/chat", json={"message": "q"})
            rebuild = c.post("/bots/orders/rebuild")
        finally:
            embedder.release.set()

    assert chat.status_code == 409
    assert chat.json()["error"]["code"] == "BOT_NOT_READY"
    assert rebuild.status_code == 409
    assert rebuild.json()["error"]["code"] == "BOT_BUILDING"


def test_failed_bot_rejects_chat_with_409(tmp_path, fake_embedder, fake_llm):
    bots = [BotConfig(name="gone", files=[tmp_path / "missing.csv"])]
    with _client(tmp_path, bots, fake_embedder, fake_llm) as c:
        _wait_status(c, "gone", "failed")
        detail = c.get("/bots/gone").json()
        chat = c.post("/bots/gone/chat", json={"message": "q"})

    assert "file not found" in detail["error"]
    assert chat.status_code == 409


def test_llm_error_is_502(tmp_path, bots, fake_embedder):
    with _client(tmp_path, bots, fake_embedder, FailingLLM()) as c:
        _wait_status(c, "orders", "ready")
        response = c.post("/bots/orders/chat", json={"message": "q"})

    assert response.status_code == 502
    assert response.json()["error"] == {"code": "LLM_ERROR", "message": "provider down"}


def test_api_key_required_when_set(tmp_path, bots, fake_embedder, fake_llm):
    with _client(tmp_path, bots, fake_embedder, fake_llm, api_key="secret") as c:
        health = c.get("/health")
        missing = c.get("/bots")
        wrong = c.get("/bots", headers={"X-API-Key": "nope"})
        right = c.get("/bots", headers={"X-API-Key": "secret"})

    assert health.status_code == 200
    assert missing.status_code == 401
    assert missing.json()["error"]["code"] == "UNAUTHORIZED"
    assert wrong.status_code == 401
    assert right.status_code == 200
