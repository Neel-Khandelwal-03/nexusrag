"""The container scripts: index bootstrapping and the health check."""

from __future__ import annotations

import importlib.util
import sys
import threading
from collections.abc import Callable, Iterator
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from nexusrag.config import Settings
from nexusrag.store import Stores
from tests.fakes import FakeGenAI, hash_embedder

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def load(name: str) -> ModuleType:
    """Import a file from scripts/ (it isn't a package)."""
    spec = importlib.util.spec_from_file_location(f"scripts_{name}", SCRIPTS / f"{name}.py")
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


bootstrap_index = load("bootstrap_index")
healthcheck = load("healthcheck")


# --------------------------------------------------------------------------- bootstrap


@pytest.fixture
def docs(tmp_path: Path) -> Path:
    data = tmp_path / "docs"
    data.mkdir()
    (data / "guide.md").write_text(
        "# Drone Guide\n\n## Battery\n\nThe battery lasts 46 minutes.\n", encoding="utf-8"
    )
    return data


@pytest.fixture
def settings(make_settings: Callable[..., Settings], tmp_path: Path, docs: Path) -> Settings:
    return make_settings(storage_dir=tmp_path / "storage", data_dir=docs)


@pytest.fixture
def patched(
    monkeypatch: pytest.MonkeyPatch, settings: Settings, fake_genai: FakeGenAI
) -> FakeGenAI:
    """Point the script at the test settings and a fake Gemini."""
    from nexusrag.llm.gemini_client import GeminiClient

    fake_genai.models.embed_fn = hash_embedder()
    monkeypatch.setattr(bootstrap_index, "get_settings", lambda: settings)
    monkeypatch.setattr(
        bootstrap_index,
        "GeminiClient",
        lambda s, **kwargs: GeminiClient(s, client=fake_genai),
    )
    return fake_genai


async def test_bootstrap_indexes_once(settings: Settings, patched: FakeGenAI, docs: Path) -> None:
    assert await bootstrap_index.bootstrap(docs, "default", force=False) == 1
    stores = Stores.open(settings)
    assert [d.filename for d in stores.registry.list_documents("default")] == ["guide.md"]
    stores.close()

    # A second start must not re-embed anything.
    embeds = len(patched.models.calls_to("embed_content"))
    assert await bootstrap_index.bootstrap(docs, "default", force=False) == 0
    assert len(patched.models.calls_to("embed_content")) == embeds
    assert await bootstrap_index.bootstrap(docs, "default", force=True) == 1


async def test_bootstrap_without_documents(settings: Settings, patched: FakeGenAI) -> None:
    assert await bootstrap_index.bootstrap(Path("does-not-exist"), "default", force=False) == 0


def test_bootstrap_failure_does_not_block_startup(
    monkeypatch: pytest.MonkeyPatch, settings: Settings, patched: FakeGenAI
) -> None:
    async def boom(*args: Any, **kwargs: Any) -> int:
        raise OSError("disk full")

    monkeypatch.setattr(bootstrap_index, "bootstrap", boom)
    assert bootstrap_index.main([]) == 0  # the app still starts and explains itself
    assert bootstrap_index.main(["--require"]) == 1


# --------------------------------------------------------------------------- health check


class _Handler(BaseHTTPRequestHandler):
    status = 200

    def do_GET(self) -> None:
        self.send_response(self.status)
        self.end_headers()
        self.wfile.write(b"ok")

    def log_message(self, *args: Any) -> None:
        pass


@pytest.fixture
def server() -> Iterator[HTTPServer]:
    httpd = HTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield httpd
    httpd.shutdown()
    httpd.server_close()


def test_healthcheck_reports_server_state(server: HTTPServer) -> None:
    url = f"http://127.0.0.1:{server.server_port}/"
    healthy, detail = healthcheck.check(url, timeout_s=5)
    assert healthy
    assert "200" in detail

    _Handler.status = 503  # the server is up but broken
    try:
        healthy, detail = healthcheck.check(url, timeout_s=5)
        assert not healthy
        assert "503" in detail
    finally:
        _Handler.status = 200

    healthy, detail = healthcheck.check("http://127.0.0.1:1/", timeout_s=1)
    assert not healthy  # nothing listening


def test_healthcheck_exit_codes(server: HTTPServer, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PORT", str(server.server_port))
    assert healthcheck.main([]) == 0
    assert healthcheck.main(["--url", "http://127.0.0.1:1/", "--timeout", "1"]) == 1


# --------------------------------------------------------------------------- deploy

deploy_space = load("deploy_space")


class FakeApi:
    """Records what the deploy script asks the Hub to do."""

    def __init__(self, stages: list[str] | None = None) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.stages = stages or ["RUNNING"]

    def create_repo(self, **kwargs: Any) -> None:
        self.calls.append(("create_repo", kwargs))

    def add_space_variable(self, **kwargs: Any) -> None:
        self.calls.append(("variable", kwargs))

    def upload_file(self, **kwargs: Any) -> None:
        self.calls.append(("upload_file", kwargs))

    def upload_folder(self, **kwargs: Any) -> None:
        self.calls.append(("upload_folder", kwargs))

    def get_space_runtime(self, **kwargs: Any) -> Any:
        stage = self.stages.pop(0) if len(self.stages) > 1 else self.stages[0]
        return type("Runtime", (), {"stage": stage})()


def test_deploy_creates_the_space_and_uploads_code() -> None:
    api = FakeApi()
    url = deploy_space.deploy(
        api,
        "me/nexusrag-staging",
        environment="staging",
        branch="staging",
        repo="me/nexusrag",
        port=8000,
        commit_message="Deploy abc1234",
    )
    assert url == "https://huggingface.co/spaces/me/nexusrag-staging"
    kinds = [kind for kind, _ in api.calls]
    assert kinds[0] == "create_repo"
    created = api.calls[0][1]
    assert created["space_sdk"] == "docker"
    assert created["exist_ok"] is True

    variables = {c["key"]: c["value"] for kind, c in api.calls if kind == "variable"}
    assert variables["ENVIRONMENT"] == "staging"
    assert variables["BOOTSTRAP_INDEX"] == "true"
    assert variables["PORT"] == "8000"
    # Secrets are never sent from here.
    assert not any("SECRET" in key or "KEY" in key for key in variables)

    readme = next(c for kind, c in api.calls if kind == "upload_file")
    assert readme["path_in_repo"] == "README.md"
    body = readme["path_or_fileobj"].decode("utf-8")
    assert body.startswith("---\n")
    assert "sdk: docker" in body
    assert "app_port: 8000" in body
    assert "github.com/me/nexusrag" in body

    folder = next(c for kind, c in api.calls if kind == "upload_folder")
    for pattern in (".env", "storage/*", "tests/*", "eval/*", "README.md"):
        assert pattern in folder["ignore_patterns"]
    assert folder["commit_message"] == "Deploy abc1234"


def test_wait_until_running(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(deploy_space.time, "sleep", lambda _: None)
    api = FakeApi(["BUILDING", "RUNNING_APP_STARTING", "RUNNING"])
    assert deploy_space.wait_until_running(api, "me/s", timeout_s=60, interval_s=0) == "RUNNING"
    failing = FakeApi(["BUILDING", "BUILD_ERROR"])
    assert deploy_space.wait_until_running(failing, "me/s", timeout_s=60, interval_s=0) == (
        "BUILD_ERROR"
    )


def test_deploy_requires_a_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    with pytest.raises(SystemExit):
        deploy_space.main(["--space", "me/nexusrag"])


# --------------------------------------------------------------------------- smoke test


def test_healthcheck_expects_page_content(server: HTTPServer) -> None:
    url = f"http://127.0.0.1:{server.server_port}/"
    assert healthcheck.check(url, 5, expect="ok")[0]
    healthy, detail = healthcheck.check(url, 5, expect="NexusRAG")
    assert not healthy
    assert "missing" in detail


def test_healthcheck_retries_until_healthy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(healthcheck.time, "sleep", lambda _: None)
    attempts = []

    def flaky(url: str, timeout_s: float, expect: str | None = None) -> tuple[bool, str]:
        attempts.append(url)
        return (len(attempts) >= 3, f"attempt {len(attempts)}")

    monkeypatch.setattr(healthcheck, "check", flaky)
    healthy, _ = healthcheck.wait_for(
        "http://x/", timeout_s=1, expect=None, retries=5, interval_s=0
    )
    assert healthy
    assert len(attempts) == 3  # stops as soon as it is healthy
