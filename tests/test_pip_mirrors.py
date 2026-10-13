import io
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from autodl_tui import pip_mirrors
from autodl_tui.config import Config
from autodl_tui.service import Service


def test_fastest_available_mirror_selected_and_failures_logged(monkeypatch):
    outcomes = {
        "阿里云": pip_mirrors.Result("阿里云", pip_mirrors.MIRRORS[0][1], 0.8),
        "腾讯云": pip_mirrors.Result("腾讯云", pip_mirrors.MIRRORS[1][1], error="HTTP 403"),
        "清华": pip_mirrors.Result("清华", pip_mirrors.MIRRORS[2][1], 0.4),
    }
    barrier = threading.Barrier(len(outcomes), timeout=2)

    def probe(name, url):
        barrier.wait()  # all three probes must run concurrently
        return outcomes[name]

    monkeypatch.setattr(pip_mirrors, "probe", probe)
    messages = []
    assert pip_mirrors.select_mirror(messages.append) == pip_mirrors.MIRRORS[2][1]
    assert any("403" in message for message in messages)
    assert any("本次使用" in message and "清华" in message for message in messages)


def test_all_failed_stops_instead_of_selecting_dead_source(monkeypatch):
    monkeypatch.setattr(pip_mirrors, "probe", lambda name, url: pip_mirrors.Result(name, url, error="timeout"))
    with pytest.raises(RuntimeError, match="均不可用"):
        pip_mirrors.select_mirror(lambda line: None)


class Response(io.BytesIO):
    def __init__(self, data, url, status=200):
        super().__init__(data)
        self.url = url
        self.status = status


@pytest.mark.parametrize("payload,successful", [
    (b"PK\x03\x04" + b"x" * pip_mirrors.SAMPLE_BYTES, True),
    (b"<html>access denied</html>", False),
])
def test_probe_fetches_same_real_wheel_sample(monkeypatch, payload, successful):
    calls = []
    wheel = f"https://cdn.example/{pip_mirrors.SAMPLE_FILE}"

    def urlopen(request, timeout):
        calls.append(request)
        assert timeout == pip_mirrors.TIMEOUT
        if len(calls) == 1:
            assert request.full_url.endswith("/setuptools/")
            return Response(f'<a href="{wheel}#sha256=abc">wheel</a>'.encode(), request.full_url)
        assert request.full_url.startswith(wheel)
        assert request.get_header("Range") == f"bytes=0-{pip_mirrors.SAMPLE_BYTES - 1}"
        return Response(payload, wheel)

    monkeypatch.setattr(pip_mirrors, "urlopen", urlopen)
    result = pip_mirrors.probe("测试", "https://mirror.example/simple")
    assert bool(result.error) != successful
    assert len(calls) == 2
    if successful:
        assert result.seconds > 0


def test_probe_timeout_and_missing_sample(monkeypatch):
    def timeout(*args, **kwargs):
        raise TimeoutError("timed out")

    monkeypatch.setattr(pip_mirrors, "urlopen", timeout)
    assert "timed out" in pip_mirrors.probe("测试", "https://mirror.example/simple").error
    monkeypatch.setattr(pip_mirrors, "urlopen", lambda *args, **kwargs: Response(b"<html>blocked</html>", "https://mirror.example"))
    assert "缺少" in pip_mirrors.probe("测试", "https://mirror.example/simple").error


def test_read_deadline_bounds_slow_stream():
    with pytest.raises(TimeoutError):
        pip_mirrors.read_sample(Response(b"x", "https://mirror.example"), 1, time.monotonic() - 1)


async def test_service_manual_source_and_auto_selection_cache(tmp_path, monkeypatch):
    calls = []

    def select(log):
        calls.append(True)
        return pip_mirrors.MIRRORS[0][1]

    monkeypatch.setattr("autodl_tui.service.select_mirror", select)
    config = Config(root=str(tmp_path), pip_index="https://pypi.org/simple")
    service = Service(config)
    await service.select_pip_index()
    assert not calls
    assert service.environment()["PIP_INDEX_URL"] == config.pip_index
    config.pip_index = "auto"
    config.validate()
    await service.select_pip_index()
    await service.select_pip_index()
    assert len(calls) == 1
    assert service.pip_index == pip_mirrors.MIRRORS[0][1]
    assert service.environment()["PIP_INDEX_URL"] == service.pip_index
    assert service.environment()["PIP_EXTRA_INDEX_URL"] == ""
    assert config.pip_index == "auto"
    config.pip_index = "https://pypi.org/simple"
    assert service.pip_index == config.pip_index


@pytest.mark.parametrize("override", ["", "https://fixed.example/simple"])
def test_bootstrap_selects_before_pip_and_respects_manual_override(tmp_path, override):
    project = Path(__file__).resolve().parents[1]
    shutil.copy(project / "run.sh", tmp_path / "run.sh")
    module = tmp_path / "autodl_tui"
    module.mkdir()
    (module / "pip_mirrors.py").write_text(
        "import sys\nfrom pathlib import Path\n"
        "Path('probed').touch()\n"
        "print('测速完成', file=sys.stderr)\n"
        f"print({pip_mirrors.MIRRORS[1][1]!r})\n")
    python = tmp_path / ".venv/bin/python"
    python.parent.mkdir(parents=True)
    python.write_text(
        f"#!{sys.executable}\nimport sys, os, json\nfrom pathlib import Path\n"
        "if sys.argv[1:4] == ['-m', 'pip', 'install']:\n"
        " Path('pip-call.json').write_text(json.dumps({'args': sys.argv[1:], 'index': os.environ.get('PIP_INDEX_URL'), 'extra': os.environ.get('PIP_EXTRA_INDEX_URL')}))\n")
    python.chmod(0o755)
    env = {**os.environ, "MODELDOCK_NETWORK": "domestic", "MODELDOCK_PYTHON": sys.executable,
           "MODELDOCK_PIP_INDEX": override, "AI_TUI_PIP_INDEX": "", "AUTODL_PIP_INDEX": ""}
    result = subprocess.run(["bash", str(tmp_path / "run.sh"), "--install-only"], env=env,
                            capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    call = json.loads((tmp_path / "pip-call.json").read_text())
    expected = override or pip_mirrors.MIRRORS[1][1]
    assert call["index"] == expected
    assert call["args"][call["args"].index("--index-url") + 1] == expected
    assert call["extra"] == ""
    assert (tmp_path / "probed").exists() == (not override)
    if not override:
        assert "测速完成" in result.stderr
    assert (tmp_path / ".venv/.modeldock-installed").exists()
    (tmp_path / "pip-call.json").unlink()
    (tmp_path / "probed").unlink(missing_ok=True)
    cached = subprocess.run(["bash", str(tmp_path / "run.sh"), "--install-only"], env=env,
                            capture_output=True, text=True, timeout=10)
    assert cached.returncode == 0, cached.stderr
    assert not (tmp_path / "probed").exists()
    assert not (tmp_path / "pip-call.json").exists()
