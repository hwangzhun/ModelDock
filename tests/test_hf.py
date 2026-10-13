import asyncio
import hashlib
import json
import shlex
import sys
from pathlib import Path

import httpx
import psutil
import pytest

from autodl_tui import hf
from autodl_tui.config import Config
from autodl_tui.downloads import Downloads, Model


@pytest.fixture
def downloads(tmp_path):
    return Downloads(Config(root=str(tmp_path)))


@pytest.fixture
def fake_hf(tmp_path, monkeypatch):
    worker = tmp_path / "fake_hf.py"
    worker.write_text('''import sys, os, json, time, subprocess
from pathlib import Path
args = sys.argv[1:]
root = Path(args[args.index('--local-dir') + 1])
meta = root / '.cache' / 'huggingface' / 'download'
meta.mkdir(parents=True, exist_ok=True)
Path(root / 'call.json').write_text(json.dumps({'args': args, 'endpoint': os.environ['HF_ENDPOINT'], 'token': os.environ.get('HF_TOKEN'), 'username': os.environ.get('HF_USERNAME'), 'high_performance': os.environ.get('HF_XET_HIGH_PERFORMANCE'), 'disable_xet': os.environ.get('HF_HUB_DISABLE_XET')}))
path = 'IQ3_S/model.gguf'
dest = root / path
dest.parent.mkdir(exist_ok=True)
mode = os.environ.get('FAKE_HF_MODE', 'success')
if mode == 'fail':
 print('network failed', flush=True)
 sys.exit(7)
if mode == 'too-large':
 print('Error: Invalid value. The file is too large to be downloaded using the regular download method. Install `hf_xet` with `pip install hf_xet` for xet-powered downloads.', flush=True)
 print('Hint: set HF_DEBUG=1 as environment variable for full traceback.', flush=True)
 sys.exit(1)
if mode == 'missing':
 sys.exit(0)
if mode == 'slow':
 child = subprocess.Popen([sys.executable, '-c', 'import signal,time; signal.signal(signal.SIGINT,signal.SIG_IGN); signal.signal(signal.SIGTERM,signal.SIG_IGN); time.sleep(60)'])
 (root / 'child-pid').write_text(str(child.pid))
 dest.write_bytes(b'abc')
 (meta / 'model.gguf.incomplete').write_text('resume')
 (root / 'pid').write_text(str(os.getpid()))
 print('model: 50%|#####| 3/6 [00:03<00:03, 1B/s]', flush=True)
 time.sleep(60)
else:
 dest.write_bytes(b'abcdef')
 (meta / 'model.gguf.incomplete').unlink(missing_ok=True)
 print('model: 100%|#####| 6B/6B [00:02<00:00, 3B/s]', flush=True)
''')
    shim = tmp_path / "hf.sh"
    shim.write_text(f"#!/usr/bin/env bash\nexec {shlex.quote(sys.executable)} {shlex.quote(str(worker))} \"$@\"\n")

    shim.chmod(0o700)
    monkeypatch.setattr(hf, "command", lambda: str(shim))
    return shim


def add(downloads, sha256="", filename=""):
    model = Model.from_url("https://huggingface.co/a/b/resolve/v1/IQ3_S/model.gguf", "gguf", filename, sha256)
    downloads.add([model])
    return downloads.jobs[0]


async def transfer(downloads, job, handler=None):
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler or (lambda r: httpx.Response(200, headers={"content-length": "6"}) if r.method == "HEAD" else pytest.fail("no HTTP model transfer")))) as client:
        await downloads.transfer(client, job)


async def test_hf_dispatch_arguments_staging_and_atomic_promotion(downloads, fake_hf, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_secret")
    monkeypatch.setenv("HUGGING_FACE_HUB_TOKEN", "hf_legacy")
    monkeypatch.setenv("HF_USERNAME", "private-user")
    job = add(downloads, hashlib.sha256(b"abcdef").hexdigest(), "renamed.gguf")
    old = Path(str(downloads.target(job.model)) + ".part")
    old.parent.mkdir(parents=True)
    old.write_bytes(b"old")
    await transfer(downloads, job)
    assert downloads.target(job.model).read_bytes() == b"abcdef"
    assert old.read_bytes() == b"old"
    assert job.status == "完成" and job.total == job.downloaded == 6
    root = downloads.config.state / "hf" / job.id
    call = json.loads((root / "call.json").read_text())
    args = call["args"]
    assert args[:3] == ["download", "a/b", "IQ3_S/model.gguf"]
    assert args[args.index("--revision") + 1] == "v1"
    assert call["endpoint"] == "https://hf-mirror.com"
    assert call["token"] is None and call["username"] is None
    assert not (root / "IQ3_S/model.gguf").exists()


@pytest.mark.parametrize("origin", ["huggingface.co", "hf-mirror.com"])
async def test_official_auth_and_custom_mirror(downloads, fake_hf, monkeypatch, origin):
    monkeypatch.setenv("HF_TOKEN", "hf_secret")
    downloads.config.hf_endpoint = "https://huggingface.co"
    job = add(downloads)
    job.model.url = job.model.url.replace("huggingface.co", origin)

    def whoami(request):
        assert request.method == "HEAD"
        assert request.url.host == "huggingface.co"
        assert request.headers["Authorization"] == "Bearer hf_secret"
        return httpx.Response(200, headers={"content-length": "6"})

    await transfer(downloads, job, whoami)
    call = json.loads((downloads.config.state / "hf" / job.id / "call.json").read_text())
    assert call["endpoint"] == "https://huggingface.co"
    assert call["token"] == "hf_secret" and call["username"] is None
    assert "hf_secret" not in (downloads.config.state / "hf-downloads.log").read_text()


@pytest.mark.parametrize("mode,match", [("fail", "退出码 7"), ("missing", "完整文件")])
async def test_hf_failure_never_promotes_target(downloads, fake_hf, monkeypatch, mode, match):
    monkeypatch.setenv("FAKE_HF_MODE", mode)
    job = add(downloads)
    with pytest.raises((RuntimeError, ValueError), match=match):
        await transfer(downloads, job)
    assert not downloads.target(job.model).exists()


@pytest.mark.parametrize("endpoint", ["https://hf-mirror.com", "https://huggingface.co"])
@pytest.mark.parametrize("disable_xet", ["0", "1", "TRUE"])
async def test_oversized_http_error_explains_xet_recovery(downloads, fake_hf, monkeypatch,
                                                       endpoint, disable_xet):
    monkeypatch.setenv("FAKE_HF_MODE", "too-large")
    monkeypatch.setenv("HF_HUB_DISABLE_XET", disable_xet)
    downloads.config.hf_endpoint = endpoint
    job = add(downloads)
    with pytest.raises(RuntimeError, match="python -m pip install -U hf_xet") as error:
        await transfer(downloads, job)
    assert ("取消此环境变量" in str(error.value)) == (disable_xet != "0")
    assert ("HF 站点改为" in str(error.value)) == (endpoint != "https://huggingface.co")
    logs = (downloads.config.state / "hf-downloads.log").read_text()
    assert "pip install hf_xet" in logs
    assert "普通 HTTP 下载" in logs
    assert not downloads.target(job.model).exists()


async def test_hf_hash_failure_refetches_on_retry(downloads, fake_hf):
    job = add(downloads, "0" * 64)
    with pytest.raises(ValueError, match="SHA256"):
        await transfer(downloads, job)
    assert not downloads.target(job.model).exists()
    assert not (downloads.config.state / "hf" / job.id / "IQ3_S/model.gguf").exists()


@pytest.mark.parametrize("cancel_task", [False, True])
async def test_hf_pause_and_cancel_stop_process_and_preserve_resume(downloads, fake_hf, monkeypatch, cancel_task):
    async def head(client, url, **kwargs):
        return httpx.Response(200, headers={"content-length": "6"}, request=httpx.Request("HEAD", url))

    monkeypatch.setattr(httpx.AsyncClient, "head", head)
    monkeypatch.setenv("FAKE_HF_MODE", "slow")
    job = add(downloads)
    task = asyncio.create_task(downloads.run())
    root = downloads.config.state / "hf" / job.id
    for _ in range(100):
        if (root / "pid").exists():
            break
        await asyncio.sleep(.02)
    pid = int((root / "pid").read_text())
    if cancel_task:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    else:
        downloads.pause(job.id)
        await asyncio.wait_for(task, 10)
    assert not psutil.pid_exists(pid)
    child_pid = int((root / "child-pid").read_text())
    try:
        child = psutil.Process(child_pid)
        assert child.status() == psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        pass
    assert job.status == "暂停"
    assert not downloads.target(job.model).exists()
    assert (root / ".cache/huggingface/download/model.gguf.incomplete").exists()
    assert (root / "IQ3_S/model.gguf").read_bytes() == b"abc"
    monkeypatch.setenv("FAKE_HF_MODE", "success")
    downloads.retry(job.id)
    await downloads.run()
    assert job.status == "完成"


def test_progress_and_hf_path_validation():
    from autodl_tui.downloads import Job
    job = Job("a", Model("https://example.com/file", "file"))
    hf.update_progress(job, "model: 50%|#####| 1.50GB/3.00GB [00:01<02:05, 12.00MB/s]")
    assert (job.downloaded, job.total, job.speed) == (1500000000, 3000000000, 12000000)
    assert hf.selection("https://example.com/file") is None
    for path in ("../escape", "a%2F..%2Fescape", "*.gguf", "a%0Aevil"):
        with pytest.raises(ValueError):
            hf.selection("https://huggingface.co/a/b/resolve/main/" + path)


async def test_missing_dependencies_actionable_error(downloads, monkeypatch):
    monkeypatch.setattr(hf.shutil, "which", lambda name: None)
    monkeypatch.setattr(hf.sys, "executable", "/missing/python")
    with pytest.raises(ValueError, match="pip install"):
        hf.command()


def test_legacy_connection_setting_ignored(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"root": str(tmp_path), "hfd_threads": 8}))
    config = Config.load(path)
    config.save(path)
    assert "hfd_threads" not in json.loads(path.read_text())


async def test_size_mismatch_never_promotes(downloads, fake_hf):
    job = add(downloads)
    with pytest.raises(ValueError, match="完整文件"):
        await transfer(downloads, job, lambda r: httpx.Response(200, headers={"content-length": "7"}))
    assert not downloads.target(job.model).exists()


async def test_linked_size_overrides_redirect_body_size(downloads, fake_hf):
    job = add(downloads)
    await transfer(downloads, job, lambda r: httpx.Response(
        302, headers={"x-linked-size": "6", "content-length": "0", "location": "https://example.com/blob"}))
    assert job.status == "完成" and job.total == 6


@pytest.mark.parametrize("override,expected", [(None, "1"), ("0", "0")])
async def test_xet_high_performance_default_and_explicit_override(downloads, fake_hf, monkeypatch, override, expected):
    if override is None:
        monkeypatch.delenv("HF_XET_HIGH_PERFORMANCE", raising=False)
    else:
        monkeypatch.setenv("HF_XET_HIGH_PERFORMANCE", override)
    monkeypatch.setenv("HF_HUB_DISABLE_XET", "1")
    job = add(downloads)
    await transfer(downloads, job, lambda r: httpx.Response(
        302, headers={"x-linked-size": "6", "x-xet-hash": "abc", "location": "https://example.com/blob"}))
    call = json.loads((downloads.config.state / "hf" / job.id / "call.json").read_text())
    assert call["high_performance"] == expected and call["disable_xet"] == "1"
    logs = (downloads.config.state / "hf-downloads.log").read_text()
    assert "服务端 Xet 元数据=有" in logs
    assert f"HF_XET_HIGH_PERFORMANCE={expected}" in logs


async def test_redirect_chain_preserves_xet_metadata(downloads, fake_hf):
    job = add(downloads)

    def handler(request):
        if request.url.host == "hf-mirror.com":
            return httpx.Response(308, headers={"location": str(request.url).replace("hf-mirror.com", "huggingface.co")})
        if request.url.host == "huggingface.co":
            return httpx.Response(302, headers={"location": "https://cdn.example.com/blob", "x-linked-size": "6",
                                                "x-xet-hash": "abc", "content-length": "123"})
        return httpx.Response(200, headers={"content-length": "6"})

    await transfer(downloads, job, handler)
    logs = (downloads.config.state / "hf-downloads.log").read_text()
    assert "服务端 Xet 元数据=有" in logs
    assert "hf-mirror.com → huggingface.co → cdn.example.com" in logs
    assert job.status == "完成" and job.total == 6
