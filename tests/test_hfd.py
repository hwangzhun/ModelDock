import asyncio
import hashlib
import json
import shlex
import sys
from pathlib import Path

import httpx
import psutil
import pytest

from autodl_tui import hfd
from autodl_tui.config import Config
from autodl_tui.downloads import Downloads, Model


@pytest.fixture
def downloads(tmp_path):
    return Downloads(Config(root=str(tmp_path)))


@pytest.fixture
def fake_hfd(tmp_path, monkeypatch):
    worker = tmp_path / "fake_hfd.py"
    worker.write_text('''import sys, os, json, time
from pathlib import Path
args = sys.argv[1:]
root = Path(args[args.index('--local-dir') + 1])
meta = root / '.hfd'
meta.mkdir(exist_ok=True)
Path(root / 'call.json').write_text(json.dumps({'args': args, 'endpoint': os.environ['HF_ENDPOINT'], 'token': os.environ.get('HF_TOKEN'), 'username': os.environ.get('HF_USERNAME')}))
path = 'IQ3_S/model.gguf'
dest = root / path
dest.parent.mkdir(exist_ok=True)
(meta / 'manifest').write_text('6\\t' + path + '\\n')
(meta / 'last_download_command').write_text('HF_TOKEN=secret')
(meta / 'aria2c_urls.txt').write_text('header=Authorization: Bearer secret')
mode = os.environ.get('FAKE_HFD_MODE', 'success')
if mode == 'fail':
 print('network failed', flush=True)
 sys.exit(7)
if mode == 'missing':
 sys.exit(0)
if mode == 'slow':
 import subprocess
 child = subprocess.Popen([sys.executable, '-c', 'import signal,time; signal.signal(signal.SIGINT,signal.SIG_IGN); signal.signal(signal.SIGTERM,signal.SIG_IGN); time.sleep(60)'])
 (root / 'child-pid').write_text(str(child.pid))
 dest.write_bytes(b'abc')
 Path(str(dest) + '.aria2').write_text('resume')
 (root / 'pid').write_text(str(os.getpid()))
 print('[ 50%] 0/1 files | 3B/6B | 1B/s | ETA 00:03', flush=True)
 time.sleep(60)
else:
 dest.write_bytes(b'abcdef')
 Path(str(dest) + '.aria2').unlink(missing_ok=True)
 print('[100%] 1/1 files | 6B/6B | 3B/s | ETA --:--', flush=True)
''')
    shim = tmp_path / "hfd.sh"
    shim.write_text(f"#!/usr/bin/env bash\nexec {shlex.quote(sys.executable)} {shlex.quote(str(worker))} \"$@\"\n")

    async def script(client, state):
        return shim

    monkeypatch.setattr(hfd, "script", script)
    return shim


def add(downloads, sha256="", filename=""):
    model = Model.from_url("https://huggingface.co/a/b/resolve/v1/IQ3_S/model.gguf", "gguf", filename, sha256)
    downloads.add([model])
    return downloads.jobs[0]


async def transfer(downloads, job, handler=None):
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler or (lambda r: pytest.fail("no HTTP model transfer")))) as client:
        await downloads.transfer(client, job)


async def test_hfd_dispatch_arguments_staging_and_atomic_promotion(downloads, fake_hfd, monkeypatch):
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
    root = downloads.config.state / "hfd" / job.id
    call = json.loads((root / "call.json").read_text())
    args = call["args"]
    assert args[0] == "a/b"
    assert args[args.index("--include") + 1] == "^IQ3_S/model.gguf$"
    assert args[args.index("--revision") + 1] == "v1"
    assert args[args.index("-x") + 1] == "8"
    assert args[args.index("-j") + 1] == "1"
    assert call["endpoint"] == "https://hf-mirror.com"
    assert call["token"] is None and call["username"] is None
    assert not (root / "IQ3_S/model.gguf").exists()
    assert not (root / ".hfd/last_download_command").exists()
    assert not (root / ".hfd/aria2c_urls.txt").exists()


async def test_official_auth_and_custom_mirror(downloads, fake_hfd, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_secret")
    downloads.config.hf_endpoint = "https://huggingface.co"
    job = add(downloads)

    def whoami(request):
        assert str(request.url) == "https://huggingface.co/api/whoami-v2"
        assert request.headers["Authorization"] == "Bearer hf_secret"
        return httpx.Response(200, json={"name": "user"})

    await transfer(downloads, job, whoami)
    call = json.loads((downloads.config.state / "hfd" / job.id / "call.json").read_text())
    assert call["token"] == "hf_secret" and call["username"] == "user"
    assert "hf_secret" not in (downloads.config.state / "hfd-downloads.log").read_text()


@pytest.mark.parametrize("mode,match", [("fail", "退出码 7"), ("missing", "完整文件")])
async def test_hfd_failure_never_promotes_target(downloads, fake_hfd, monkeypatch, mode, match):
    monkeypatch.setenv("FAKE_HFD_MODE", mode)
    job = add(downloads)
    with pytest.raises((RuntimeError, ValueError), match=match):
        await transfer(downloads, job)
    assert not downloads.target(job.model).exists()


async def test_hfd_hash_failure_refetches_on_retry(downloads, fake_hfd):
    job = add(downloads, "0" * 64)
    with pytest.raises(ValueError, match="SHA256"):
        await transfer(downloads, job)
    assert not downloads.target(job.model).exists()
    assert not (downloads.config.state / "hfd" / job.id / "IQ3_S/model.gguf").exists()


@pytest.mark.parametrize("cancel_task", [False, True])
async def test_hfd_pause_and_cancel_stop_process_and_preserve_resume(downloads, fake_hfd, monkeypatch, cancel_task):
    monkeypatch.setenv("FAKE_HFD_MODE", "slow")
    job = add(downloads)
    task = asyncio.create_task(downloads.run())
    root = downloads.config.state / "hfd" / job.id
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
    assert (root / "IQ3_S/model.gguf.aria2").exists()
    assert (root / "IQ3_S/model.gguf").read_bytes() == b"abc"
    assert not (root / ".hfd/aria2c_urls.txt").exists()
    monkeypatch.setenv("FAKE_HFD_MODE", "success")
    downloads.retry(job.id)
    await downloads.run()
    assert job.status == "完成"


def test_progress_and_hfd_path_validation():
    from autodl_tui.downloads import Job
    job = Job("a", Model("https://example.com/file", "file"))
    hfd.update_progress(job, "[ 50%] 0/1 files | 1.50GB/3.00GB | 12.00MB/s | ETA 02:05")
    assert (job.downloaded, job.total, job.speed) == (1500000000, 3000000000, 12000000)
    assert hfd.selection("https://example.com/file") is None
    for path in ("../escape", "a%2F..%2Fescape", "*.gguf", "a%0Aevil"):
        with pytest.raises(ValueError):
            hfd.selection("https://huggingface.co/a/b/resolve/main/" + path)


async def test_script_bootstrap_checksum_fallback_and_cache(downloads, monkeypatch):
    # Exercise fetching without executing the fetched bytes.
    data = b"#!/bin/bash\nexit 0\n"
    monkeypatch.setattr(hfd, "HFD_SHA256", hashlib.sha256(data).hexdigest())
    monkeypatch.setattr(hfd.shutil, "which", lambda name: "/bin/" + name)
    calls = []

    def handler(request):
        calls.append(str(request.url))
        return httpx.Response(200, content=b"wrong" if len(calls) == 1 else data)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        path = await hfd.script(client, downloads.config.state)
        assert path.read_bytes() == data
        assert await hfd.script(client, downloads.config.state) == path
    assert len(calls) == 2


async def test_missing_dependencies_actionable_error(downloads, monkeypatch):
    monkeypatch.setattr(hfd.shutil, "which", lambda name: None)
    async with httpx.AsyncClient() as client:
        with pytest.raises(ValueError, match="apt-get install -y curl aria2"):
            await hfd.script(client, downloads.config.state)


@pytest.mark.parametrize("threads", [0, 11, "8", True])
def test_hfd_thread_setting_validation(tmp_path, threads):
    with pytest.raises(ValueError, match="连接数"):
        Config(root=str(tmp_path), hfd_threads=threads).validate()
