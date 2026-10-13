import asyncio
import socket
import sys

import psutil
import pytest

from autodl_tui.config import Config, atomic_json, redact
from autodl_tui.service import Service, workspace_lock


@pytest.fixture
def service(tmp_path):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    config = Config(root=str(tmp_path), port=port, host="127.0.0.1")
    config.comfy.mkdir()
    config.interpreter.parent.mkdir(parents=True)
    config.interpreter.symlink_to(sys.executable)
    (config.comfy / "main.py").write_text("import time\nwhile True: time.sleep(1)\n")
    return Service(config)


async def test_start_reattach_and_stop(service):
    try:
        await service.start()
        process = service.managed_process()
        assert process
        second = Service(service.config)
        assert second.managed_process().pid == process.pid
        await second.start()
        assert second.managed_process().pid == process.pid
        await second.stop()
        assert not service.managed_process()
    finally:
        if service.child:
            if service.child.poll() is None:
                service.child.kill()
            service.child.wait()


async def test_stale_identity_does_not_kill_unrelated_process(service):
    process = psutil.Process()
    atomic_json(service.config.state / "service.json", {
        "pid": process.pid, "created": process.create_time() - 10, "command": process.cmdline(),
    })
    assert service.managed_process() is None
    with pytest.raises(RuntimeError, match="没有"):
        await service.stop()
    assert process.is_running()


async def test_occupied_port_is_preserved(service):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", service.config.port))
        sock.listen()
        with pytest.raises(RuntimeError, match="占用"):
            await service.start()
        assert not service.managed_process()


async def test_cancel_deployment_command_kills_child(service, tmp_path):
    pid_file = tmp_path / "command-pid"
    code = f"import os,time; open({str(pid_file)!r},'w').write(str(os.getpid())); time.sleep(60)"
    task = asyncio.create_task(service.command([sys.executable, "-c", code]))
    for _ in range(50):
        if pid_file.exists():
            break
        await asyncio.sleep(0.02)
    assert pid_file.exists()
    pid = int(pid_file.read_text())
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not psutil.pid_exists(pid)


def test_workspace_lock_and_model_config(service):
    with workspace_lock(service.config):
        with pytest.raises(RuntimeError):
            with workspace_lock(service.config):
                pass
    existing = service.config.comfy / "extra_model_paths.yaml"
    existing.write_text("user_config: {}")
    path = service.write_model_paths()
    assert str(service.config.models) in path.read_text()
    assert existing.read_text() == "user_config: {}"


def test_xet_package_names_survive_redaction(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("CIVITAI_TOKEN", raising=False)
    assert redact("pip install hf_xet hf_transfer hf_secret hf_xetABC") == (
        "pip install hf_xet hf_transfer hf_*** hf_***")
    monkeypatch.setenv("HF_TOKEN", "hf_xet")
    assert redact("Bearer hf_xet") == "Bearer ***"


def test_config_roundtrip_and_redaction(tmp_path, monkeypatch):
    config = Config(root=str(tmp_path))
    path = tmp_path / "config.json"
    config.save(path)
    assert Config.load(path) == config
    assert "TOKEN" not in path.read_text()
    assert path.stat().st_mode & 0o777 == 0o600
    monkeypatch.setenv("CIVITAI_TOKEN", "private-value")
    assert redact("https://x?token=private-value&x=2 hf_secret") == "https://x?token=***&x=2 hf_***"


@pytest.mark.parametrize("args", ["--port 8188", "--listen=0.0.0.0", "--extra-model-paths-config other.yaml"])
def test_reserved_arguments_rejected(tmp_path, args):
    with pytest.raises(ValueError, match="不能覆盖"):
        Config(root=str(tmp_path), extra_args=args).validate()


async def test_deploy_reuses_existing_source_without_git_pull(service, monkeypatch):
    (service.config.comfy / "requirements.txt").write_text("")
    commands = []

    async def command(args, cwd=None):
        commands.append(args)

    monkeypatch.setattr(service, "command", command)
    monkeypatch.setattr("autodl_tui.service.select_mirror", lambda log: "https://mirrors.aliyun.com/pypi/simple")
    await service.deploy()
    assert not any(args[0] == "git" for args in commands)
    assert any("-r" in args for args in commands)
    requirements = next(args for args in commands if "-r" in args)
    assert requirements[requirements.index("--index-url") + 1] == "https://mirrors.aliyun.com/pypi/simple"
    assert service.environment()["PIP_INDEX_URL"] == "https://mirrors.aliyun.com/pypi/simple"
    assert (service.config.state / "model-paths.yaml").exists()
    assert not (service.config.state / "service.json").exists()
