import asyncio
import json
import socket
import sys

import httpx
import pytest
from textual.widgets import Input, Select, Static

from autodl_tui.adapters import TOOLS
from autodl_tui.app import AIApp
from autodl_tui.config import Config
from autodl_tui.downloads import Downloads, Model
from autodl_tui.service import Service


def host(**overrides):
    return {"system": "Linux", "arch": "x86_64", "writable": True, "git": "/bin/git",
            "cmake": "/bin/cmake", "compiler": "/bin/c++", "nvidia": False, "amd": False,
            "cuda": None, "hip": None, **overrides}


def test_legacy_config_backup_and_paths(tmp_path):
    path = tmp_path / "config.json"
    original = {"root": str(tmp_path), "port": 6008, "extra_args": "--lowvram"}
    path.write_text(json.dumps(original))
    c = Config.load(path)
    assert c.version == 2 and c.comfy == tmp_path / "ComfyUI"
    assert c.port == 6008 and c.extra_args == "--lowvram"
    c.save(path)
    assert json.loads(path.with_suffix('.json.v1.bak').read_text()) == original
    assert c.for_tool('forge').extra_args == ''
    assert c.for_tool('forge').comfy == tmp_path / 'Forge'
    assert c.for_tool('forge').service_state != c.service_state


def test_model_manifest_gguf_and_existing_v1(tmp_path):
    downloads = Downloads(Config(root=str(tmp_path)))
    downloads.add([Model('https://example.com/a.gguf', 'a.gguf', 'gguf')])
    path = tmp_path / 'manifest.json'
    downloads.export_manifest(path)
    assert json.loads(path.read_text())['version'] == 1
    assert downloads.import_manifest(path) == 0
    assert downloads.target(downloads.jobs[0].model) == tmp_path / 'models/gguf/a.gguf'


@pytest.mark.parametrize('backend,flags', [('cpu', ('OFF', 'OFF')), ('cuda', ('ON', 'OFF')),
                                          ('hip', ('OFF', 'ON'))])
async def test_llama_build_strategy(tmp_path, monkeypatch, backend, flags):
    monkeypatch.setattr('autodl_tui.adapters.capabilities', lambda _: host(cuda='/nvcc', hip='/hipcc'))
    service = Service(Config(root=str(tmp_path), tools={'llamacpp': {'backend': backend}}), 'llamacpp')
    service.config.comfy.mkdir()
    (service.config.comfy / 'CMakeLists.txt').touch()
    commands = []

    async def command(args, cwd=None):
        commands.append(args)

    monkeypatch.setattr(service, 'command', command)
    await service.adapter.deploy(service)
    assert f'-DGGML_CUDA={flags[0]}' in commands[0]
    assert f'-DGGML_HIP={flags[1]}' in commands[0]
    assert 'llama-server' in commands[1]


def test_backend_auto_and_explicit_missing(tmp_path, monkeypatch):
    adapter = TOOLS['llamacpp']
    c = Config(root=str(tmp_path))
    monkeypatch.setattr('autodl_tui.adapters.capabilities', lambda _: host())
    assert adapter.backend(c) == 'cpu'
    c.backend = 'hip'
    with pytest.raises(ValueError, match='hip'):
        adapter.backend(c)
    c.backend = 'auto'
    monkeypatch.setattr('autodl_tui.adapters.capabilities', lambda _: host(hip='/hipcc'))
    assert adapter.backend(c) == 'hip'


@pytest.mark.parametrize('tool', ['forge', 'a1111'])
async def test_webui_independent_environment_and_cpu_launch(tmp_path, monkeypatch, tool):
    monkeypatch.setattr('autodl_tui.adapters.capabilities', lambda _: host())
    monkeypatch.setattr('autodl_tui.adapters.python_version', lambda _: (3, 10))
    service = Service(Config(root=str(tmp_path)), tool)
    c = service.config
    c.comfy.mkdir()
    for file in service.adapter.required:
        (c.comfy / file).touch()
    commands = []

    async def command(args, cwd=None):
        commands.append(args)

    async def torch():
        pass

    monkeypatch.setattr(service, 'command', command)
    monkeypatch.setattr(service, 'prepare_torch', torch)
    monkeypatch.setattr(service, 'verify_torch', torch)
    await service.adapter.deploy(service)
    assert commands[0] == [c.python, '-m', 'venv', str(c.comfy / '.venv')]
    assert '--exit' in commands[1] and '--skip-torch-cuda-test' in commands[1]
    args = service.adapter.launch(service)
    assert '--skip-install' in args and '--use-cpu' in args
    assert args[args.index('--ckpt-dir') + 1] == str(c.models / 'checkpoints')
    assert '--xformers' not in args


def test_webui_python_check(tmp_path, monkeypatch):
    monkeypatch.setattr('autodl_tui.adapters.capabilities', lambda _: host())
    monkeypatch.setattr('autodl_tui.adapters.python_version', lambda _: (3, 12))
    assert 'Python 3.10' in TOOLS['forge'].check(Config(root=str(tmp_path)))[0]


def free_port():
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        return sock.getsockname()[1]


async def test_two_live_services_reattach_and_isolated_stop(tmp_path):
    code = '''import argparse
from http.server import BaseHTTPRequestHandler, HTTPServer
p = argparse.ArgumentParser(); p.add_argument('--port', type=int)
a, _ = p.parse_known_args()
class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200); self.end_headers(); self.wfile.write(b'{"system": {}, "status": "ok"}')
HTTPServer(('127.0.0.1', a.port), Handler).serve_forever()
'''
    c = Config(root=str(tmp_path), host='127.0.0.1', port=free_port(),
               tools={'forge': {'port': free_port()}})
    services = [Service(c), Service(c, 'forge')]
    for service in services:
        conf = service.config
        conf.comfy.mkdir()
        conf.interpreter.parent.mkdir(parents=True)
        conf.interpreter.symlink_to(sys.executable)
        (conf.comfy / service.adapter.entry).write_text(code)
    try:
        for service in services:
            await service.start()
        assert all(await asyncio.gather(*(service.health() for service in services)))
        assert Service(c).managed_process() and Service(c, 'forge').managed_process()
        await services[0].stop()
        assert await services[1].health()
        assert services[1].managed_process()
        await services[1].stop()
    finally:
        for service in services:
            if service.managed_process():
                await service.stop()


@pytest.mark.parametrize('dimensions', [(80, 24), (120, 50)])
async def test_tool_switch_gguf_and_independent_save(tmp_path, dimensions):
    c = Config(root=str(tmp_path))
    model = c.models / 'gguf/a.gguf'
    model.parent.mkdir(parents=True)
    model.touch()
    app = AIApp(c, tmp_path / 'config.json')
    async with app.run_test(size=dimensions) as pilot:
        await pilot.pause()
        app.query_one('#tool-select', Select).value = 'llamacpp'
        await pilot.pause()
        assert app.service.adapter.id == 'llamacpp'
        assert app.query_one('#config-port', Input).value == '8080'
        app.query_one('#gguf-select', Select).value = str(model)
        await pilot.pause()
        assert app.query_one('#config-model', Input).value == str(model)
        app.query_one('#config-port', Input).value = '8090'
        app.save_settings()
        assert Config.load(tmp_path / 'config.json').for_tool('llamacpp').port == 8090
        app.query_one('#tool-select', Select).value = 'comfyui'
        await pilot.pause()
        assert app.query_one('#config-port', Input).value == '6006'
        assert '6006' in str(app.query_one('#access-hint', Static).render())


async def test_health_contracts(tmp_path, monkeypatch):
    real = httpx.AsyncClient
    def client(**kwargs):
        return real(transport=httpx.MockTransport(lambda req: httpx.Response(
            200, json={'status': 'loading'})), **kwargs)
    monkeypatch.setattr('autodl_tui.service.httpx.AsyncClient', client)
    assert not await Service(Config(root=str(tmp_path)), 'llamacpp').health()
    assert not await Service(Config(root=str(tmp_path))).health()


def test_invalid_tool_fields_rejected(tmp_path):
    with pytest.raises(ValueError, match='无效字段'):
        Config(root=str(tmp_path), tools={'forge': {'root': '/bad'}}).for_tool('forge')


@pytest.mark.parametrize('network', ['domestic', 'official'])
def test_first_run_and_legacy_migration(tmp_path, monkeypatch, network):
    from autodl_tui import __main__ as cli
    destination = tmp_path / 'ai/config.json'
    legacy = tmp_path / 'old/config.json'
    captured = []

    class App:
        def __init__(self, config, path):
            captured.append(config)
        def run(self):
            pass

    monkeypatch.setattr('autodl_tui.app.AutoDLApp', App)
    monkeypatch.setattr(cli, 'config_path', lambda: destination)
    monkeypatch.setattr(cli, 'legacy_config_path', lambda: legacy)
    monkeypatch.setattr(sys, 'argv', ['ai-tui', '--root', str(tmp_path / 'work')])
    monkeypatch.setenv('AI_TUI_NETWORK', network)
    cli.main()
    assert captured[-1].network == network
    assert captured[-1].for_tool('forge').repo.endswith('stable-diffusion-webui-forge.git')
    destination.unlink()
    legacy.parent.mkdir()
    old = {'root': str(tmp_path / 'old-work'), 'port': 6008, 'hf_endpoint': 'https://huggingface.co'}
    legacy.write_text(json.dumps(old))
    monkeypatch.setattr(sys, 'argv', ['ai-tui'])
    cli.main()
    assert captured[-1].port == 6008 and captured[-1].network == 'official'
    assert json.loads(legacy.with_suffix('.json.v1.bak').read_text()) == old
    assert Config.load(destination).root == old['root']


async def test_rocm_missing_wheel_never_installs_cuda(tmp_path, monkeypatch):
    monkeypatch.setattr('autodl_tui.service.capabilities', lambda _: host(amd=True))
    c = Config(root=str(tmp_path), backend='hip')
    c.interpreter.parent.mkdir(parents=True)
    c.interpreter.symlink_to(sys.executable)
    service = Service(c)
    commands = []
    async def command(args, cwd=None):
        commands.append(args)
    monkeypatch.setattr(service, 'command', command)
    with pytest.raises(RuntimeError, match='ROCm'):
        await service.prepare_torch()
    assert commands == []


async def test_managed_child_is_stopped(tmp_path):
    import psutil
    c = Config(root=str(tmp_path), port=free_port())
    c.comfy.mkdir()
    c.interpreter.parent.mkdir(parents=True)
    c.interpreter.symlink_to(sys.executable)
    pid_file = tmp_path / 'child-pid'
    (c.comfy / 'main.py').write_text(
        f"import subprocess,sys,time\np = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
        f"open({str(pid_file)!r}, 'w').write(str(p.pid))\ntime.sleep(60)\n")
    service = Service(c)
    try:
        await service.start()
        child = psutil.Process(int(pid_file.read_text()))
        await service.stop()
        assert not child.is_running() or child.status() == psutil.STATUS_ZOMBIE
    finally:
        if service.managed_process():
            await service.stop()


def test_tool_runtime_settings_do_not_leak(tmp_path):
    c = Config(root=str(tmp_path), backend='hip', model='/model.gguf', extra_args='--lowvram', context=8192)
    forge = c.for_tool('forge')
    assert forge.backend == 'auto' and forge.model == '' and forge.context == 4096
    assert forge.extra_args == ''


async def test_cancel_command_cleans_descendant(tmp_path):
    import psutil
    service = Service(Config(root=str(tmp_path)))
    pid_file = tmp_path / 'command-child'
    code = ("import subprocess,sys,time; "
            "p=subprocess.Popen([sys.executable,'-c','import signal,time; "
            "signal.signal(signal.SIGTERM,signal.SIG_IGN); time.sleep(60)']); "
            f"open({str(pid_file)!r},'w').write(str(p.pid)); time.sleep(60)")
    task = asyncio.create_task(service.command([sys.executable, '-c', code]))
    for _ in range(100):
        if pid_file.exists():
            break
        await asyncio.sleep(.02)
    child = psutil.Process(int(pid_file.read_text()))
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    for _ in range(50):
        if not child.is_running() or child.status() == psutil.STATUS_ZOMBIE:
            break
        await asyncio.sleep(.02)
    assert not child.is_running() or child.status() == psutil.STATUS_ZOMBIE
