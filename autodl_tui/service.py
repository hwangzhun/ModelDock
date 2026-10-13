from __future__ import annotations

import asyncio
import fcntl
import json
import os
import shlex
import shutil
import signal
import socket
import subprocess
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from urllib.parse import urlsplit

import httpx
import psutil

from .adapters import TOOLS
from .config import CATEGORIES, Config, atomic_json, redact
from .environment import capabilities
from .pip_mirrors import select_mirror


@contextmanager
def workspace_lock(config: Config, name: str = "operation"):
    config.state.mkdir(parents=True, exist_ok=True)
    with (config.state / f"{name}.lock").open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("此工作目录已有任务运行，请勿重复操作") from None
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


class Service:
    def __init__(self, config: Config, tool_id: str = "comfyui"):
        self.config = config.for_tool(tool_id)
        self.adapter = TOOLS[tool_id]
        self.failed = ""
        self.child: subprocess.Popen | None = None
        self._selected_pip_index = ""

    @property
    def pip_index(self) -> str:
        return self._selected_pip_index if self.config.pip_index == "auto" else self.config.pip_index

    async def select_pip_index(self) -> None:
        if self.config.pip_index == "auto" and not self._selected_pip_index:
            self._selected_pip_index = await asyncio.to_thread(select_mirror, self.log)

    def environment(self) -> dict[str, str]:
        env = {**os.environ, "PYTHONUNBUFFERED": "1", "GIT_TERMINAL_PROMPT": "0",
               "HF_ENDPOINT": self.config.hf_endpoint}
        if self.config.pip_index:
            env["PIP_INDEX_URL"] = self.pip_index or "https://pypi.org/simple"
            env["PIP_EXTRA_INDEX_URL"] = ""
        # HF libraries may automatically use inherited credentials with HF_ENDPOINT.
        # Keep the TUI's token available for official downloads, but don't leak it to mirrors.
        if urlsplit(self.config.hf_endpoint).hostname != "huggingface.co":
            env.pop("HF_TOKEN", None)
            env.pop("HUGGING_FACE_HUB_TOKEN", None)
            env["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
        return env

    def log(self, line: str) -> None:
        self.config.service_state.mkdir(parents=True, exist_ok=True)
        path = self.config.service_state / "operations.log"
        if path.exists() and path.stat().st_size > 5 * 1024 * 1024:
            path.replace(path.with_suffix(".log.1"))
        with path.open("a", encoding="utf-8") as stream:
            stream.write(f"{datetime.now():%H:%M:%S}  {redact(line)}\n")

    async def command(self, args: list[str], cwd: Path | None = None) -> None:
        self.log("$ " + shlex.join(args))
        proc = await asyncio.create_subprocess_exec(
            *args, cwd=cwd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
            start_new_session=True, env=self.environment(),
            limit=1024 * 1024,
        )
        succeeded = False
        try:
            assert proc.stdout
            while line := await proc.stdout.readline():
                self.log(line.decode(errors="replace").rstrip())
            if await proc.wait():
                raise RuntimeError("命令执行失败，请在日志中查看原因")
            succeeded = True
        finally:
            if not succeeded:
                try:
                    os.killpg(proc.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    await asyncio.wait_for(proc.wait(), 5)
                except asyncio.TimeoutError:
                    pass
                # The root may exit before a compiler/downloader child. Clean up the
                # original session's remaining members as well on failure/cancellation.
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                await proc.wait()

    def managed_process(self) -> psutil.Process | None:
        # A PID alone is unsafe: Linux can reuse it for an unrelated process.
        if self.child is not None:
            self.child.poll()  # reap a child that has exited
        path = self.config.service_state / "service.json"
        try:
            record = json.loads(path.read_text())
            proc = psutil.Process(record["pid"])
            if abs(proc.create_time() - record["created"]) > 0.01:
                return None
            if proc.status() == psutil.STATUS_ZOMBIE or proc.cmdline() != record["command"]:
                return None
            if str(self.config.comfy / self.adapter.entry) not in proc.cmdline():
                return None
            return proc
        except (OSError, ValueError, TypeError, KeyError, psutil.Error):
            return None

    async def health(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=1, trust_env=False) as client:
                response = await client.get(f"http://127.0.0.1:{self.config.port}{self.adapter.health_path}")
                if self.adapter.id == "comfyui":
                    return response.status_code == 200 and isinstance(response.json().get("system"), dict)
                if self.adapter.id == "llamacpp":
                    return response.status_code == 200 and response.json().get("status") == "ok"
                return response.status_code == 200
        except (httpx.HTTPError, ValueError, AttributeError):
            return False

    def write_model_paths(self) -> Path:
        for category in CATEGORIES:
            if category == "gguf":
                continue
            (self.config.models / category).mkdir(parents=True, exist_ok=True)
        # JSON-quoted values are also valid YAML scalars. Do not replace user configuration.
        lines = ["autodl_tui:", f"  base_path: {json.dumps(str(self.config.models))}"]
        lines.extend(f"  {category}: {category}" for category in CATEGORIES if category != "gguf")
        path = self.config.service_state / "model-paths.yaml"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path

    async def clone_comfy(self) -> None:
        await self.clone_source()

    async def clone_source(self) -> None:
        config = self.config
        if config.comfy.exists():
            if not all((config.comfy / file).is_file() for file in self.adapter.required):
                raise RuntimeError("安装目录已存在但源码不完整；不会覆盖用户文件")
            return
        config.comfy.parent.mkdir(parents=True, exist_ok=True)
        staging = config.comfy.parent / f".{config.comfy.name}-cloning"
        sources = list(dict.fromkeys(source for source in (config.repo, config.repo_fallback) if source))
        for source in sources:
            # Only discard our own incomplete staging; existing ComfyUI is never replaced.
            if staging.exists():
                shutil.rmtree(staging)
            self.log(f"使用 {self.adapter.name} 下载源：{source}")
            try:
                await asyncio.wait_for(self.command([
                    "git", "-c", "http.lowSpeedLimit=1024", "-c", "http.lowSpeedTime=60",
                    "clone", "--depth", "1", source, str(staging),
                ]), timeout=600)
                if not all((staging / file).is_file() for file in self.adapter.required):
                    raise RuntimeError("镜像返回的仓库缺少工具入口或依赖清单")
            except (RuntimeError, asyncio.TimeoutError) as error:
                self.log(f"当前源失败：{error or '下载超时'}；尝试下一个已配置的镜像")
                continue
            staging.rename(config.comfy)
            return
        raise RuntimeError("工具下载源均失败，请在设置中更换 Git 镜像地址后重新部署；详情见操作日志")

    async def deploy(self) -> None:
        self.config.validate()
        with workspace_lock(self.config):
            if self.managed_process() or await self.health():
                raise RuntimeError("请先停止当前工具，再部署或修复")
            problems = await asyncio.to_thread(self.adapter.check, self.config)
            if problems:
                raise RuntimeError("\n".join(problems))
            self.config.base.mkdir(parents=True, exist_ok=True)
            if shutil.disk_usage(self.config.base).free < 4 * 1024**3:
                raise RuntimeError("剩余空间不足 4 GiB")
            self._selected_pip_index = ""
            await self.adapter.deploy(self)
            self.failed = ""

    @property
    def runtime_log(self) -> str:
        return "comfyui.log" if self.adapter.id == "comfyui" else "runtime.log"

    @property
    def installed(self) -> bool:
        c = self.config
        return (c.comfy / self.adapter.entry).exists() and (
            self.adapter.id == "llamacpp" or c.interpreter.exists())

    def torch_probe(self, backend: str) -> str:
        code = "import torch, torchvision; " + {
            "cpu": "assert torch.version.hip is None and torch.version.cuda is None",
            "hip": "assert torch.version.hip and torch.cuda.is_available()",
            "cuda": "assert torch.version.cuda and torch.cuda.is_available()",
        }[backend]
        versions = {"forge": ("2.3.1", "0.18.1"), "a1111": ("2.1.2", "0.16.2")}
        if self.adapter.id in versions:
            torch, vision = versions[self.adapter.id]
            code += f"; assert torch.__version__.split('+')[0] == {torch!r}"
            code += f"; assert torchvision.__version__.split('+')[0] == {vision!r}"
        return code

    async def verify_torch(self) -> None:
        await self.command([str(self.config.interpreter), "-c",
                            self.torch_probe(self.adapter.runtime_backend(self.config))])

    async def prepare_torch(self) -> None:
        c = self.config
        host = capabilities(c.base)
        backend = c.backend
        if backend == "auto":
            backend = "hip" if host["amd"] else ("cuda" if host["nvidia"] else "cpu")
        probe_code = self.torch_probe(backend)
        proc = await asyncio.create_subprocess_exec(str(c.interpreter), "-c", probe_code,
                                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            available = await asyncio.wait_for(proc.wait(), 60) == 0
        finally:
            if proc.returncode is None:
                proc.kill()
                await proc.wait()
        if available and not c.torch_index:
            self.log("复用经后端检查的 PyTorch")
            return
        if backend == "hip" and not c.torch_index:
            raise RuntimeError("请提供与 GPU / ROCm / 工具版本兼容的 PyTorch wheel 源，或使用已有 ROCm Python 环境")
        packages = ["torch", "torchvision", "torchaudio"]
        if self.adapter.id == "forge":
            packages = ["torch==2.3.1", "torchvision==0.18.1"]
        elif self.adapter.id == "a1111":
            packages = ["torch==2.1.2", "torchvision==0.16.2"]
        index = c.torch_index or ("https://download.pytorch.org/whl/cpu" if backend == "cpu" else
                                  "https://download.pytorch.org/whl/cu121")
        await self.command([str(c.interpreter), "-m", "pip", "install", "--index-url", index, *packages])
        await self.command([str(c.interpreter), "-c", probe_code])

    async def deploy_comfy(self) -> None:
        config = self.config
        config.validate()
        if self.managed_process() or await self.health():
            raise RuntimeError("请先停止 ComfyUI，再部署或修复依赖")
        if not shutil.which("git"):
            raise RuntimeError("缺少 git，请先安装 git")
        config.base.mkdir(parents=True, exist_ok=True)
        if shutil.disk_usage(config.base).free < 4 * 1024**3:
            raise RuntimeError("数据盘剩余空间不足 4 GiB，请先清理空间")
        if not config.comfy.exists():
            await self.clone_comfy()
        if not (config.comfy / "main.py").is_file() or not (config.comfy / "requirements.txt").is_file():
            raise RuntimeError("ComfyUI 目录已存在但源码不完整，请检查目录；不会覆盖现有文件")
        if not config.interpreter.exists():
            await self.command([config.python, "-m", "venv", "--system-site-packages", str(config.comfy / ".venv")])
        python = str(config.interpreter)
        await self.select_pip_index()
        pip = [python, "-m", "pip", "install"]
        index = ["--index-url", self.pip_index] if self.pip_index else []
        await self.command([*pip, *index, "--upgrade", "pip"])
        await self.prepare_torch()
        await self.command([*pip, *index, "-r", str(config.comfy / "requirements.txt")], cwd=config.comfy)
        self.write_model_paths()
        await self.verify_torch()
        await self.command([python, "-c", "import torch; print('PyTorch:', torch.__version__, 'CUDA:', torch.cuda.is_available())"])
        self.log("部署完成。可启动 ComfyUI；CUDA 为 False 时请检查镜像、驱动与 PyTorch。")

    async def start(self) -> None:
        config = self.config
        with workspace_lock(config):
            if self.managed_process():
                self.log(f"{self.adapter.name} 已在运行")
                return
            if not self.installed:
                raise RuntimeError(f"请先点击「部署 / 修复」安装 {self.adapter.name}")
            with socket.socket() as sock:
                try:
                    sock.bind((config.host, config.port))
                except OSError:
                    raise RuntimeError(f"端口 {config.port} 已被占用；请更换端口或停止原服务") from None
            args = self.adapter.launch(self)
            config.service_state.mkdir(parents=True, exist_ok=True)
            self.failed = ""
            log_path = config.service_state / self.runtime_log
            if log_path.exists() and log_path.stat().st_size > 10 * 1024 * 1024:
                log_path.replace(log_path.with_suffix(".log.1"))
            with log_path.open("ab", buffering=0) as output:
                self.child = subprocess.Popen(args, cwd=config.comfy, stdin=subprocess.DEVNULL,
                                              stdout=output, stderr=output, start_new_session=True,
                                              env=self.environment())
            try:
                proc = psutil.Process(self.child.pid)
                atomic_json(config.service_state / "service.json", {
                    "pid": proc.pid, "created": proc.create_time(), "command": args,
                })
            except Exception:
                os.killpg(self.child.pid, signal.SIGTERM)
                await asyncio.to_thread(self.child.wait, timeout=5)
                raise
            self.log(f"启动 {self.adapter.name}，PID {proc.pid}，端口 {config.port}")
            await asyncio.sleep(1)
            if not self.managed_process():
                self.failed = "启动后退出，请查看运行日志"
                raise RuntimeError(self.failed)

    async def stop(self) -> None:
        with workspace_lock(self.config):
            proc = self.managed_process()
            if not proc:
                raise RuntimeError("没有由此工具管理的运行进程；不会停止外部进程")
            # psutil verifies process identity again before sending the signal.
            group = os.getpgid(proc.pid)
            if group != proc.pid:
                raise RuntimeError("进程会话身份不匹配，不会停止")
            # Recheck the root identity before signalling its managed process group.
            if not self.managed_process():
                raise RuntimeError("进程身份已改变")
            descendants = proc.children(recursive=True)
            os.killpg(group, signal.SIGTERM)
            def alive(process):
                try:
                    return process.is_running() and process.status() != psutil.STATUS_ZOMBIE
                except psutil.Error:
                    return False
            for _ in range(100):
                if not any(alive(process) for process in [proc, *descendants]):
                    break
                await asyncio.sleep(0.1)
            for process in [*descendants, proc]:
                if alive(process):
                    try:
                        process.kill()  # psutil checks PID reuse against captured identity.
                    except psutil.NoSuchProcess:
                        pass
            if self.child:
                await asyncio.to_thread(self.child.wait, timeout=5)
                self.child = None
            self.failed = ""
            (self.config.service_state / "service.json").unlink(missing_ok=True)
            self.log(f"{self.adapter.name} 已停止")


def tail(path: Path, limit: int = 48_000) -> str:
    try:
        with path.open("rb") as stream:
            stream.seek(0, 2)
            size = stream.tell()
            stream.seek(max(0, size - limit))
            raw = stream.read(limit)
        if size > limit:
            raw = raw.partition(b"\n")[2]
        return redact(raw.decode("utf-8", errors="replace"))
    except FileNotFoundError:
        return "暂无日志，执行部署或启动后会显示在这里。"
