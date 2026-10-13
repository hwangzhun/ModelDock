"""Run the official hf download CLI in a resumable staging directory."""
from __future__ import annotations

import asyncio
import os
import re
import shutil
import signal
import sys
from pathlib import Path
from urllib.parse import unquote, urlsplit, urlunsplit

import httpx

from .config import redact

ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
NUMBER = r"([\d.]+\s*(?:[kKMGT]i?B?|B))"
PROGRESS = re.compile(NUMBER + r"\s*/\s*" + NUMBER + r".*?,\s*" + NUMBER + r"/s")


def selection(url: str) -> tuple[str, str, str, str] | None:
    parsed = urlsplit(url)
    parts = parsed.path.strip("/").split("/")
    if parsed.hostname not in ("huggingface.co", "hf-mirror.com") or len(parts) < 5 or parts[2] != "resolve":
        return None
    repo = "/".join(parts[:2])
    if not re.fullmatch(r"[\w.-]+/[\w.-]+", repo):
        raise ValueError("HF 仓库 ID 无效")
    path = unquote("/".join(parts[4:]))
    if (any(p in ("", ".", "..") for p in path.split("/")) or "\\" in path or "*" in path or path.startswith("-")
            or any(ord(c) < 32 for c in path)):
        raise ValueError("仓库文件路径不能包含空段、遍历路径、控制字符或星号")
    return repo, parts[3], path, urlunsplit(parsed._replace(path="", query="", fragment=""))


def log(state: Path, line: str) -> None:
    state.mkdir(parents=True, exist_ok=True)
    path = state / "hf-downloads.log"
    if path.exists() and path.stat().st_size > 5 * 1024 * 1024:
        path.replace(path.with_suffix(".log.1"))
    with path.open("a", encoding="utf-8") as stream:
        stream.write(redact(line) + "\n")


def command() -> str:
    # Prefer the CLI installed alongside ModelDock, even when the venv is not activated.
    bundled = Path(sys.executable).parent / "hf"
    executable = str(bundled) if bundled.is_file() and os.access(bundled, os.X_OK) else shutil.which("hf")
    if not executable:
        raise ValueError("缺少 hf 命令，请在 ModelDock 的 Python 环境执行：pip install 'huggingface_hub>=1,<2'")
    return executable


def byte_count(value: str) -> int:
    match = re.fullmatch(r"([\d.]+)\s*([kKMGT]?)(i?)(B?)", value.strip())
    if not match:
        return 0
    power = " KMGT".index(match[2].upper() or " ")
    return int(float(match[1]) * (1024 if match[3] else 1000) ** power)


def update_progress(job, text: str) -> None:
    match = PROGRESS.search(text)
    if match:
        job.downloaded, total, job.speed = (byte_count(value) for value in match.groups())
        if not job.total:
            job.total = total


async def stop(proc: asyncio.subprocess.Process) -> None:
    # Stop the CLI and any transfer workers in its session.
    for sig, timeout in ((signal.SIGINT, 3), (signal.SIGTERM, 2), (signal.SIGKILL, 2)):
        try:
            os.killpg(proc.pid, sig)
        except ProcessLookupError:
            break
        try:
            await asyncio.wait_for(proc.wait(), timeout)
        except asyncio.TimeoutError:
            continue
        # The shell can exit while a child ignores the signal.
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        break
    await proc.wait()


async def download(downloads, client: httpx.AsyncClient, job, target: Path) -> None:
    url, headers = downloads.request(job.model)
    chosen = selection(job.model.url)
    assert chosen is not None
    repo, revision, path, _ = chosen
    parsed = urlsplit(url)
    endpoint = urlunsplit(parsed._replace(path="", query="", fragment=""))
    job.status = "下载中"
    executable = command()
    staging = downloads.config.state / "hf" / job.id
    if staging.is_symlink():
        raise ValueError("hf 临时目录不可为符号链接")
    staging.mkdir(parents=True, exist_ok=True, mode=0o700)
    source = staging / path
    if not source.resolve().is_relative_to(staging.resolve()):
        raise ValueError("hf 临时文件指向下载目录之外")
    if source.is_symlink() or (staging / ".cache").is_symlink():
        raise ValueError("hf 临时文件不可为符号链接")
    env = dict(os.environ)
    env["HF_ENDPOINT"] = endpoint
    for key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HF_USERNAME"):
        env.pop(key, None)
    token = os.environ.get("HF_TOKEN", "") if endpoint == "https://huggingface.co" else ""
    env["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "0" if token else "1"
    env["HF_HUB_DISABLE_UPDATE_CHECK"] = "1"
    env["HF_HUB_DISABLE_PROGRESS_BARS"] = "0"
    if token:
        env["HF_TOKEN"] = token
    if job.id in downloads.cancelled:
        job.status = "暂停"
        return
    # Read the original redirect response: x-linked-size describes LFS/Xet payloads.
    response = await client.head(url, headers=headers, follow_redirects=False)
    if not response.is_redirect:
        response.raise_for_status()
    if response.is_redirect and "x-linked-size" not in response.headers:
        response = await client.head(url, headers=headers, follow_redirects=True)
        response.raise_for_status()
    size = response.headers.get("x-linked-size", response.headers.get("content-length", ""))
    if not size.isdecimal():
        raise ValueError("HF 未返回文件大小，无法校验下载完整性")
    total = int(size)
    job.total = total
    args = [executable, "download", repo, path, "--revision", unquote(revision),
            "--local-dir", str(staging)]
    log(downloads.config.state, f"hf download：{repo}/{path}，站点 {endpoint}")
    if Path(str(target) + ".part").exists():
        log(downloads.config.state, "旧版 .part 保留；hf download 使用独立的缓存续传文件，首次从头下载")
    if job.id in downloads.cancelled:
        job.status = "暂停"
        return
    proc = await asyncio.create_subprocess_exec(*args, env=env, cwd=staging,
                                              stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
                                              start_new_session=True)
    tail: list[str] = []

    async def output():
        buffer = ""
        assert proc.stdout
        while block := await proc.stdout.read(4096):
            buffer += block.decode(errors="replace")
            lines = re.split(r"[\r\n]", buffer)
            buffer = lines.pop()[-8192:]
            for line in lines:
                clean = ANSI.sub("", line).strip()
                if clean:
                    update_progress(job, clean)
                    tail.append(clean)
                    del tail[:-8]
                    log(downloads.config.state, clean)
        if buffer.strip():
            clean = ANSI.sub("", buffer).strip()
            update_progress(job, clean)
            tail.append(clean)
            log(downloads.config.state, clean)

    reader = asyncio.create_task(output())
    waiter = asyncio.create_task(proc.wait())
    try:
        while not waiter.done():
            if job.id in downloads.cancelled:
                await stop(proc)
                job.status = "暂停"
                return
            incomplete = list((staging / ".cache" / "huggingface" / "download").rglob("*.incomplete"))
            allocated = 0
            for partial in incomplete:
                try:
                    allocated += partial.stat().st_blocks * 512
                except FileNotFoundError:
                    # The CLI may rename a completed temporary file while we poll.
                    pass
            if source.exists():
                allocated += source.stat().st_blocks * 512
            if shutil.disk_usage(target.parent).free < max(0, total - allocated) + 64 * 1024**2:
                raise ValueError("磁盘剩余空间不足以保存此模型")
            if reader.done() and reader.exception():
                raise reader.exception()
            await asyncio.wait({waiter}, timeout=0.2)
        await reader
        if job.id in downloads.cancelled:
            job.status = "暂停"
            return
        if proc.returncode:
            details = "；".join(tail[-3:])
            raise RuntimeError(f"hf 下载失败（退出码 {proc.returncode}）：{details}；详情见 hf-downloads.log")
        if (source.is_symlink() or not source.resolve().is_relative_to(staging.resolve())
                or not source.is_file() or source.stat().st_size != total):
            raise ValueError("hf 未生成完整文件；保留临时文件，可重试续传")
        if job.model.sha256:
            from .downloads import file_hash
            job.status = "校验中"
            if await asyncio.to_thread(file_hash, source) != job.model.sha256.lower():
                # Remove the invalid staging copy so the CLI refetches it on retry.
                source.unlink()
                raise ValueError("SHA256 校验失败；重试将重新下载")
        if job.id in downloads.cancelled:
            job.status = "暂停"
            return
        os.link(source, target)
        source.unlink()
        job.downloaded = job.total = total
        job.status = "完成"
    finally:
        await stop(proc)
        await waiter
        await reader
