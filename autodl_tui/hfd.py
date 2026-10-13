"""Run the HF-Mirror hfd script in an isolated, resumable staging directory."""
from __future__ import annotations

import asyncio
import hashlib
import os
import re
import shutil
import signal
from pathlib import Path
from urllib.parse import unquote, urlsplit, urlunsplit

import httpx

from .config import redact

# Pinned upstream script; the mirror and upstream URLs return the same bytes.
HFD_SHA256 = "524894ef4f8e82da6180156a96bd9678d56d77e877b7f69a9d253da97d28866f"
HFD_PATCHED_SHA256 = "4e5635db48951019fa88f513968e2873d2990ce692787d736bd39f7de4a095a5"
INTEGER_PATCHES = (
    ('print (n+0)" "(s+0)', 'printf "%.0f %.0f\\n", n, s'),
    ('print want[q] "\\t" q', 'printf "%.0f\\t%s\\n", want[q], q'),
    ('END { print b*512, d-a }', 'END { printf "%.0f %.0f\\n", b*512, d-a }'),
)
HFD_URLS = (
    "https://hf-mirror.com/hfd/hfd.sh",
    "https://gist.githubusercontent.com/padeoe/697678ab8e528b85a2a7bddafea1fa4f/raw/774ee6d344fb94438b1831d49ae83647c2d609ea/hfd.sh",
)
ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
NUMBER = r"([\d.]+\s*(?:TB|GB|MB|KB|B))"
PROGRESS = re.compile(r"\|\s*" + NUMBER + r"\s*/\s*" + NUMBER + r"\s*\|\s*" + NUMBER + r"/s")


def selection(url: str) -> tuple[str, str, str, str] | None:
    parsed = urlsplit(url)
    parts = parsed.path.strip("/").split("/")
    if parsed.hostname not in ("huggingface.co", "hf-mirror.com") or len(parts) < 5 or parts[2] != "resolve":
        return None
    repo = "/".join(parts[:2])
    if not re.fullmatch(r"[\w.-]+/[\w.-]+", repo):
        raise ValueError("HF 仓库 ID 无效")
    path = unquote("/".join(parts[4:]))
    if (any(p in ("", ".", "..") for p in path.split("/")) or "\\" in path or "*" in path
            or any(ord(c) < 32 for c in path)):
        raise ValueError("仓库文件路径不能包含空段、遍历路径、控制字符或星号")
    return repo, parts[3], path, urlunsplit(parsed._replace(path="", query="", fragment=""))


def log(state: Path, line: str) -> None:
    state.mkdir(parents=True, exist_ok=True)
    path = state / "hfd-downloads.log"
    if path.exists() and path.stat().st_size > 5 * 1024 * 1024:
        path.replace(path.with_suffix(".log.1"))
    with path.open("a", encoding="utf-8") as stream:
        stream.write(redact(line) + "\n")


def compatible_script(data: bytes) -> bytes:
    """Keep awk's large byte counts decimal: Bash cannot calculate scientific notation."""
    text = data.decode("utf-8")
    for original, replacement in INTEGER_PATCHES:
        if text.count(original) != 1:
            raise ValueError("hfd 大文件兼容补丁与脚本版本不匹配")
        text = text.replace(original, replacement)
    return text.encode("utf-8")


def save_script(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as stream:
        stream.write(data)
    temporary.replace(path)


async def script(client: httpx.AsyncClient, state: Path) -> Path:
    missing = [name for name in ("bash", "curl", "aria2c") if not shutil.which(name)]
    if missing:
        raise ValueError("hfd 缺少 " + "、".join(missing) +
                         "；Debian/Ubuntu 请执行：apt-get update && apt-get install -y curl aria2")
    path = state / "tools" / "hfd.sh"
    if path.is_symlink():
        raise ValueError("hfd 脚本不可为符号链接")
    if path.is_file():
        cached = path.read_bytes()
        digest = hashlib.sha256(cached).hexdigest()
        if digest == HFD_PATCHED_SHA256:
            return path
        if digest == HFD_SHA256:
            save_script(path, compatible_script(cached))
            log(state, "已修复缓存 hfd 的大文件整数统计；保留下载和 aria2 续传文件")
            return path
    for url in HFD_URLS:
        try:
            response = await client.get(url)
            response.raise_for_status()
            if hashlib.sha256(response.content).hexdigest() != HFD_SHA256:
                raise ValueError("hfd 脚本校验不匹配")
            save_script(path, compatible_script(response.content))
            return path
        except (httpx.HTTPError, ValueError) as error:
            log(state, f"获取 hfd 失败：{url}：{error}")
    raise ValueError("无法获取经校验的 hfd 脚本，请检查 HF-Mirror / GitHub 网络；详情见 hfd-downloads.log")


def byte_count(value: str) -> int:
    match = re.fullmatch(r"([\d.]+)\s*(TB|GB|MB|KB|B)", value.strip())
    return int(float(match[1]) * 1000 ** ("B", "KB", "MB", "GB", "TB").index(match[2])) if match else 0


def update_progress(job, text: str) -> None:
    match = PROGRESS.search(text)
    if match:
        job.downloaded, job.total, job.speed = (byte_count(value) for value in match.groups())


async def stop(proc: asyncio.subprocess.Process) -> None:
    # Signal the whole session: hfd owns aria2, curl, and a progress-monitor child.
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


def manifest_size(staging: Path, path: str) -> int:
    manifest = staging / ".hfd" / "manifest"
    if not manifest.exists():
        return 0
    rows = manifest.read_text(encoding="utf-8").splitlines()
    entries = [row.split("\t", 1) for row in rows]
    if len(entries) != 1 or len(entries[0]) != 2 or entries[0][1] != path:
        raise ValueError("hfd 文件清单与所选文件不一致")
    return int(entries[0][0])


def repair_cached_totals(staging: Path, path: str) -> None:
    total = manifest_size(staging, path)
    if total:
        # Old hfd may have already persisted "1 5.50634e+10". Use the exact manifest,
        # not float conversion of that rounded string. The data and .aria2 stay untouched.
        for name in ("repo_info", "filelist_stats"):
            (staging / ".hfd" / name).write_text(f"1 {total}\n", encoding="utf-8")


async def download(downloads, client: httpx.AsyncClient, job, target: Path) -> None:
    url, _ = downloads.request(job.model)
    chosen = selection(job.model.url)
    assert chosen is not None
    repo, revision, path, _ = chosen
    parsed = urlsplit(url)
    endpoint = urlunsplit(parsed._replace(path="", query="", fragment=""))
    job.status = "下载中"
    script_path = await script(client, downloads.config.state)
    staging = downloads.config.state / "hfd" / job.id
    if staging.is_symlink():
        raise ValueError("hfd 临时目录不可为符号链接")
    staging.mkdir(parents=True, exist_ok=True, mode=0o700)
    source = staging / path
    if not source.resolve().is_relative_to(staging.resolve()):
        raise ValueError("hfd 临时文件指向下载目录之外")
    if source.is_symlink() or Path(str(source) + ".aria2").is_symlink():
        raise ValueError("hfd 临时文件不可为符号链接")
    repair_cached_totals(staging, path)
    env = dict(os.environ)
    env["HF_ENDPOINT"] = endpoint
    for key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HF_USERNAME"):
        env.pop(key, None)
    token = os.environ.get("HF_TOKEN", "") if endpoint == "https://huggingface.co" else ""
    if token:
        env["HF_TOKEN"] = token
        username = os.environ.get("HF_USERNAME", "")
        if not username:
            response = await client.get("https://huggingface.co/api/whoami-v2",
                                        headers={"Authorization": f"Bearer {token}"})
            response.raise_for_status()
            username = response.json().get("name", "")
        env["HF_USERNAME"] = username
    # Upstream's include is an unanchored regex after wildcard conversion; anchor the exact file.
    include = "^" + re.escape(path).replace(r"\.", ".") + "$"
    args = ["bash", str(script_path), repo, "--include", include, "--revision", revision,
            "--local-dir", str(staging), "--tool", "aria2c", "-x", str(downloads.config.hfd_threads), "-j", "1"]
    log(downloads.config.state, f"hfd：{repo}/{path}，每文件 {downloads.config.hfd_threads} 个连接，站点 {endpoint}")
    if Path(str(target) + ".part").exists():
        log(downloads.config.state, "旧版 .part 保留；hfd 使用独立的 aria2 续传文件，首次从头下载")
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
            total = manifest_size(staging, path)
            if total:
                job.total = total
                allocated = source.stat().st_blocks * 512 if source.exists() else 0
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
            native_log = staging / ".hfd" / "download.log"
            if native_log.is_file():
                with native_log.open("rb") as stream:
                    stream.seek(max(0, native_log.stat().st_size - 4096))
                    native_tail = redact(stream.read().decode(errors="replace"))
                log(downloads.config.state, native_tail)
                tail.extend(native_tail.splitlines()[-3:])
            details = "；".join(tail[-3:])
            raise RuntimeError(f"hfd 下载失败（退出码 {proc.returncode}）：{details}；详情见 hfd-downloads.log")
        total = manifest_size(staging, path)
        if not total or not source.is_file() or source.stat().st_size != total or Path(str(source) + ".aria2").exists():
            raise ValueError("hfd 未生成完整文件；保留临时文件，可重试续传")
        if job.model.sha256:
            from .downloads import file_hash
            job.status = "校验中"
            if await asyncio.to_thread(file_hash, source) != job.model.sha256.lower():
                # hfd skips same-size files: remove only our staging copy so a retry actually downloads.
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
        try:
            await stop(proc)
            await waiter
            await reader
        finally:
            # hfd writes token-bearing headers and a command fingerprint. Do not retain credentials.
            for name in ("last_download_command", "aria2c_urls.txt", "list_state"):
                (staging / ".hfd" / name).unlink(missing_ok=True)
