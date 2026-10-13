from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import shutil
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.parse import quote, unquote, urlsplit, urlunsplit

import httpx

from .config import CATEGORIES, Config, atomic_json, redact, validate_url
from .hfd import download as hfd_download
from .hfd import selection as hfd_selection
from .service import workspace_lock


def repository_url(source: str, repo: str, path: str, revision: str = "") -> str:
    repo, path = repo.strip(), path.strip()
    if not re.fullmatch(r"[\w.-]+/[\w.-]+", repo):
        raise ValueError("仓库 ID 格式应为 所有者/仓库名")
    if not path or path.startswith("/") or any(p in ("..", ".", "") for p in path.split("/")):
        raise ValueError("请输入有效的仓库内文件路径")
    if source == "modelscope":
        base, revision = "https://modelscope.cn/models", revision.strip() or "master"
    elif source == "huggingface":
        base, revision = "https://huggingface.co", revision.strip() or "main"
    else:
        raise ValueError("不支持的仓库来源")
    return f"{base}/{quote(repo, safe='/')}/resolve/{quote(revision, safe='')}/{quote(path, safe='/')}"


@dataclass
class Model:
    url: str
    filename: str
    category: str = "checkpoints"
    sha256: str = ""

    def validate(self) -> None:
        validate_url(self.url)
        if (not self.filename or self.filename in (".", "..") or
                any(c in self.filename for c in '/\\\x00') or
                any(ord(c) < 32 for c in self.filename) or self.filename.startswith(".")):
            raise ValueError("文件名不可为空、以点开头或包含路径分隔符")
        if self.filename.endswith((".part", ".part.json")):
            raise ValueError("文件名不能使用下载临时文件的扩展名")
        if self.category not in CATEGORIES:
            raise ValueError("请选择有效的模型分类")
        if self.sha256 and not re.fullmatch(r"[a-fA-F0-9]{64}", self.sha256):
            raise ValueError("SHA256 应为 64 位十六进制字符串，也可留空")

    @classmethod
    def from_url(cls, url: str, category: str, filename: str = "", sha256: str = "") -> Model:
        url = url.strip()
        parsed = urlsplit(url)
        # Accept a Hugging Face file page pasted from the browser.
        if parsed.hostname in ("huggingface.co", "hf-mirror.com"):
            url = urlunsplit(parsed._replace(path=parsed.path.replace("/blob/", "/resolve/", 1)))
        filename = filename.strip() or unquote(urlsplit(url).path.rsplit("/", 1)[-1])
        if filename.isdecimal():
            raise ValueError("Civitai 下载链接请填写文件名，例如 model.safetensors")
        model = cls(url, filename, category, sha256.strip())
        model.validate()
        return model


@dataclass
class Job:
    id: str
    model: Model
    status: str = "等待"
    downloaded: int = 0
    total: int = 0
    speed: float = 0
    error: str = ""


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(4 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


class Downloads:
    def __init__(self, config: Config):
        self.config = config
        self.jobs: list[Job] = []
        self.active: str | None = None
        self.cancelled: set[str] = set()
        self.running = False
        self.load()

    @property
    def queue_path(self) -> Path:
        return self.config.state / "downloads.json"

    def load(self) -> None:
        if not self.queue_path.exists():
            return
        data = json.loads(self.queue_path.read_text(encoding="utf-8"))
        for row in data:
            model = Model(**row["model"])
            model.validate()
            job = Job(row["id"], model, row.get("status", "暂停"))
            if job.status in ("等待", "下载中", "校验中"):
                job.status = "暂停"
            self.jobs.append(job)

    def save(self) -> None:
        atomic_json(self.queue_path, [{"id": job.id, "model": asdict(job.model), "status": job.status}
                                     for job in self.jobs])

    def add(self, models: list[Model]) -> int:
        for model in models:
            model.validate()
        added = 0
        for model in models:
            if any((j.model.category, j.model.filename) == (model.category, model.filename) for j in self.jobs):
                continue
            self.jobs.append(Job(uuid.uuid4().hex[:12], model))
            added += 1
        self.save()
        return added

    def retry(self, job_id: str | None = None) -> None:
        for job in self.jobs:
            if (job_id is None or job.id == job_id) and job.status in ("暂停", "失败", "已存在", "完成"):
                self.cancelled.discard(job.id)
                job.status, job.error = "等待", ""
        self.save()

    def pause(self, job_id: str | None = None) -> None:
        for job in self.jobs:
            if (job_id is None or job.id == job_id) and job.status in ("等待", "下载中", "校验中"):
                self.cancelled.add(job.id)
                if job.id != self.active:
                    job.status = "暂停"
        self.save()

    def remove(self, job_id: str) -> None:
        if self.active == job_id:
            raise ValueError("请先暂停此下载，再移出清单")
        self.jobs = [job for job in self.jobs if job.id != job_id]
        self.save()

    def export_manifest(self, path: Path) -> None:
        atomic_json(path, {"version": 1, "models": [asdict(job.model) for job in self.jobs]})

    def import_manifest(self, path: Path) -> int:
        if path.stat().st_size > 2 * 1024 * 1024:
            raise ValueError("模型清单不能超过 2 MiB")
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or data.get("version") != 1 or not isinstance(data.get("models"), list):
            raise ValueError("清单格式无效，需要 version: 1 和 models 数组")
        return self.add([Model(**row) for row in data["models"]])

    def target(self, model: Model) -> Path:
        model.validate()
        path = self.config.models / model.category / model.filename
        if not path.resolve().is_relative_to(self.config.models.resolve()):
            raise ValueError("目标路径指向模型目录之外")
        for suffix in (".part", ".part.json"):
            if Path(str(path) + suffix).is_symlink():
                raise ValueError("下载临时文件不可为符号链接")
        return path

    def request(self, model: Model) -> tuple[str, dict[str, str]]:
        url = model.url
        parsed = urlsplit(url)
        if parsed.hostname == "huggingface.co":
            mirror = urlsplit(self.config.hf_endpoint)
            url = urlunsplit(parsed._replace(scheme=mirror.scheme, netloc=mirror.netloc))
        host = urlsplit(url).hostname
        token = os.environ.get("HF_TOKEN", "") if host == "huggingface.co" else (
            os.environ.get("CIVITAI_TOKEN", "") if host in ("civitai.com", "www.civitai.com") else "")
        headers = {"Accept-Encoding": "identity", "User-Agent": "ModelDock/0.3"}
        if token and urlsplit(url).scheme == "https":
            headers["Authorization"] = f"Bearer {token}"
        return url, headers

    async def expand(self, model: Model, client: httpx.AsyncClient | None = None) -> list[Model]:
        """Expand HF folder selections into individual, resumable file jobs."""
        parsed = urlsplit(model.url)
        parts = parsed.path.strip("/").split("/")
        if (parsed.hostname not in ("huggingface.co", "hf-mirror.com") or len(parts) < 4
                or parts[2] not in ("tree", "resolve")):
            return [model]
        explicit = parts[2] == "tree"
        folder = "/".join(parts[4:])
        if not explicit and Path(unquote(folder)).suffix:
            return [model]
        api = urlunsplit(parsed._replace(
            path=f"/api/models/{parts[0]}/{parts[1]}/tree/{parts[3]}" + (f"/{folder}" if folder else ""),
            query="recursive=true&limit=1000", fragment=""))
        url, headers = self.request(Model(api, "listing"))
        if client is None:
            async with httpx.AsyncClient(follow_redirects=True, timeout=30) as owned:
                return await self.expand(model, owned)
        models: list[Model] = []
        seen_pages: set[str] = set()
        prefix = unquote(folder).rstrip("/")
        while url:
            if url in seen_pages:
                raise ValueError("HF 目录分页重复，请稍后重试")
            seen_pages.add(url)
            response = await client.get(url, headers=headers)
            if response.status_code == 404 and not explicit and len(seen_pages) == 1:
                # Extensionless files remain supported; the tree endpoint only lists folders.
                return [model]
            response.raise_for_status()
            if model.sha256 or model.filename != unquote(parts[-1]):
                raise ValueError("HF 目录下载请留空文件名和 SHA256；单文件请填写完整文件路径")
            rows = response.json()
            if not isinstance(rows, list):
                raise ValueError("HF 目录响应格式无效")
            for row in rows:
                if row.get("type") != "file":
                    continue
                path = row.get("path", "")
                if prefix and not path.startswith(prefix + "/"):
                    raise ValueError("HF 返回了所选目录之外的文件")
                file_url = urlunsplit(parsed._replace(
                    path=f"/{parts[0]}/{parts[1]}/resolve/{parts[3]}/{quote(path, safe='/')}",
                    query="", fragment=""))
                models.append(Model.from_url(file_url, model.category))
            next_url = response.links.get("next", {}).get("url", "")
            if next_url and urlsplit(next_url).netloc != urlsplit(url).netloc:
                raise ValueError("HF 目录分页地址不属于当前站点")
            if next_url and urlsplit(next_url).scheme != urlsplit(url).scheme:
                raise ValueError("HF 目录分页协议发生变化")
            url = next_url
        if not models:
            raise ValueError("HF 目录中没有可下载的文件")
        if len({m.filename for m in models}) != len(models):
            raise ValueError("HF 目录含有同名文件，请分别选择子目录或具体文件")
        return models

    async def run(self) -> None:
        if self.running:
            return
        self.running = True
        try:
            with workspace_lock(self.config, "downloads"):
                async with httpx.AsyncClient(follow_redirects=True, timeout=httpx.Timeout(30, connect=20)) as client:
                    while job := next((j for j in self.jobs if j.status == "等待"), None):
                        self.active = job.id
                        try:
                            await self.transfer(client, job)
                        except asyncio.CancelledError:
                            job.status = "暂停"
                            raise
                        except Exception as error:
                            job.status = "失败"
                            job.error = redact(str(error))
                        finally:
                            job.speed = 0
                            self.active = None
                            self.save()
        finally:
            self.running = False

    async def transfer(self, client: httpx.AsyncClient, job: Job) -> None:
        target = self.target(job.model)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if job.model.sha256 and await asyncio.to_thread(file_hash, target) != job.model.sha256.lower():
                raise ValueError("已有文件的 SHA256 不匹配；请手动移走该文件后重试，不会覆盖")
            job.status = "已存在"
            job.downloaded = job.total = target.stat().st_size
            return
        if hfd_selection(job.model.url):
            await hfd_download(self, client, job, target)
            return
        partial = Path(str(target) + ".part")
        metadata = Path(str(target) + ".part.json")
        url, headers = self.request(job.model)
        offset = 0
        try:
            previous = json.loads(metadata.read_text())
            if previous.get("url") == url and previous.get("validator") and partial.exists():
                offset = partial.stat().st_size
                headers.update({"Range": f"bytes={offset}-", "If-Range": previous["validator"]})
        except (OSError, ValueError, AttributeError):
            pass
        job.status = "下载中"
        async with client.stream("GET", url, headers=headers) as response:
            if response.status_code == 416:
                # Conservative restart: do not assume a partial file is complete from length alone.
                metadata.unlink(missing_ok=True)
                raise ValueError("服务器拒绝续传范围；再次重试将从头下载")
            if response.status_code == 404:
                raise ValueError("下载地址不存在（HTTP 404）；仓库路径必须指向具体文件。"
                                 "若填写的是 IQ3_S 等目录，请移除此任务，再添加 HF 目录链接以展开文件清单")
            response.raise_for_status()
            if "text/html" in response.headers.get("content-type", "").lower():
                raise ValueError("收到网页而非模型文件，请检查下载链接、访问权限或登录令牌")
            if response.headers.get("content-encoding", "identity") != "identity":
                raise ValueError("服务器返回压缩传输，无法安全续传，请更换直链")
            length = int(response.headers.get("content-length", "0"))
            if response.status_code == 206:
                match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("content-range", ""))
                if not match or int(match[1]) != offset or int(match[2]) + 1 != int(match[3]):
                    metadata.unlink(missing_ok=True)
                    raise ValueError("服务器返回的续传范围不正确，请重试")
                total = int(match[3])
            elif response.status_code == 200:
                offset, total = 0, length
            else:
                raise ValueError(f"无法处理下载响应：HTTP {response.status_code}")
            if total and shutil.disk_usage(target.parent).free < total - offset + 64 * 1024**2:
                raise ValueError("磁盘剩余空间不足以保存此模型")
            validator = response.headers.get("etag", "")
            if validator.startswith("W/"):
                validator = ""
            validator = validator or response.headers.get("last-modified", "")
            # Truncate before replacing metadata so an interrupted restart can't pair old bytes with a new ETag.
            with partial.open("ab" if offset else "wb") as output:
                atomic_json(metadata, {"url": url, "validator": validator})
                job.downloaded, job.total = offset, total
                started = time.monotonic()
                async for block in response.aiter_bytes(256 * 1024):
                    if job.id in self.cancelled:
                        job.status = "暂停"
                        return
                    output.write(block)
                    job.downloaded += len(block)
                    job.speed = (job.downloaded - offset) / max(0.01, time.monotonic() - started)
                    await asyncio.sleep(0)
            if total and job.downloaded != total:
                raise ValueError("文件大小与服务器声明不一致，可重试续传")
        if job.id in self.cancelled:
            job.status = "暂停"
            return
        if job.model.sha256:
            job.status = "校验中"
            if await asyncio.to_thread(file_hash, partial) != job.model.sha256.lower():
                metadata.unlink(missing_ok=True)
                raise ValueError("SHA256 校验失败；保留 .part 文件，再次重试将重新下载")
        if job.id in self.cancelled:
            job.status = "暂停"
            return
        # link() refuses to overwrite a file created by another process during the download.
        os.link(partial, target)
        partial.unlink()
        metadata.unlink(missing_ok=True)
        job.status = "完成"

