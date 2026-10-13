import asyncio
import hashlib
import json
from pathlib import Path

import httpx
import pytest

from autodl_tui.config import Config, atomic_json
from autodl_tui.downloads import Downloads, Model, repository_url


@pytest.fixture
def downloads(tmp_path):
    return Downloads(Config(root=str(tmp_path)))


def add(downloads, sha256=""):
    downloads.add([Model("https://example.com/model.safetensors", "model.safetensors", sha256=sha256)])
    return downloads.jobs[0]


async def transfer(downloads, job, handler):
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        await downloads.transfer(client, job)


async def test_download_hash_and_atomic_finish(downloads):
    data = b"model-data" * 100
    job = add(downloads, hashlib.sha256(data).hexdigest())
    await transfer(downloads, job, lambda r: httpx.Response(200, content=data, headers={"etag": '"v1"'}))
    path = downloads.target(job.model)
    assert path.read_bytes() == data
    assert not Path(str(path) + ".part").exists()
    assert job.status == "完成"


def partial(downloads, job, data=b"abc"):
    path = downloads.target(job.model)
    path.parent.mkdir(parents=True, exist_ok=True)
    Path(str(path) + ".part").write_bytes(data)
    atomic_json(Path(str(path) + ".part.json"), {"url": job.model.url, "validator": '"v1"'})
    return path


async def test_range_resume(downloads):
    job = add(downloads)
    path = partial(downloads, job)

    def handler(request):
        assert request.headers["Range"] == "bytes=3-"
        assert request.headers["If-Range"] == '"v1"'
        return httpx.Response(206, content=b"def", headers={"content-range": "bytes 3-5/6", "etag": '"v1"'})

    await transfer(downloads, job, handler)
    assert path.read_bytes() == b"abcdef"


async def test_ignored_range_restarts_instead_of_appending(downloads):
    job = add(downloads)
    path = partial(downloads, job)
    await transfer(downloads, job, lambda r: httpx.Response(200, content=b"new-file"))
    assert path.read_bytes() == b"new-file"


async def test_bad_range_never_promotes_partial(downloads):
    job = add(downloads)
    path = partial(downloads, job)
    with pytest.raises(ValueError, match="续传范围"):
        await transfer(downloads, job, lambda r: httpx.Response(206, content=b"def", headers={
            "content-range": "bytes 2-4/5"}))
    assert not path.exists()
    assert Path(str(path) + ".part").read_bytes() == b"abc"


async def test_bad_hash_retains_partial_but_invalidates_resume(downloads):
    job = add(downloads, "0" * 64)
    with pytest.raises(ValueError, match="SHA256"):
        await transfer(downloads, job, lambda r: httpx.Response(200, content=b"bad-data"))
    path = downloads.target(job.model)
    assert not path.exists()
    assert Path(str(path) + ".part").exists()
    assert not Path(str(path) + ".part.json").exists()


async def test_html_error_is_not_model(downloads):
    job = add(downloads)
    with pytest.raises(ValueError, match="网页"):
        await transfer(downloads, job, lambda r: httpx.Response(200, text="login", headers={
            "content-type": "text/html; charset=UTF-8"}))
    assert not downloads.target(job.model).exists()


async def test_existing_file_never_overwritten(downloads):
    job = add(downloads)
    target = downloads.target(job.model)
    target.parent.mkdir(parents=True)
    target.write_bytes(b"existing")
    await transfer(downloads, job, lambda r: pytest.fail("must not fetch existing file"))
    assert target.read_bytes() == b"existing"
    assert job.status == "已存在"


async def test_pause_preserves_partial(downloads):
    job = add(downloads)

    class Stream(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b"a" * (256 * 1024)
            downloads.pause(job.id)
            yield b"b" * (256 * 1024)

    downloads.active = job.id
    await transfer(downloads, job, lambda r: httpx.Response(200, stream=Stream(), headers={"etag": '"v1"'}))
    path = downloads.target(job.model)
    assert job.status == "暂停"
    assert not path.exists()
    assert Path(str(path) + ".part").stat().st_size == 256 * 1024


async def test_cancel_task_persists_paused_queue(downloads, monkeypatch):
    job = add(downloads)

    async def slow_transfer(client, selected):
        selected.status = "下载中"
        await asyncio.sleep(60)

    monkeypatch.setattr(downloads, "transfer", slow_transfer)
    task = asyncio.create_task(downloads.run())
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert job.status == "暂停"
    assert Downloads(downloads.config).jobs[0].status == "暂停"


@pytest.mark.parametrize("filename", ["../escape", "/tmp/escape", "x/y", "..", ".", "x\\y", "x\n", ".hidden"])
def test_reject_traversal_and_invalid_filename(filename):
    with pytest.raises(ValueError):
        Model("https://example.com/file", filename).validate()


def test_symlink_escape_blocked(downloads, tmp_path):
    job = add(downloads)
    downloads.config.models.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (downloads.config.models / "checkpoints").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="之外"):
        downloads.target(job.model)


def test_manifest_validation_is_atomic(downloads, tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"version": 1, "models": [
        {"url": "https://example.com/a", "filename": "a"},
        {"url": "https://example.com/b", "filename": "../b"},
    ]}))
    with pytest.raises(ValueError):
        downloads.import_manifest(manifest)
    assert not downloads.jobs


def test_manifest_roundtrip_and_duplicate(downloads, tmp_path):
    job = add(downloads)
    path = tmp_path / "manifest.json"
    downloads.export_manifest(path)
    other = Downloads(Config(root=str(tmp_path / "other")))
    assert other.import_manifest(path) == 1
    assert other.jobs[0].model == job.model
    assert other.import_manifest(path) == 0


def test_source_builders_and_hf_page_conversion():
    assert repository_url("modelscope", "Qwen/Qwen-Image", "vae/权重.safetensors").startswith(
        "https://modelscope.cn/models/Qwen/Qwen-Image/resolve/master/vae/%")
    assert repository_url("huggingface", "a/b", "x/y", "v1") == "https://huggingface.co/a/b/resolve/v1/x/y"
    model = Model.from_url("https://huggingface.co/a/b/blob/main/x.safetensors", "vae")
    assert "/resolve/main/" in model.url
    assert model.filename == "x.safetensors"


def test_tokens_only_go_to_official_https_origin(downloads, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_secret")
    downloads.config.hf_endpoint = "https://huggingface.co"
    model = Model("https://huggingface.co/a/b/resolve/main/file", "file")
    assert downloads.request(model)[1]["Authorization"] == "Bearer hf_secret"
    downloads.config.hf_endpoint = "https://hf-mirror.com"
    url, headers = downloads.request(model)
    assert url.startswith("https://hf-mirror.com/")
    assert "Authorization" not in headers
    assert "Authorization" not in downloads.request(Model("https://evil.com/file", "file"))[1]


async def test_416_invalidates_resume(downloads):
    job = add(downloads)
    path = partial(downloads, job)
    with pytest.raises(ValueError, match="从头"):
        await transfer(downloads, job, lambda r: httpx.Response(416))
    assert not Path(str(path) + ".part.json").exists()
    assert not path.exists()
