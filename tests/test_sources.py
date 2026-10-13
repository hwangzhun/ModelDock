import asyncio
from pathlib import Path

import pytest
from textual.widgets import Input

from autodl_tui.app import AutoDLApp
from autodl_tui.config import DOMESTIC_SOURCES, OFFICIAL_SOURCES, Config
from autodl_tui.downloads import Downloads, Model
from autodl_tui.service import Service


def test_domestic_hf_rewrite_keeps_repo_and_revision(tmp_path):
    downloads = Downloads(Config(root=str(tmp_path)))
    model = Model("https://huggingface.co/owner/repo/resolve/v1/vae/file?download=true", "file")
    url, _ = downloads.request(model)
    assert url == "https://hf-mirror.com/owner/repo/resolve/v1/vae/file?download=true"
    # Direct ModelScope URLs never get rewritten to a guessed HF repository.
    model.url = "https://modelscope.cn/models/owner/repo/resolve/master/file"
    assert downloads.request(model)[0] == model.url


def test_child_environment_uses_mirrors_without_implicit_hf_token(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_secret")
    monkeypatch.setenv("HUGGING_FACE_HUB_TOKEN", "hf_legacy")
    monkeypatch.setenv("PIP_EXTRA_INDEX_URL", "https://pypi.org/simple")
    service = Service(Config(root=str(tmp_path)))
    env = service.environment()
    assert env["HF_ENDPOINT"] == DOMESTIC_SOURCES["hf_endpoint"]
    assert DOMESTIC_SOURCES["pip_index"] == "auto"
    assert env["PIP_INDEX_URL"] == OFFICIAL_SOURCES["pip_index"]  # before a deployment selects a mirror
    assert env["PIP_EXTRA_INDEX_URL"] == ""
    assert "HF_TOKEN" not in env and "HUGGING_FACE_HUB_TOKEN" not in env
    assert env["HF_HUB_DISABLE_IMPLICIT_TOKEN"] == "1"
    service.config.hf_endpoint = OFFICIAL_SOURCES["hf_endpoint"]
    assert service.environment()["HF_TOKEN"] == "hf_secret"


async def test_clone_failure_switches_source_and_cleans_partial(tmp_path, monkeypatch):
    service = Service(Config(root=str(tmp_path)))
    attempted = []

    async def command(args):
        attempted.append(args[-2])
        staging = Path(args[-1])
        staging.mkdir()
        if len(attempted) == 1:
            (staging / "partial").write_text("unfinished")
            raise RuntimeError("502")
        assert not (staging / "partial").exists()
        (staging / "main.py").touch()
        (staging / "requirements.txt").touch()

    monkeypatch.setattr(service, "command", command)
    await service.clone_comfy()
    assert attempted == [DOMESTIC_SOURCES["repo"], DOMESTIC_SOURCES["repo_fallback"]]
    assert (service.config.comfy / "main.py").exists()


async def test_cancel_clone_does_not_start_fallback(tmp_path, monkeypatch):
    service = Service(Config(root=str(tmp_path)))
    attempted = []

    async def command(args):
        attempted.append(args[-2])
        raise asyncio.CancelledError

    monkeypatch.setattr(service, "command", command)
    with pytest.raises(asyncio.CancelledError):
        await service.clone_comfy()
    assert attempted == [service.config.repo]


async def test_source_preset_updates_old_configuration(tmp_path):
    config = Config(root=str(tmp_path), **OFFICIAL_SOURCES)
    app = AutoDLApp(config, tmp_path / "config.json")
    async with app.run_test(size=(120, 42)) as pilot:
        await pilot.press("f4")
        await pilot.pause(0.2)
        await pilot.click("#domestic-sources")
        await pilot.pause()
        for key, value in DOMESTIC_SOURCES.items():
            assert app.query_one(f"#config-{key}", Input).value == value
        assert config.repo == OFFICIAL_SOURCES["repo"]  # editing has not saved yet
        app.save_settings()
        assert Config.load(tmp_path / "config.json").repo == DOMESTIC_SOURCES["repo"]
