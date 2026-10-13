
import pytest
from textual.widgets import Input, Select, Static, TabbedContent

from autodl_tui.app import AutoDLApp, ModelDialog
from autodl_tui.config import Config
from autodl_tui.downloads import Model


@pytest.mark.parametrize("dimensions", [(120, 42), (80, 24)])
async def test_navigation_and_modelscope_form(tmp_path, monkeypatch, dimensions):
    app = AutoDLApp(Config(root=str(tmp_path)), tmp_path / "config.json")
    monkeypatch.setattr(app, "start_downloads", lambda: None)
    async with app.run_test(size=dimensions) as pilot:
        await pilot.pause(0.2)
        await pilot.press("f2")
        assert app.query_one(TabbedContent).active == "models"
        await pilot.press("ctrl+n")
        await pilot.pause()
        assert isinstance(app.screen, ModelDialog)
        app.screen.query_one("#source", Select).value = "modelscope"
        await pilot.pause()
        app.screen.query_one("#repo-id", Input).value = "Qwen/Qwen-Image"
        app.screen.query_one("#repo-file", Input).value = "vae/model.safetensors"
        app.screen.query_one("#category", Select).value = "vae"
        app.screen.confirm_model()
        await pilot.pause()
        assert len(app.downloads.jobs) == 1
        model = app.downloads.jobs[0].model
        assert model.url == "https://modelscope.cn/models/Qwen/Qwen-Image/resolve/master/vae/model.safetensors"
        assert model.category == "vae"
        await pilot.press("f3")
        assert app.query_one(TabbedContent).active == "logs"
        await pilot.press("f4")
        assert app.query_one(TabbedContent).active == "settings"
        app.query_one("#config-port", Input).value = "6008"
        app.save_settings()
        assert Config.load(tmp_path / "config.json").port == 6008
        await pilot.press("ctrl+q")


async def test_error_does_not_crash_ui(tmp_path):
    app = AutoDLApp(Config(root=str(tmp_path)), tmp_path / "config.json")
    async with app.run_test(size=(120, 42)) as pilot:
        await pilot.pause(0.2)
        await pilot.click("#start")
        await pilot.pause(0.3)
        assert not app.busy
        assert "请先" in str(app.query_one("#activity", Static).render())


async def test_invalid_model_keeps_dialog_open(tmp_path):
    app = AutoDLApp(Config(root=str(tmp_path)), tmp_path / "config.json")
    async with app.run_test() as pilot:
        await pilot.press("ctrl+n")
        await pilot.pause()
        app.screen.query_one("#source", Select).value = "url"
        await pilot.pause()
        app.screen.query_one("#model-url", Input).value = "not-a-url"
        app.screen.confirm_model()
        assert isinstance(app.screen, ModelDialog)
        assert not app.downloads.jobs
        await pilot.press("escape")
        await pilot.pause()
        assert not isinstance(app.screen, ModelDialog)


async def test_hf_directory_form_queues_both_shards(tmp_path, monkeypatch):
    app = AutoDLApp(Config(root=str(tmp_path)), tmp_path / "config.json")
    monkeypatch.setattr(app, "start_downloads", lambda: None)

    async def expand(model):
        assert model.url == "https://huggingface.co/a/b/resolve/main/IQ3_S"
        assert model.category == "gguf"
        return [Model(f"https://hf-mirror.com/a/b/resolve/main/IQ3_S/{name}.gguf", f"{name}.gguf", "gguf")
                for name in ("part1", "part2")]

    monkeypatch.setattr(app.downloads, "expand", expand)
    async with app.run_test() as pilot:
        await pilot.press("ctrl+n")
        await pilot.pause()
        app.screen.query_one("#source", Select).value = "huggingface"
        app.screen.query_one("#repo-id", Input).value = "a/b"
        app.screen.query_one("#repo-file", Input).value = "IQ3_S"
        app.screen.query_one("#category", Select).value = "gguf"
        app.screen.confirm_model()
        await pilot.pause()
        assert [j.model.filename for j in app.downloads.jobs] == ["part1.gguf", "part2.gguf"]


async def test_hf_settings_and_global_download_log(tmp_path):
    app = AutoDLApp(Config(root=str(tmp_path)), tmp_path / "config.json")
    async with app.run_test(size=(120, 42)) as pilot:
        await pilot.press("f4")
        await pilot.pause()
        assert not app.query("#config-hfd_threads")
        app.save_settings()
        assert Config.load(tmp_path / "config.json").hf_endpoint == app.config.hf_endpoint
        app.query_one("#tool-select", Select).value = "forge"
        await pilot.pause()
        app.query_one("#log-source", Select).value = "hf-downloads.log"
        app.refresh_log()
        assert str(tmp_path / ".autodl-tui/hf-downloads.log") in str(app.query_one("#log-path", Static).render())
