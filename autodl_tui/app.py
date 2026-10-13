from __future__ import annotations

import asyncio
import os
import time
from dataclasses import replace
from pathlib import Path

from rich.text import Text
from textual import on
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import (
    Button,
    DataTable,
    Footer,
    Header,
    Input,
    Label,
    RichLog,
    Select,
    Static,
    Switch,
    TabbedContent,
    TabPane,
)

from .adapters import TOOLS
from .config import CATEGORIES, DOMESTIC_SOURCES, OFFICIAL_SOURCES, Config, redact
from .downloads import Downloads, Model, repository_url
from .environment import capabilities
from .monitor import size, snapshot
from .service import Service, tail, workspace_lock


class ModelDialog(ModalScreen[Model | None]):
    BINDINGS = [("escape", "dismiss(None)", "取消")]

    def compose(self) -> ComposeResult:
        with VerticalScroll(id="dialog"):
            yield Label("添加模型", classes="dialog-title")
            yield Select([("ModelScope 魔搭（国内源）", "modelscope"),
                          ("Hugging Face 仓库（使用设置中的镜像）", "huggingface"),
                          ("直链 / Civitai", "url")], value="modelscope", allow_blank=False, id="source")
            yield Label("文件下载地址（直链模式）")
            yield Input(placeholder="https://…/model.safetensors 或 HF /tree/ 目录链接", id="model-url")
            yield Label("仓库 ID / 文件路径（仓库模式）")
            yield Input(placeholder="例如：Qwen/Qwen-Image", id="repo-id")
            yield Input(placeholder="文件路径；HF 也支持目录，例如 IQ3_S（展开所有文件）", id="repo-file")
            yield Input(placeholder="版本（留空：HF main / ModelScope master）", id="revision")
            yield Label("保存分类 / 文件名")
            yield Select([(name, name) for name in CATEGORIES], value="checkpoints", allow_blank=False,
                         id="category")
            yield Input(placeholder="留空从链接推断；Civitai 请填写 model.safetensors", id="filename")
            yield Input(placeholder="SHA256（可选，用于完整性校验）", id="sha256")
            yield Static("", id="dialog-error", classes="error")
            with Horizontal(classes="buttons"):
                yield Button("加入下载队列", variant="primary", id="confirm-model")
                yield Button("取消", id="cancel-dialog")

    def on_mount(self) -> None:
        self.toggle_source()

    @on(Select.Changed, "#source")
    def toggle_source(self) -> None:
        direct = self.query_one("#source", Select).value == "url"
        self.query_one("#model-url", Input).disabled = not direct
        for name in ("repo-id", "repo-file", "revision"):
            self.query_one(f"#{name}", Input).disabled = direct

    @on(Button.Pressed, "#cancel-dialog")
    def cancel_dialog(self) -> None:
        self.dismiss(None)

    @on(Button.Pressed, "#confirm-model")
    def confirm_model(self) -> None:
        def value(name: str) -> str:
            return self.query_one(f"#{name}", Input).value
        try:
            source = self.query_one("#source", Select).value
            url = value("model-url") if source == "url" else repository_url(
                str(source), value("repo-id"), value("repo-file"), value("revision"))
            model = Model.from_url(url, str(self.query_one("#category", Select).value),
                                   value("filename"), value("sha256"))
            self.dismiss(model)
        except ValueError as error:
            self.query_one("#dialog-error", Static).update(str(error))


class PathDialog(ModalScreen[str | None]):
    BINDINGS = [("escape", "dismiss(None)", "取消")]

    def __init__(self, title: str, default: str):
        super().__init__()
        self.label, self.default = title, default

    def compose(self) -> ComposeResult:
        with Vertical(id="path-dialog"):
            yield Label(self.label, classes="dialog-title")
            yield Input(value=self.default, id="manifest-path")
            yield Static("清单保留原始下载链接；请勿在链接参数中附带令牌。", classes="muted")
            with Horizontal(classes="buttons"):
                yield Button("确定", variant="primary", id="confirm-path")
                yield Button("取消", id="cancel-path")

    @on(Button.Pressed)
    def choose(self, event: Button.Pressed) -> None:
        self.dismiss(self.query_one(Input).value if event.button.id == "confirm-path" else None)


TOOL_FIELDS = ("python", "repo", "repo_fallback", "host", "port", "torch_index", "extra_args",
               "install_dir", "backend", "model", "context", "gpu_layers")
GLOBAL_FIELDS = ("root", "hf_endpoint", "pip_index")


class AutoDLApp(App):
    TITLE = "ModelDock · 模型坞"
    SUB_TITLE = "实例工作台"
    CSS_PATH = "app.tcss"
    BINDINGS = [("ctrl+q", "quit", "退出"), ("f1", "tab('overview')", "概览"),
                ("f2", "tab('models')", "模型"), ("f3", "tab('logs')", "日志"),
                ("f4", "tab('settings')", "设置"), ("ctrl+n", "add_model", "添加模型")]

    def __init__(self, config: Config, config_file: Path):
        super().__init__()
        self.config = config
        self.config_file = config_file
        self.services = {key: Service(config, key) for key in TOOLS}
        self.tool_id = "comfyui"
        self.service = self.services[self.tool_id]
        self.view_refreshing = False
        self.console_lock = workspace_lock(config, "console")
        self.console_lock.__enter__()
        try:
            self.downloads = Downloads(config)
        except Exception:
            self.console_lock.__exit__(None, None, None)
            raise
        self.busy = False
        self.operation_worker = None
        self.refreshing = False
        self.last_log = ""
        self.last_rows: list = []
        self.previous_net = None

    @property
    def main_screen(self):
        return self.screen_stack[0]

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Static("MODELDOCK     云服务器 · 多工具工作台", id="brand")
        with TabbedContent(initial="overview"):
            with TabPane("工作台", id="overview"):
                with VerticalScroll():
                    yield Select([(tool.name, key) for key, tool in TOOLS.items()],
                                 value=self.tool_id, allow_blank=False, id="tool-select")
                    yield Static("", id="tools-summary", markup=False)
                    yield Static("", id="compatibility", markup=False, classes="muted")
                    with Horizontal(id="hero"):
                        with Vertical(id="service-card", classes="card"):
                            yield Static("ComfyUI SERVICE", id="tool-title", classes="eyebrow")
                            yield Static("○ 检查中", id="service-status")
                            yield Static("", id="service-detail", markup=False)
                        with Vertical(id="workspace-card", classes="card"):
                            yield Static("WORKSPACE / 数据盘", classes="eyebrow")
                            yield Static(str(self.config.base), id="workspace-path", markup=False)
                            yield Static("模型与服务分开存储，可复用模型清单。", classes="muted")
                    with Horizontal(classes="buttons", id="service-buttons"):
                        yield Button("部署 / 修复", id="deploy")
                        yield Button("▶ 一键启动", variant="primary", id="start")
                        yield Button("■ 停止", id="stop")
                        yield Button("重启", id="restart")
                        yield Button("取消任务", id="cancel-operation", disabled=True)
                    yield Static("就绪 · 首次使用先检查设置，再部署所选工具。", id="activity", markup=False)
                    with Horizontal(id="metrics"):
                        with Vertical(classes="card metric"):
                            yield Static("CPU / 处理器", classes="eyebrow")
                            yield Static("—", id="cpu-metric")
                        with Vertical(classes="card metric"):
                            yield Static("RAM / 内存", classes="eyebrow")
                            yield Static("—", id="ram-metric")
                        with Vertical(classes="card metric"):
                            yield Static("DISK / 数据盘", classes="eyebrow")
                            yield Static("—", id="disk-metric")
                    with Vertical(classes="card", id="gpu-card"):
                        yield Static("GPU / NVIDIA · AMD", classes="eyebrow")
                        yield Static("检测中…", id="gpu-metric", markup=False)
                    yield Static("", id="network-metric")
                    yield Static("01  部署环境    →    02  添加模型    →    03  启动服务", id="steps")
                    yield Static("", id="access-hint", classes="hint", markup=False)
            with TabPane("模型下载", id="models"):
                yield Static("模型库 / 下载队列", classes="section-title")
                yield Static("国内优先：ModelScope · HF 镜像 · 自定义直链 / Civitai", classes="muted")
                with Horizontal(classes="buttons"):
                    yield Button("＋ 添加模型", variant="primary", id="add-model")
                    yield Button("全部继续", id="resume-all")
                    yield Button("全部暂停", id="pause-all")
                    yield Button("导入清单", id="import-manifest")
                    yield Button("导出清单", id="export-manifest")
                yield DataTable(id="download-table", cursor_type="row", zebra_stripes=True)
                yield Static("还没有模型。添加一个下载，或导入上次保存的清单。", id="download-detail", markup=False)
                with Horizontal(classes="buttons"):
                    yield Button("继续 / 重试选中项", id="retry-selected")
                    yield Button("暂停选中项", id="pause-selected")
                    yield Button("移出清单", id="remove-selected")
                yield Static("HF 使用 hf download 下载 · 文件依次下载 · 暂停保留续传文件 · 移出清单不删除文件", classes="hint")
            with TabPane("实时日志", id="logs"):
                with Horizontal(id="log-toolbar"):
                    yield Select([("当前工具运行日志", "comfyui.log"), ("部署 / 操作日志", "operations.log"),
                                  ("HF / hf download 下载日志", "hf-downloads.log")],
                                 value="comfyui.log", allow_blank=False, id="log-source")
                    yield Label("自动滚动")
                    yield Switch(value=True, id="log-follow")
                yield Static("", id="log-path", markup=False, classes="muted")
                yield RichLog(id="log-view", wrap=True, markup=False, highlight=False, max_lines=2000)
            with TabPane("设置", id="settings"):
                with VerticalScroll(id="settings-form"):
                    yield Static("全局配置 / 当前工具配置", classes="section-title")
                    yield Static("设置保存后生效。更换工作目录会切换到该目录的模型清单。", classes="muted")
                    with Horizontal(classes="buttons"):
                        yield Button("填入国内源", variant="primary", id="domestic-sources")
                        yield Button("填入官方源", id="official-sources")
                    yield Static("国内预设：GH-Proxy + GitClone 备用 / HF-Mirror / pip 三源自动测速。填入后点击保存。",
                                 classes="hint")
                    for key, label in (
                        ("root", "数据盘目录"), ("python", "当前工具 Python（WebUI 使用 Python 3.10）"),
                        ("repo", "当前工具 Git 镜像 / 仓库地址"),
                        ("repo_fallback", "当前工具备用 Git 镜像（失败自动切换，留空禁用）"),
                        ("host", "监听地址：0.0.0.0 / 127.0.0.1"),
                        ("port", "当前工具服务端口"),
                        ("hf_endpoint", "Hugging Face 站点 / 镜像，例如 https://hf-mirror.com"),
                        ("pip_index", "pip 镜像：auto 测速阿里 / 腾讯 / 清华；也可填固定地址或留空"),
                        ("torch_index", "PyTorch wheel 源（留空优先复用已有 PyTorch）"),
                        ("extra_args", "当前工具附加参数"),
                        ("install_dir", "当前工具安装目录（ComfyUI 留空使用默认）"),
                        ("backend", "计算后端：auto / cuda / hip / cpu"),
                        ("model", "llama.cpp GGUF 文件绝对路径"),
                        ("context", "llama.cpp 上下文长度"),
                        ("gpu_layers", "llama.cpp GPU 卸载层数（0 为 CPU，-1 为全部）"),
                    ):
                        yield Label(label)
                        yield Input(value=str(getattr(self.config, key)), id=f"config-{key}")
                    yield Select([("手动填写 GGUF 路径", "" )], value="", allow_blank=False, id="gguf-select")
                    yield Static("访问令牌 · 仅当前会话，不写入配置", classes="section-title")
                    yield Label("Hugging Face Token（也可用 HF_TOKEN 环境变量）")
                    yield Input(value=os.environ.get("HF_TOKEN", ""), password=True, id="hf-token")
                    yield Label("Civitai Token（也可用 CIVITAI_TOKEN 环境变量）")
                    yield Input(value=os.environ.get("CIVITAI_TOKEN", ""), password=True, id="civitai-token")
                    yield Static("HF 令牌只发送到官方域名，不会发给镜像。ModelScope 当前支持公开文件。",
                                 classes="hint")
                    yield Button("保存设置", variant="primary", id="save-settings")
                    yield Static(str(self.config_file), classes="muted", markup=False)
        yield Footer()

    def on_mount(self) -> None:
        self.main_screen.query_one("#download-table", DataTable).add_columns("模型文件", "分类", "状态", "进度", "速度")
        self.set_interval(1, self.refresh_view)
        self.set_interval(3, self.refresh_monitor)
        self.run_worker(self.refresh_monitor())
        self.run_worker(self.refresh_view())
        self.update_tool_fields()
        self.run_worker(self.check_environment())

    def on_unmount(self) -> None:
        self.console_lock.__exit__(None, None, None)

    def action_tab(self, name: str) -> None:
        self.main_screen.query_one(TabbedContent).active = name

    def action_quit(self) -> None:
        if self.busy:
            self.notify("请先取消或等待部署 / 服务任务完成，再退出", severity="warning")
            return
        self.downloads.pause()
        self.exit()

    def message(self, text: str, error: bool = False) -> None:
        self.main_screen.query_one("#activity", Static).update(redact(text))
        self.notify(redact(text), severity="error" if error else "information", timeout=7)

    async def refresh_monitor(self) -> None:
        if self.refreshing:
            return
        self.refreshing = True
        try:
            data = await asyncio.to_thread(snapshot, self.config.base)
            memory, disk = data["memory"], data["disk"]
            self.main_screen.query_one("#cpu-metric", Static).update(f"{data['cpu']:.0f}%\n{os.cpu_count()} 个逻辑核心")
            self.main_screen.query_one("#ram-metric", Static).update(f"{memory.percent:.0f}%\n{size(memory.used)} / {size(memory.total)}")
            self.main_screen.query_one("#disk-metric", Static).update(f"剩余 {size(disk.free)}\n总计 {size(disk.total)}")
            self.main_screen.query_one("#gpu-metric", Static).update(data["gpu"])
            now, net = time.monotonic(), data["net"]
            if self.previous_net and net:
                before, old = self.previous_net
                seconds = max(0.1, now - before)
                self.main_screen.query_one("#network-metric", Static).update(
                    f"NETWORK   ↓ {size(max(0, net.bytes_recv - old.bytes_recv) / seconds)}/s"
                    f"     ↑ {size(max(0, net.bytes_sent - old.bytes_sent) / seconds)}/s")
            if net:
                self.previous_net = now, net
        except OSError as error:
            self.main_screen.query_one("#gpu-metric", Static).update(f"监控暂不可用：{error}")
        finally:
            self.refreshing = False

    async def refresh_view(self) -> None:
        if self.view_refreshing:
            return
        self.view_refreshing = True
        try:
            services = list(self.services.values())
            health = await asyncio.gather(*(service.health() for service in services))
            summaries = []
            for service, ready in zip(services, health):
                proc = service.managed_process()
                if not proc and (service.config.service_state / "service.json").exists():
                    service.failed = service.failed or "进程已退出或记录身份不匹配"
                status = "● 就绪" if ready and proc else ("◌ 启动中 / 未就绪" if proc else (
                    "● 外部服务" if ready else ("× 失败" if service.failed else "○ 未启动")))
                c = service.config
                summaries.append(f"{service.adapter.name}: {status} :{c.port}")
                if service is self.service:
                    self.main_screen.query_one("#service-status", Static).update(status)
                    self.main_screen.query_one("#service-detail", Static).update(
                        f"{'已部署' if service.installed else '等待部署'} · 端口 {c.port}"
                        + (f" · PID {proc.pid}" if proc else "") + "\n"
                        + (service.failed or f"本机 http://127.0.0.1:{c.port}"))
            self.main_screen.query_one("#tools-summary", Static).update("\n".join(summaries))
        finally:
            self.view_refreshing = False
        rows = []
        for job in self.downloads.jobs:
            progress = f"{job.downloaded / job.total:.0%}" if job.total else "—"
            progress += f" · {size(job.downloaded)}"
            rows.append((job.id, job.model.filename, job.model.category, job.status,
                         progress, f"{size(job.speed)}/s" if job.speed else "—"))
        if rows != self.last_rows:
            table = self.main_screen.query_one("#download-table", DataTable)
            cursor = table.cursor_row
            table.clear()
            for row in rows:
                table.add_row(*(Text(value) for value in row[1:]), key=row[0])
            if rows:
                table.move_cursor(row=min(cursor, len(rows) - 1))
            self.last_rows = rows
        self.show_selected_detail()
        self.refresh_log()
        files = tuple((self.config.models / "gguf").glob("*.gguf"))
        if files != getattr(self, "last_gguf", None):
            self.last_gguf = files
            self.update_gguf_options()

    def refresh_log(self) -> None:
        source = str(self.main_screen.query_one("#log-source", Select).value)
        path = (self.config.state if source == "hf-downloads.log" else self.service.config.service_state) / source
        self.main_screen.query_one("#log-path", Static).update(str(path))
        if not self.main_screen.query_one("#log-follow", Switch).value:
            return
        content = tail(path)
        key = str(path) + content
        if key != self.last_log:
            log = self.main_screen.query_one("#log-view", RichLog)
            log.clear()
            log.write(Text(content))
            log.scroll_end(animate=False)
            self.last_log = key

    def selected_job(self):
        table = self.main_screen.query_one("#download-table", DataTable)
        if not table.row_count:
            return None
        key = str(table.coordinate_to_cell_key(table.cursor_coordinate).row_key.value)
        return next((job for job in self.downloads.jobs if job.id == key), None)

    @on(DataTable.RowHighlighted, "#download-table")
    def show_selected_detail(self) -> None:
        job = self.selected_job()
        if job:
            self.main_screen.query_one("#download-detail", Static).update(
                f"{job.model.filename} → {self.config.models / job.model.category}\n"
                + (job.error or f"状态：{job.status}"))
        else:
            self.main_screen.query_one("#download-detail", Static).update("还没有模型。添加一个下载，或导入上次保存的清单。")

    def action_add_model(self) -> None:
        self.push_screen(ModelDialog(), self.add_model)

    def add_model(self, model: Model | None) -> None:
        if model:
            self.run_worker(self.prepare_models(model), group="model-listing")

    async def prepare_models(self, model: Model) -> None:
        try:
            self.message("正在解析模型文件清单…")
            models = await self.downloads.expand(model)
            count = self.downloads.add(models)
            self.message(f"已加入 {count} 个文件" if count else "同名目标已在清单中，请选中后继续或移除")
            self.start_downloads()
        except Exception as error:
            self.message(redact(str(error)), True)

    def start_downloads(self) -> None:
        if not self.downloads.running:
            self.run_worker(self.download_queue(), group="downloads", exclusive=True)

    async def download_queue(self) -> None:
        try:
            await self.downloads.run()
        except (OSError, RuntimeError) as error:
            self.message(str(error), True)

    def launch_operation(self, name: str) -> None:
        if self.busy:
            return
        self.busy = True
        for button in ("deploy", "start", "stop", "restart", "save-settings"):
            self.main_screen.query_one(f"#{button}", Button).disabled = True
        self.main_screen.query_one("#tool-select", Select).disabled = True
        self.main_screen.query_one("#cancel-operation", Button).disabled = False
        self.operation_worker = self.run_worker(self.operation(name), group="service")

    async def operation(self, name: str) -> None:
        labels = {"deploy": "部署 / 修复", "start": "启动", "stop": "停止", "restart": "重启"}
        self.message(f"正在{labels[name]}，详情见日志")
        service = self.service
        self.main_screen.query_one("#log-source", Select).value = "operations.log" if name == "deploy" else service.runtime_log
        try:
            if name == "restart":
                if self.service.managed_process():
                    await self.service.stop()
                await self.service.start()
            else:
                await getattr(self.service, name)()
            self.message(f"{labels[name]}任务完成" + ("，等待服务就绪" if name in ("start", "restart") else ""))
        except asyncio.CancelledError:
            self.service.log("任务已取消，可重新执行部署修复未完成的环境")
            self.message("任务已取消；已启动的工具可用「停止」关闭")
            raise
        except Exception as error:
            self.service.log(str(error))
            self.message(str(error), True)
        finally:
            self.busy = False
            for button in ("deploy", "start", "stop", "restart", "save-settings"):
                self.main_screen.query_one(f"#{button}", Button).disabled = False
            self.main_screen.query_one("#tool-select", Select).disabled = False
            self.main_screen.query_one("#cancel-operation", Button).disabled = True

    @on(Button.Pressed)
    def clicked(self, event: Button.Pressed) -> None:
        name = event.button.id
        try:
            if name in ("deploy", "start", "stop", "restart"):
                self.launch_operation(name)
            elif name == "cancel-operation" and self.operation_worker:
                self.operation_worker.cancel()
            elif name == "add-model":
                self.action_add_model()
            elif name == "resume-all":
                self.downloads.retry()
                self.start_downloads()
            elif name == "pause-all":
                self.downloads.pause()
            elif name in ("retry-selected", "pause-selected", "remove-selected"):
                job = self.selected_job()
                if not job:
                    return
                if name == "retry-selected":
                    self.downloads.retry(job.id)
                    self.start_downloads()
                elif name == "pause-selected":
                    self.downloads.pause(job.id)
                else:
                    self.downloads.remove(job.id)
            elif name in ("import-manifest", "export-manifest"):
                exporting = name == "export-manifest"
                self.push_screen(PathDialog("导出模型清单" if exporting else "导入模型清单",
                                            str(self.config.base / "models-manifest.json")),
                                 lambda path: self.manifest(path, exporting))
            elif name == "save-settings":
                self.save_settings()
            elif name in ("domestic-sources", "official-sources"):
                sources = DOMESTIC_SOURCES if name == "domestic-sources" else OFFICIAL_SOURCES
                sources = dict(sources)
                adapter = TOOLS[self.tool_id]
                if self.tool_id != "comfyui":
                    sources["repo"] = ("https://gh-proxy.com/" if name == "domestic-sources" else "") + adapter.repo
                    sources["repo_fallback"] = ""
                self.pending_network = "domestic" if name == "domestic-sources" else "official"
                for key, value in sources.items():
                    self.main_screen.query_one(f"#config-{key}", Input).value = value
                self.message("下载源已填入，点击「保存设置」生效")
        except (OSError, ValueError, RuntimeError) as error:
            self.message(str(error), True)

    def manifest(self, value: str | None, exporting: bool) -> None:
        if value is None:
            return
        try:
            if not value.strip():
                raise ValueError("请输入清单路径")
            path = Path(value).expanduser().resolve()
            if exporting:
                if path.exists():
                    raise ValueError("目标文件已存在，请使用新的文件名，以免覆盖原清单")
                self.downloads.export_manifest(path)
                self.message(f"已导出 {path}")
            else:
                count = self.downloads.import_manifest(path)
                self.message(f"已导入 {count} 个模型，点击「全部继续」开始下载")
        except (OSError, ValueError, TypeError, KeyError) as error:
            self.message(str(error), True)

    @on(Select.Changed, "#tool-select")
    def select_tool(self, event: Select.Changed) -> None:
        if str(event.value) not in TOOLS or self.busy:
            return
        self.tool_id = str(event.value)
        self.service = self.services[self.tool_id]
        self.update_tool_fields()
        self.run_worker(self.check_environment(), group="environment", exclusive=True)

    @on(Select.Changed, "#gguf-select")
    def select_gguf(self, event: Select.Changed) -> None:
        if event.value:
            self.main_screen.query_one("#config-model", Input).value = str(event.value)

    def update_tool_fields(self) -> None:
        c = self.service.config
        for key in TOOL_FIELDS:
            self.main_screen.query_one(f"#config-{key}", Input).value = str(getattr(c, key))
        self.main_screen.query_one("#tool-title", Static).update(self.service.adapter.name + " SERVICE")
        logs = self.main_screen.query_one("#log-source", Select)
        logs.set_options([("当前工具运行日志", self.service.runtime_log), ("部署 / 操作日志", "operations.log"),
                          ("HF / hf download 下载日志", "hf-downloads.log")])
        logs.value = self.service.runtime_log
        self.last_log = ""
        self.update_gguf_options()
        port = c.port
        extra = f"\nAPI: http://127.0.0.1:{port}/v1" if self.tool_id == "llamacpp" else ""
        self.main_screen.query_one("#access-hint", Static).update(
            f"监听 {c.host}:{port} · 浏览器 http://127.0.0.1:{port}{extra}\n"
            f"SSH 转发（替换地址）：ssh -N -L {port}:127.0.0.1:{port} 用户@服务器 -p SSH端口\n"
            + ("AutoDL 可在控制台配置自定义服务端口。\n" if os.path.isdir('/root/autodl-tmp') else "")
            + "退出 TUI 保留服务；下载暂停。并行运行前请检查 GPU 显存。")
        for key in ("model", "context", "gpu_layers"):
            self.main_screen.query_one(f"#config-{key}", Input).disabled = self.tool_id != "llamacpp"
        self.main_screen.query_one("#gguf-select", Select).disabled = self.tool_id != "llamacpp"

    def update_gguf_options(self) -> None:
        files = sorted((self.config.models / "gguf").glob("*.gguf"))
        selector = self.main_screen.query_one("#gguf-select", Select)
        options = [("手动填写 GGUF 路径", "")] + [(file.name, str(file)) for file in files if file.is_file()]
        current = selector.value
        selector.set_options(options)
        selector.value = current if current in [value for _, value in options] else ""

    async def check_environment(self) -> None:
        service = self.service
        problems = await asyncio.to_thread(service.adapter.check, service.config)
        host = await asyncio.to_thread(capabilities, self.config.base)
        if service is self.service:
            self.main_screen.query_one("#compatibility", Static).update(
                f"{host['system']} / {host['arch']} · " +
                ("；".join(problems) if problems else "部署前置检查通过；部署时验证计算后端和依赖"))

    def save_settings(self) -> None:
        if self.busy:
            raise ValueError("请先等待当前操作完成")
        global_values = {key: self.main_screen.query_one(f"#config-{key}", Input).value.strip() for key in GLOBAL_FIELDS}
        values = {key: self.main_screen.query_one(f"#config-{key}", Input).value.strip() for key in TOOL_FIELDS}
        for key in ("port", "context", "gpu_layers"):
            values[key] = int(values[key])
        global_changed = any(getattr(self.config, key) != value for key, value in global_values.items())
        if self.service.managed_process() or (global_changed and (
                self.downloads.running or any(service.managed_process() for service in self.services.values()))):
            raise ValueError("请先停止受设置变更影响的服务；修改全局设置前还需暂停下载")
        config = replace(self.config, **global_values, tools=dict(self.config.tools),
                         network=getattr(self, "pending_network", self.config.network))
        if self.tool_id == "comfyui":
            config = replace(config, **values)
        else:
            config.tools[self.tool_id] = values
        # Freeze each tool's defaults so changing ComfyUI does not alter another tool.
        for key in ("forge", "a1111", "llamacpp"):
            if key not in config.tools:
                old = self.config.for_tool(key)
                config.tools[key] = {field: getattr(old, field) for field in TOOL_FIELDS}
        config.validate()
        for tool_id in TOOLS:
            config.for_tool(tool_id).validate()
        ports = [config.for_tool(key).port for key in TOOLS]
        if len(set(ports)) != len(ports):
            raise ValueError("各工具端口必须不同")
        directories = [config.for_tool(key).comfy for key in TOOLS]
        if len(set(directories)) != len(directories):
            raise ValueError("各工具安装目录必须不同")
        new_lock = None
        try:
            if config.base != self.config.base:
                new_lock = workspace_lock(config, "console")
                new_lock.__enter__()
            downloads = Downloads(config) if not self.downloads.running else self.downloads
            config.save(self.config_file)
        except Exception:
            if new_lock:
                new_lock.__exit__(None, None, None)
            raise
        if new_lock:
            self.console_lock.__exit__(None, None, None)
            self.console_lock = new_lock
        previous = self.services
        self.config, self.downloads = config, downloads
        self.downloads.config = config
        self.services = {key: Service(config, key) for key in TOOLS}
        for key, service in self.services.items():
            service.child = previous[key].child
            service.failed = previous[key].failed
        self.service = self.services[self.tool_id]
        for env, selector in (("HF_TOKEN", "hf-token"), ("CIVITAI_TOKEN", "civitai-token")):
            value = self.main_screen.query_one(f"#{selector}", Input).value.strip()
            if value:
                os.environ[env] = value
            else:
                os.environ.pop(env, None)
        self.main_screen.query_one("#workspace-path", Static).update(str(config.base))
        self.last_rows, self.last_log = [], ""
        self.main_screen.query_one("#download-table", DataTable).clear()
        self.update_tool_fields()
        self.run_worker(self.check_environment(), group="environment", exclusive=True)
        self.message("设置已保存；访问令牌仅保留在当前会话")


AIApp = AutoDLApp

ModelDockApp = AutoDLApp
