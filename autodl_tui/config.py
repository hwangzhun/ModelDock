from __future__ import annotations

import json
import os
import re
import shlex
import sys
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from urllib.parse import urlsplit

CATEGORIES = (
    "checkpoints", "diffusion_models", "text_encoders", "vae", "loras",
    "controlnet", "clip_vision", "upscale_models", "embeddings", "gguf",
)

DOMESTIC_SOURCES = {
    "repo": "https://gh-proxy.com/https://github.com/Comfy-Org/ComfyUI.git",
    "repo_fallback": "https://gitclone.com/github.com/comfyanonymous/ComfyUI.git",
    "hf_endpoint": "https://hf-mirror.com",
    "pip_index": "auto",
}
OFFICIAL_SOURCES = {
    "repo": "https://github.com/Comfy-Org/ComfyUI.git",
    "repo_fallback": "",
    "hf_endpoint": "https://huggingface.co",
    "pip_index": "https://pypi.org/simple",
}


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".tmp-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def validate_url(value: str) -> str:
    parsed = urlsplit(value)
    if parsed.scheme not in ("http", "https") or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError("请输入不含账号密码的 HTTP(S) 地址")
    if any(ord(c) < 32 for c in value):
        raise ValueError("地址不能包含控制字符")
    return value


def redact(value: str) -> str:
    for name in ("HF_TOKEN", "CIVITAI_TOKEN"):
        token = os.environ.get(name)
        if token:
            value = value.replace(token, "***")
    value = re.sub(r"(?i)([?&](?:token|access_token|key|api_key)=)[^&\s]+", r"\1***", value)
    value = re.sub(r"\bhf_[a-zA-Z0-9]+", "hf_***", value)
    return value


@dataclass
class Config:
    root: str = field(default_factory=lambda: "/root/autodl-tmp" if os.path.isdir("/root/autodl-tmp")
                      else str(Path.home() / "ai-workspace"))
    python: str = field(default_factory=lambda: getattr(sys, "_base_executable", sys.executable))
    repo: str = DOMESTIC_SOURCES["repo"]
    repo_fallback: str = DOMESTIC_SOURCES["repo_fallback"]
    host: str = "0.0.0.0"
    port: int = 6006
    hf_endpoint: str = DOMESTIC_SOURCES["hf_endpoint"]
    pip_index: str = DOMESTIC_SOURCES["pip_index"]
    hfd_threads: int = 8
    torch_index: str = ""
    extra_args: str = ""

    version: int = 2
    network: str = "domestic"
    tools: dict = field(default_factory=dict)
    install_dir: str = ""
    backend: str = "auto"
    model: str = ""
    context: int = 4096
    gpu_layers: int = 0
    _tool_id: str = field(default="comfyui", repr=False, compare=False)

    def for_tool(self, tool_id: str) -> Config:
        from .adapters import TOOLS
        if tool_id == "comfyui":
            return self
        adapter = TOOLS[tool_id]
        values = {"port": adapter.port, "repo": adapter.repo,
                  "repo_fallback": "", "install_dir": str(self.base / adapter.directory),
                  "extra_args": "", "torch_index": "", "backend": "auto", "model": "",
                  "context": 4096, "gpu_layers": 0, "python": self.python}
        if self.network == "domestic":
            values["repo"] = "https://gh-proxy.com/" + adapter.repo
        overrides = self.tools.get(tool_id, {})
        allowed = {"port", "repo", "repo_fallback", "install_dir", "extra_args", "torch_index",
                   "backend", "model", "context", "gpu_layers", "python", "host"}
        if not isinstance(overrides, dict) or set(overrides) - allowed:
            raise ValueError("工具配置含有无效字段")
        values.update(overrides)
        from dataclasses import replace
        return replace(self, **values, _tool_id=tool_id)

    @property
    def base(self) -> Path:
        return Path(self.root).expanduser().resolve()

    @property
    def comfy(self) -> Path:
        return Path(self.install_dir).expanduser().resolve() if self.install_dir else self.base / "ComfyUI"

    @property
    def models(self) -> Path:
        return self.base / "models"

    @property
    def state(self) -> Path:
        return self.base / ".autodl-tui"

    @property
    def service_state(self) -> Path:
        return self.state if self._tool_id == "comfyui" else self.state / "tools" / self._tool_id

    @property
    def interpreter(self) -> Path:
        return self.comfy / ".venv" / "bin" / "python"

    def validate(self) -> None:
        if not isinstance(self.hfd_threads, int) or isinstance(self.hfd_threads, bool) or not 1 <= self.hfd_threads <= 10:
            raise ValueError("hfd 每文件连接数应为 1–10 的整数")
        if not self.root.strip() or not Path(self.root).expanduser().is_absolute():
            raise ValueError("工作目录必须是绝对路径")
        if not isinstance(self.port, int) or isinstance(self.port, bool) or not 1 <= self.port <= 65535:
            raise ValueError("端口应在 1–65535 之间")
        if self.host not in ("0.0.0.0", "127.0.0.1"):
            raise ValueError("监听地址请选择 0.0.0.0 或 127.0.0.1")
        if not Path(self.python).is_absolute() or not Path(self.python).is_file() or not os.access(self.python, os.X_OK):
            raise ValueError("请选择存在且可执行的 Python 解释器（绝对路径）")
        validate_url(self.repo)
        validate_url(self.hf_endpoint)
        parsed = urlsplit(self.hf_endpoint)
        if parsed.path not in ("", "/") or parsed.query or parsed.fragment:
            raise ValueError("HF 镜像只填写站点地址，不包含路径或参数")
        if self.pip_index and self.pip_index != "auto":
            validate_url(self.pip_index)
        for value in (self.repo_fallback, self.torch_index):
            if value:
                validate_url(value)
        if self.install_dir and not Path(self.install_dir).expanduser().is_absolute():
            raise ValueError("安装目录必须是绝对路径")
        if self.backend not in ("auto", "cuda", "hip", "cpu"):
            raise ValueError("后端应为 auto / cuda / hip / cpu")
        if (not isinstance(self.context, int) or not isinstance(self.gpu_layers, int)
                or self.context < 1 or self.gpu_layers < -1):
            raise ValueError("上下文必须大于零，GPU 层数必须为 -1 或非负数")
        if self.network not in ("domestic", "official"):
            raise ValueError("无效的网络方案")
        args = shlex.split(self.extra_args)
        reserved = ("--listen", "--port", "--host", "--server-name", "--extra-model-paths-config",
                    "--ckpt-dir", "--vae-dir", "--lora-dir", "--embeddings-dir",
                    "--model", "-m", "--ctx-size", "-c", "--n-gpu-layers", "-ngl")
        if any(a.split("=", 1)[0] in reserved or any(
                a.startswith(flag) and a != flag for flag in ("-m", "-c", "-ngl")) for a in args):
            raise ValueError("附加参数不能覆盖监听地址、端口和模型路径")

    @classmethod
    def load(cls, path: Path) -> Config:
        if not path.exists():
            return cls()
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("配置必须是 JSON 对象")
        config = cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__ and not k.startswith("_")})
        config.validate()
        if data.get("version", 1) not in (1, 2):
            raise ValueError("不支持此配置版本")
        if data.get("version", 1) == 1:
            config.network = "official" if config.hf_endpoint == OFFICIAL_SOURCES["hf_endpoint"] else "domestic"
            config.version = 2
            if "root" not in data:
                config.root = str(Path.home() / "autodl-workspace")
        if not isinstance(config.tools, dict) or any(k not in ("forge", "a1111", "llamacpp") for k in config.tools):
            raise ValueError("无效的工具配置")
        for tool_id in config.tools:
            config.for_tool(tool_id).validate()
        return config

    def save(self, path: Path) -> None:
        self.validate()
        for tool_id in self.tools:
            self.for_tool(tool_id).validate()
        if path.exists():
            previous = json.loads(path.read_text())
            if previous.get("version", 1) == 1:
                backup = path.with_suffix(path.suffix + ".v1.bak")
                if not backup.exists():
                    atomic_json(backup, previous)
        data = asdict(self)
        data.pop("_tool_id", None)
        atomic_json(path, data)


def config_path() -> Path:
    return Path(os.environ.get("XDG_CONFIG_HOME", str(Path.home() / ".config"))) / "modeldock/config.json"


def legacy_config_path() -> Path:
    base = config_path().parent.parent
    previous = base / "ai-tui/config.json"
    return previous if previous.exists() else base / "autodl-tui/config.json"
