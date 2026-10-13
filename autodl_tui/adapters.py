"""Built-in deployment and launch strategies. Add tools here, not in the UI."""
from __future__ import annotations

import json
import os
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from .environment import capabilities, python_version

if TYPE_CHECKING:
    from .config import Config
    from .service import Service


@dataclass(frozen=True)
class ToolAdapter:
    id: str
    name: str
    repo: str
    directory: str
    port: int
    entry: str
    required: tuple[str, ...]
    health_path: str

    def check(self, config: Config) -> list[str]:
        host = capabilities(config.base)
        problems = []
        if host['system'] != 'Linux':
            problems.append('需要 Linux 服务器')
        if not host['writable']:
            problems.append('工作目录不可写，请更换目录或调整权限')
        if not host['git']:
            problems.append('缺少 git，请通过系统包管理器安装')
        if self.id == 'llamacpp':
            for key in ('cmake', 'compiler'):
                if not host[key]:
                    problems.append(f'缺少 {key}，请安装 CMake 和 C++ 编译工具链')
            try:
                self.backend(config)
            except ValueError as error:
                problems.append(str(error))
        else:
            try:
                version = python_version(config.python)
                if self.id in ('forge', 'a1111') and version != (3, 10):
                    problems.append('WebUI 需要独立的 Python 3.10，请在工具设置中指定；TUI 可使用其他 Python')
                elif version < (3, 10):
                    problems.append('需要 Python 3.10+')
            except ValueError as error:
                problems.append(str(error))
            if config.backend == 'cuda' and not host['nvidia']:
                problems.append('未检测到 NVIDIA 驱动，请先配置 CUDA 环境')
            if config.backend == 'hip' and not host['amd']:
                problems.append('未检测到 AMD ROCm 设备，请先配置 ROCm 环境')
        return problems

    def runtime_backend(self, config: Config) -> str:
        host = capabilities(config.base)
        if config.backend != 'auto':
            return config.backend
        return 'hip' if host['amd'] else ('cuda' if host['nvidia'] else 'cpu')

    def backend(self, config: Config) -> str:
        host = capabilities(config.base)
        if config.backend == 'auto':
            return 'hip' if host['hip'] else ('cuda' if host['cuda'] else 'cpu')
        if config.backend in ('cuda', 'hip') and not host[config.backend]:
            raise ValueError(f'缺少 {config.backend} 编译工具链，请安装 nvcc 或 hipcc，或选择 CPU')
        return config.backend

    def launch(self, service: Service) -> list[str]:
        c = service.config
        if self.id == 'comfyui':
            args = [str(c.interpreter), '-u', str(c.comfy / self.entry), '--listen', c.host,
                    '--port', str(c.port), '--extra-model-paths-config', str(service.write_model_paths())]
            if self.runtime_backend(c) == 'cpu':
                args += ['--cpu']
        elif self.id in ('forge', 'a1111'):
            args = [str(c.interpreter), '-u', str(c.comfy / self.entry), '--skip-install',
                    '--port', str(c.port), '--server-name', c.host, '--no-download-sd-model']
            if c.host == '0.0.0.0':
                args += ['--listen']
            for flag, category in (('--ckpt-dir', 'checkpoints'), ('--vae-dir', 'vae'),
                                   ('--lora-dir', 'loras'), ('--embeddings-dir', 'embeddings')):
                path = c.models / category
                path.mkdir(parents=True, exist_ok=True)
                args += [flag, str(path)]
            if self.runtime_backend(c) == 'cpu':
                args += ['--use-cpu', 'all', '--skip-torch-cuda-test']
        else:
            model = Path(c.model).expanduser()
            if not c.model or not model.is_file() or model.suffix.lower() != '.gguf':
                raise ValueError('请先选择存在的 GGUF 模型文件（分片模型选择第一片）')
            metadata = c.service_state / 'build.json'
            if metadata.exists() and c.backend in ('cuda', 'hip'):
                if json.loads(metadata.read_text()).get('backend') != c.backend:
                    raise ValueError('构建后端与设置不同，请重新部署 / 修复')
            args = [str(c.comfy / self.entry), '--host', c.host, '--port', str(c.port),
                    '-m', str(model.resolve()), '-c', str(c.context),
                    '-ngl', str(0 if c.backend == 'cpu' else c.gpu_layers)]
        return [*args, *shlex.split(c.extra_args)]

    async def deploy(self, service: Service) -> None:
        c = service.config
        if self.id == 'comfyui':
            await service.deploy_comfy()
            return
        problems = self.check(c)
        if problems:
            raise RuntimeError('\n'.join(problems))
        await service.clone_source()
        if self.id == 'llamacpp':
            backend = self.backend(c)
            build = c.comfy / 'build'
            await service.command(['cmake', '-S', str(c.comfy), '-B', str(build),
                                   '-DCMAKE_BUILD_TYPE=Release',
                                   f'-DGGML_CUDA={"ON" if backend == "cuda" else "OFF"}',
                                   f'-DGGML_HIP={"ON" if backend == "hip" else "OFF"}'])
            await service.command(['cmake', '--build', str(build), '--config', 'Release',
                                   '--target', 'llama-server', '-j', str(min(8, os.cpu_count() or 1))])
            from .config import atomic_json
            atomic_json(c.service_state / 'build.json', {'backend': backend})
        else:
            await service.select_pip_index()
            if not c.interpreter.exists():
                await service.command([c.python, '-m', 'venv', str(c.comfy / '.venv')])
            await service.prepare_torch()
            # Upstream prepares pinned requirements and auxiliary repositories, then exits.
            args = [str(c.interpreter), str(c.comfy / 'launch.py'), '--exit']
            if self.runtime_backend(c) == 'cpu':
                args += ['--skip-torch-cuda-test', '--use-cpu', 'all']
            await service.command(args, cwd=c.comfy)
            await service.verify_torch()
        service.log(f'{self.name} 部署完成')


TOOLS = {tool.id: tool for tool in (
    ToolAdapter('comfyui', 'ComfyUI', 'https://github.com/Comfy-Org/ComfyUI.git', 'ComfyUI',
                6006, 'main.py', ('main.py', 'requirements.txt'), '/system_stats'),
    ToolAdapter('forge', 'Forge', 'https://github.com/lllyasviel/stable-diffusion-webui-forge.git',
                'Forge', 7860, 'launch.py', ('launch.py', 'requirements_versions.txt'), '/'),
    ToolAdapter('a1111', 'AUTOMATIC1111', 'https://github.com/AUTOMATIC1111/stable-diffusion-webui.git',
                'AUTOMATIC1111', 7861, 'launch.py', ('launch.py', 'requirements_versions.txt'), '/'),
    ToolAdapter('llamacpp', 'llama.cpp', 'https://github.com/ggml-org/llama.cpp.git',
                'llama.cpp', 8080, 'build/bin/llama-server', ('CMakeLists.txt',), '/health'),
)}
