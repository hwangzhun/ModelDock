"""Read-only host capability checks; never install system packages."""
from __future__ import annotations

import os
import platform
import shutil
import subprocess
from pathlib import Path


def python_version(python: str) -> tuple[int, int]:
    try:
        result = subprocess.run([python, "-c", "import sys; print('%s.%s' % sys.version_info[:2])"],
                                capture_output=True, text=True, timeout=10, check=True)
        return tuple(map(int, result.stdout.strip().split('.')))  # type: ignore[return-value]
    except (OSError, ValueError, subprocess.SubprocessError):
        raise ValueError("无法检测 Python，请选择有效的解释器") from None


def capabilities(root: Path) -> dict:
    ancestor = root
    while not ancestor.exists():
        ancestor = ancestor.parent
    return {
        "system": platform.system(), "arch": platform.machine(),
        "writable": os.access(ancestor, os.W_OK),
        "nvidia": bool(shutil.which("nvidia-smi") or Path("/dev/nvidiactl").exists()),
        "amd": bool(shutil.which("rocm-smi") or shutil.which("amd-smi") or Path('/dev/kfd').exists()),
        "cuda": shutil.which("nvcc"), "hip": shutil.which("hipcc"),
        "cmake": shutil.which("cmake"), "compiler": shutil.which("c++"),
        "git": shutil.which("git"),
    }
