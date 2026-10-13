from __future__ import annotations

import csv
import json
import shutil
import subprocess
from pathlib import Path

import psutil


def size(value: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(value) < 1024:
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} PiB"


def snapshot(root: Path) -> dict:
    ancestor = root
    while not ancestor.exists():
        ancestor = ancestor.parent
    memory = psutil.virtual_memory()
    disk = shutil.disk_usage(ancestor)
    gpu = []
    gpu_error = "未检测到 NVIDIA GPU"
    try:
        result = subprocess.run([
            "nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
            "--format=csv,noheader,nounits",
        ], capture_output=True, text=True, timeout=3, check=True)
        for row in csv.reader(result.stdout.splitlines()):
            if len(row) == 6:
                name, use, used, total, temp, power = [x.strip() for x in row]
                gpu.append(f"{name}  ·  {use}%\n显存 {used} / {total} MiB  ·  {temp}°C  ·  {power} W")
    except (OSError, subprocess.SubprocessError):
        gpu_error = "NVIDIA GPU 不可用 / nvidia-smi 未安装"
    if not gpu:
        try:
            result = subprocess.run(["rocm-smi", "--showproductname", "--showuse", "--showmeminfo", "vram",
                                     "--showtemp", "--showpower", "--json"],
                                    capture_output=True, text=True, timeout=3, check=True)
            data = json.loads(result.stdout)
            for card, metrics in data.items():
                if isinstance(metrics, dict):
                    gpu.append(card + " · AMD\n" + " · ".join(f"{key}: {value}" for key, value in metrics.items()))
        except (OSError, ValueError, subprocess.SubprocessError):
            gpu_error = "GPU 指标不可用 / nvidia-smi、rocm-smi 未安装或设备不可见"
    net = psutil.net_io_counters()
    return {"cpu": psutil.cpu_percent(), "memory": memory, "disk": disk,
            "gpu": "\n\n".join(gpu) or gpu_error, "net": net}
