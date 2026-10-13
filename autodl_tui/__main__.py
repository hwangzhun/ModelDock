from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from .config import DOMESTIC_SOURCES, OFFICIAL_SOURCES, Config, atomic_json, config_path, legacy_config_path


def main() -> None:
    parser = argparse.ArgumentParser(description="ModelDock · 通用云服务器 AI 工具工作台")
    parser.add_argument("--config", type=Path, default=config_path(), help="配置文件路径")
    parser.add_argument("--root", type=Path, help="覆盖本次工作目录")
    parser.add_argument("--version", action="version", version="ModelDock 0.3.0")
    args = parser.parse_args()
    try:
        source = args.config
        if source == config_path() and not source.exists() and legacy_config_path().exists():
            source = legacy_config_path()
            backup = source.with_suffix(".json.v1.bak")
            if not backup.exists():
                atomic_json(backup, json.loads(source.read_text()))
        config = Config.load(source)
        if not source.exists():
            network = os.environ.get("MODELDOCK_NETWORK", os.environ.get("AI_TUI_NETWORK", ""))
            if network not in ("domestic", "official"):
                if not sys.stdin.isatty():
                    raise ValueError("首次启动请设置 MODELDOCK_NETWORK=domestic 或 official，或在交互终端选择")
                answer = input("首次网络设置：1 国内加速 / 2 官方源 [1]: ").strip()
                if answer not in ("", "1", "2"):
                    raise ValueError("请选择 1 或 2")
                network = "official" if answer == "2" else "domestic"
            config.network = network
            for key, value in (OFFICIAL_SOURCES if network == "official" else DOMESTIC_SOURCES).items():
                setattr(config, key, value)
        if args.root:
            config.root = str(args.root.expanduser().resolve())
        config.validate()
        if source != args.config or not args.config.exists() or json.loads(source.read_text()).get("version", 1) == 1:
            config.save(args.config)
        from .app import AutoDLApp
        AutoDLApp(config, args.config).run()
    except (ValueError, OSError, TypeError, RuntimeError) as error:
        print(f"启动失败：{error}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
