#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$project_dir"
python_bin="${MODELDOCK_PYTHON:-${AI_TUI_PYTHON:-${AUTODL_PYTHON:-python3}}}"
for arg in "$@"; do
    if [[ "$arg" == --version || "$arg" == --help || "$arg" == -h ]]; then
        exec "$python_bin" -m modeldock "$@"
    fi
done
export AI_TUI_NETWORK="${MODELDOCK_NETWORK:-${AI_TUI_NETWORK:-}}"
bootstrap_index="${MODELDOCK_PIP_INDEX:-${AI_TUI_PIP_INDEX:-${AUTODL_PIP_INDEX:-}}}"
if [[ ! -x .venv/bin/python || ! -f .venv/.modeldock-installed ]]; then
    if [[ -z "$bootstrap_index" ]]; then
        if [[ -z "${AI_TUI_NETWORK:-}" ]]; then
            configured_network="$("$python_bin" - "$@" <<'PYCONFIG'
import json
import os
import sys
from pathlib import Path
base = Path(os.environ.get('XDG_CONFIG_HOME', str(Path.home() / '.config')))
paths = [base / 'modeldock/config.json', base / 'ai-tui/config.json', base / 'autodl-tui/config.json']
for index, arg in enumerate(sys.argv[1:], 1):
    if arg == '--config' and index + 1 < len(sys.argv):
        paths = [Path(sys.argv[index + 1]).expanduser()]
    elif arg.startswith('--config='):
        paths = [Path(arg.split('=', 1)[1]).expanduser()]
for path in paths:
    if path.exists():
        try:
            config = json.loads(path.read_text())
            print(config.get('network', 'official' if config.get('hf_endpoint') == 'https://huggingface.co' else 'domestic'))
        except (ValueError, OSError, AttributeError):
            pass
        break
PYCONFIG
)"
            if [[ -n "$configured_network" ]]; then
                export AI_TUI_NETWORK="$configured_network"
            elif [[ -t 0 ]]; then
                read -r -p '首次网络设置：1 国内加速 / 2 官方源 [1]: ' network_choice
                case "$network_choice" in
                    ''|1) export AI_TUI_NETWORK=domestic ;;
                    2) export AI_TUI_NETWORK=official ;;
                    *) echo '请选择 1 或 2' >&2; exit 1 ;;
                esac
            else
                echo '请设置 AI_TUI_NETWORK=domestic 或 official 后启动' >&2
                exit 1
            fi
        fi
        case "$AI_TUI_NETWORK" in
            domestic) bootstrap_index=https://pypi.tuna.tsinghua.edu.cn/simple ;;
            official) bootstrap_index=https://pypi.org/simple ;;
            *) echo 'AI_TUI_NETWORK 应为 domestic 或 official' >&2; exit 1 ;;
        esac
    fi
fi

if [[ ! -x .venv/bin/python ]]; then
    "$python_bin" -m venv .venv || {
        echo '创建环境失败。请使用 Linux Python 3.10+ 环境，或安装 python3-venv。' >&2
        exit 1
    }
fi
if [[ ! -f .venv/.modeldock-installed ]] || ! .venv/bin/python -c 'import textual, httpx, psutil, modeldock' 2>/dev/null; then
    bootstrap_index="${bootstrap_index:-https://pypi.org/simple}"
    PIP_INDEX_URL="$bootstrap_index" PIP_EXTRA_INDEX_URL="" \
        .venv/bin/python -m pip install --index-url "$bootstrap_index" -e .
    touch .venv/.modeldock-installed
fi
if [[ "${1:-}" == --install-only ]]; then
    echo "ModelDock 安装完成：$project_dir/.venv/bin/modeldock"
    exit 0
fi
exec .venv/bin/python -m modeldock "$@"
