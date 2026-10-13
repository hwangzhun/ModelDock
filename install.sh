#!/usr/bin/env bash
# Download and run ModelDock on the server. Works with curl | bash and saved scripts.
set -euo pipefail

repo="${MODELDOCK_REPO:-https://github.com/hwangzhun/ModelDock.git}"
ref="${MODELDOCK_REF:-}"
install_root="${MODELDOCK_INSTALL_DIR:-${XDG_DATA_HOME:-$HOME/.local/share}/modeldock}"
bin_dir="${MODELDOCK_BIN_DIR:-$HOME/.local/bin}"
update=false
install_only=false
app_args=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --repo|--ref|--install-dir)
            [[ $# -ge 2 && -n "$2" ]] || { echo "$1 需要一个值" >&2; exit 2; }
            case "$1" in
                --repo) repo="$2" ;;
                --ref) ref="$2" ;;
                --install-dir) install_root="$2" ;;
            esac
            shift 2 ;;
        --update) update=true; shift ;;
        --install-only) install_only=true; shift ;;
        --) shift; app_args+=("$@"); break ;;
        --help|-h)
            echo 'ModelDock 远程安装与启动'
            echo 'install.sh [--repo URL] [--ref 分支或标签] [--install-dir 绝对路径] [--update] [--install-only] [-- TUI参数]'
            exit 0 ;;
        *) app_args+=("$1"); shift ;;
    esac
done
[[ "$install_root" == /* && "$install_root" != / && "$bin_dir" == /* && "$bin_dir" != / ]] || { echo '安装目录必须为非根目录的绝对路径' >&2; exit 2; }
[[ "$repo" != -* && "$ref" != -* ]] || { echo '仓库或版本不能以 - 开头' >&2; exit 2; }
command -v git >/dev/null || { echo '缺少 git，请先通过系统包管理器安装 git' >&2; exit 1; }
python_bin="${MODELDOCK_PYTHON:-${AI_TUI_PYTHON:-${AUTODL_PYTHON:-python3}}}"
"$python_bin" -c 'import sys; assert sys.version_info >= (3, 10), "ModelDock 需要 Python 3.10+"'
mkdir -p "$install_root"
lock="$install_root/.installer-lock"
mkdir "$lock" 2>/dev/null || { echo "另一个安装器正在运行；若上次异常终止，请检查后删除 $lock" >&2; exit 1; }
staging=''
cleanup() {
    if [[ -n "$staging" ]]; then rm -rf -- "$staging"; fi
    rmdir -- "$lock" 2>/dev/null || true
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
app="$install_root/app"
if [[ ! -e "$app" ]]; then
    staging="$(mktemp -d "$install_root/.download-XXXXXX")"
    clone_args=(--depth 1)
    if [[ -n "$ref" ]]; then clone_args+=(--branch "$ref"); fi
    git clone "${clone_args[@]}" -- "$repo" "$staging/app"
    [[ -f "$staging/app/run.sh" && -f "$staging/app/pyproject.toml" ]] || {
        echo '下载内容不是有效的 ModelDock 项目' >&2; exit 1;
    }
    mv -- "$staging/app" "$app"
elif [[ ! -d "$app/.git" || ! -f "$app/run.sh" ]]; then
    echo "安装目录已有其他内容，不会覆盖：$app" >&2; exit 1
elif [[ "$(git -C "$app" remote get-url origin)" != "$repo" ]]; then
    echo '已安装仓库与指定地址不同，请使用另一安装目录' >&2; exit 1
fi
if "$update"; then
    [[ -z "$(git -C "$app" status --porcelain)" ]] || {
        echo '安装目录有本地修改，更新已取消，请先处理修改' >&2; exit 1;
    }
    git -C "$app" fetch --depth 1 origin "${ref:-HEAD}"
    git -C "$app" checkout --detach FETCH_HEAD
    rm -f -- "$app/.venv/.modeldock-installed"
elif [[ -n "$ref" && -z "$staging" ]]; then
    echo '已有安装将继续使用当前版本；切换版本请添加 --update' >&2
fi
# Keep the installer lock through environment preparation.
if "$install_only" || [[ -t 0 ]]; then
    bash "$app/run.sh" --install-only
elif { true </dev/tty; } 2>/dev/null; then
    bash "$app/run.sh" --install-only </dev/tty
else
    echo '没有交互终端，无法打开 TUI。请使用 SSH -t，或加 --install-only 安装后在终端运行 modeldock。' >&2
    exit 1
fi
# Build a persistent launcher outside the environment without changing PATH.
mkdir -p "$bin_dir"
launcher="$bin_dir/modeldock"
if [[ -e "$launcher" || -L "$launcher" ]]; then
    if [[ -L "$launcher" ]] || ! grep -q '^# ModelDock managed launcher$' "$launcher"; then
        echo "已有其他启动器，不会覆盖：$launcher；可直接执行 $app/run.sh" >&2
        exit 1
    fi
fi
launcher_tmp="$(mktemp "$bin_dir/.modeldock-XXXXXX")"
printf '#!/usr/bin/env bash\n# ModelDock managed launcher\nexec bash %q "$@"\n' "$app/run.sh" > "$launcher_tmp"
chmod +x "$launcher_tmp"
mv -- "$launcher_tmp" "$launcher"
echo "ModelDock 目录：$app"
echo "以后启动：$launcher"
cleanup
trap - EXIT
if "$install_only"; then exit 0; fi
# A piped installer must give the terminal back to Textual.
if [[ -t 0 ]]; then
    exec bash "$app/run.sh" "${app_args[@]}"
else
    exec bash "$app/run.sh" "${app_args[@]}" </dev/tty
fi
