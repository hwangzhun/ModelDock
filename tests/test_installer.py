import os
import subprocess
from pathlib import Path

import pytest

PROJECT = Path(__file__).resolve().parents[1]


def git(repo, *args):
    return subprocess.run(['git', '-C', str(repo), *args], check=True, capture_output=True, text=True)


@pytest.fixture
def remote(tmp_path):
    repo = tmp_path / 'remote'
    repo.mkdir()
    git(repo, 'init', '-b', 'main')
    git(repo, 'config', 'user.name', 'Test')
    git(repo, 'config', 'user.email', 'test@example.invalid')
    (repo / 'run.sh').write_text('#!/usr/bin/env bash\nprintf "version-one:%s\\n" "$*"\n')
    (repo / 'pyproject.toml').write_text('[project]\nname="modeldock"\n')
    git(repo, 'add', '.')
    git(repo, 'commit', '-m', 'Initial')
    return repo


def install(tmp_path, remote, *args):
    env = {**os.environ, 'MODELDOCK_BIN_DIR': str(tmp_path / 'bin'), 'MODELDOCK_INSTALL_DIR': str(tmp_path / 'installed'),
           'MODELDOCK_REPO': remote.as_uri()}
    return subprocess.run(['bash', str(PROJECT / 'install.sh'), '--install-only', *args],
                          env=env, text=True, capture_output=True)


def test_install_cached_launch_and_update(tmp_path, remote):
    result = install(tmp_path, remote)
    assert result.returncode == 0, result.stderr
    assert 'version-one:--install-only' in result.stdout
    launcher = tmp_path / 'bin/modeldock'
    launch = subprocess.run([str(launcher), '--root', '/data/models with spaces'], capture_output=True, text=True)
    assert launch.returncode == 0 and '/data/models with spaces' in launch.stdout
    (remote / 'run.sh').write_text('#!/usr/bin/env bash\necho version-two\n')
    git(remote, 'add', '.')
    git(remote, 'commit', '-m', 'Update')
    assert 'version-one' in install(tmp_path, remote).stdout
    assert 'version-two' in install(tmp_path, remote, '--update').stdout
    (tmp_path / 'installed/app/run.sh').write_text('local changes')
    result = install(tmp_path, remote, '--update')
    assert result.returncode != 0 and '本地修改' in result.stderr
    assert (tmp_path / 'installed/app/run.sh').read_text() == 'local changes'


def test_install_failure_leaves_no_partial_app(tmp_path):
    result = install(tmp_path, tmp_path / 'missing')
    assert result.returncode != 0
    assert not (tmp_path / 'installed/app').exists()
    assert not (tmp_path / 'installed/.installer-lock').exists()
    assert not list((tmp_path / 'installed').glob('.download-*'))


def test_existing_directory_is_not_overwritten(tmp_path, remote):
    app = tmp_path / 'installed/app'
    app.mkdir(parents=True)
    (app / 'user-file').write_text('keep')
    result = install(tmp_path, remote)
    assert result.returncode != 0 and '不会覆盖' in result.stderr
    assert (app / 'user-file').read_text() == 'keep'
