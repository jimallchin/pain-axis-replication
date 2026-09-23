"""Check orchestration without downloads, inference or expensive analysis."""

import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
import zipfile

import pytest


@pytest.fixture
def reproduction(tmp_path):
    root = Path(__file__).resolve().parents[1]
    shutil.copy(root / "reproduce.sh", tmp_path)
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    fetch = scripts / "fetch_upstream.sh"
    fetch.write_text('#!/bin/bash\necho \'["fetch-upstream"]\' >> "$CALL_LOG"\n')
    fetch.chmod(0o755)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    uv = bin_dir / "uv"
    uv_script = bin_dir / "uv.py"
    uv.write_text(f'#!/bin/bash\nexec {shlex.quote(sys.executable)} {shlex.quote(str(uv_script))} "$@"\n')
    uv_script.write_text('''import json, os, sys
args = sys.argv[1:]
with open(os.environ["CALL_LOG"], "a") as f:
    f.write(json.dumps(args) + "\\n")
if "-c" in args and "torch.cuda" in args[-1]:
    sys.exit(int(os.environ.get("NO_CUDA", "0")))
''')
    uv.chmod(0o755)
    log = tmp_path / "calls.jsonl"
    env = {**os.environ, "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}", "CALL_LOG": str(log)}

    def run(command, **overrides):
        log.write_text("")
        result = subprocess.run(["bash", "reproduce.sh", command], cwd=tmp_path,
                                env={**env, **overrides}, capture_output=True, text=True, check=False, timeout=60)
        calls = [json.loads(line) for line in log.read_text().splitlines()]
        return result, calls

    return tmp_path, run


def test_setup_never_downloads_weights_or_checks_cuda(reproduction):
    root, run = reproduction
    result, calls = run("setup", NO_CUDA="1")
    assert result.returncode == 0, result.stderr
    assert calls == [["sync", "--group", "dev"], ["fetch-upstream"], ["run", "pytest", "-q"]]
    assert not (root / "adapters").exists()


@pytest.mark.parametrize("command", ["cpu", "analyze", "compare", "A", "C"])
def test_analysis_commands_do_not_select_gpu_dependencies(reproduction, command):
    _, run = reproduction
    result, calls = run(command, NO_CUDA="1")
    assert result.returncode == 0, result.stderr
    assert calls
    assert all(call[:2] == ["run", "python"] for call in calls)
    assert all("hf" not in call and "--extra" not in call for call in calls)


def test_gpu_setup_checks_cuda_before_downloading_weights(reproduction):
    root, run = reproduction
    result, calls = run("setup-gpu", NO_CUDA="1")
    assert result.returncode != 0
    assert calls[0] == ["sync", "--group", "dev", "--extra", "gpu"]
    assert len(calls) == 2 and "torch.cuda" in calls[1][-1]
    assert not (root / "adapters").exists()


def test_gpu_setup_reuses_adapter_and_downloads_base_after_cuda_check(reproduction):
    root, run = reproduction
    adapter = root / "adapters/Qwen_2.5_32B_instruct/released"
    adapter.mkdir(parents=True)
    (adapter / "adapter_config.json").write_text("{}")
    result, calls = run("setup-gpu")
    assert result.returncode == 0, result.stderr
    assert len(calls) == 3
    assert "torch.cuda" in calls[1][-1]
    assert calls[2] == ["run", "--extra", "gpu", "hf", "download", "Qwen/Qwen2.5-32B-Instruct",
                        "--revision", "5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd"]


def test_gpu_stages_keep_their_model_dependencies(reproduction):
    _, run = reproduction
    result, calls = run("gpu")
    assert result.returncode == 0, result.stderr
    assert len(calls) > 1
    assert all(call[:4] == ["run", "--extra", "gpu", "python"] for call in calls)


@pytest.mark.parametrize("command", ["gpu", "B", "D", "F", "G", "H", "I", "J", "K"])
def test_model_stages_stop_before_inference_without_cuda(reproduction, command):
    _, run = reproduction
    result, calls = run(command, NO_CUDA="1")
    assert result.returncode != 0
    assert len(calls) == 1 and "torch.cuda" in calls[0][-1]


def test_logs_unpack_without_python_or_model_dependencies(reproduction):
    root, run = reproduction
    (root / "logs").mkdir()
    with zipfile.ZipFile(root / "logs/pain-axis-replication-logs.zip", "w") as archive:
        archive.writestr("README.txt", "archive instructions")
        archive.writestr("runs/replication/trial.jsonl", '{"trial": 1}\n')
    result, calls = run("logs", NO_CUDA="1")
    assert result.returncode == 0, result.stderr
    assert calls == []
    assert (root / "runs/replication/trial.jsonl").read_text() == '{"trial": 1}\n'
    assert not (root / "README.txt").exists()
