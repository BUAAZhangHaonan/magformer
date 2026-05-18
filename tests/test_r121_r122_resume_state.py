from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "tools" / "check_r121_r122_resume_state.py"
R121_DIR = Path("output/vc_suda/r121_depth_boundary_w001_smoke_b1_iter1_loss_only_fg00075")
R122_DIR = Path("output/vc_suda/r122_depth_boundary_w001_r114warm_pseudo300")
R122_CKPT = R122_DIR / "checkpoint_iter_0000099.pth"
REMAINING75_METRICS = (
    Path("output/diagnostics")
    / "r122_depth_boundary_w001_iter0099_remaining75_1024_backmap_topk200_20260518"
    / "metrics.cocoeval.json"
)
VAL28_METRICS = (
    Path("output/diagnostics")
    / "r122_depth_boundary_w001_iter0099_val28_1024_backmap_topk200_20260518"
    / "metrics.cocoeval.json"
)
BUCKET_COMPARE = (
    Path("output/diagnostics")
    / "r122_depth_boundary_w001_iter0099_bucket_compare_20260518"
    / "bucket_compare.csv"
)
GO_NO_GO = (
    Path("output/diagnostics")
    / "r122_depth_boundary_w001_go_no_go_20260518"
    / "go_no_go.json"
)
WATCHER_LOG = Path("output/diagnostics/r125_cuda_resume_r121_watcher_20260518.log")
R126_WATCHER_LOG = Path("output/diagnostics/r126_cuda_resume_r121_watcher_g4567_20260518.log")


def _write(repo_root: Path, relative: Path, text: str = "ok") -> Path:
    path = repo_root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _run(repo_root: Path, *extra: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--repo-root", str(repo_root), *extra],
        capture_output=True,
        text=True,
        check=False,
    )


def _run_with_env(repo_root: Path, env: dict[str, str], *extra: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--repo-root", str(repo_root), *extra],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )


def _load_stdout(result: subprocess.CompletedProcess[str]) -> dict[str, object]:
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def _r121_passed(repo_root: Path) -> None:
    _write(
        repo_root,
        R121_DIR / "train.log",
        "depth sanity passed\niter: 0/1 loss_depth_boundary: 0.12\nmax_iter: 1\ntotal training time\n",
    )


def _r122_checkpoint(repo_root: Path) -> None:
    _write(repo_root, R122_CKPT, "checkpoint placeholder")


def _r122_metrics(repo_root: Path) -> None:
    _write(repo_root, REMAINING75_METRICS, '{"bbox/AP": 1.0}\n')
    _write(repo_root, VAL28_METRICS, '{"bbox/AP": 2.0}\n')


def _bucket_compare(repo_root: Path) -> None:
    _write(repo_root, BUCKET_COMPARE, "run,bucket_type,bucket\nr122_remaining75,area_bucket,tiny_area_le_256\n")


def test_watcher_cuda_unknown_without_r121_output_blocks_resume(tmp_path: Path) -> None:
    _write(tmp_path, WATCHER_LOG, "RuntimeError: CUDA unknown error\nSetting the available devices to be zero.\n")

    payload = _load_stdout(_run(tmp_path))

    assert payload["state"] == "BLOCKED_CUDA"
    assert any("CUDA unknown error" in reason for reason in payload["reasons"])
    assert "driver" in payload["next_action"].lower() or "cuda" in payload["next_action"].lower()


def test_r126_probe_success_takes_precedence_over_stale_r125_cuda_error(tmp_path: Path) -> None:
    _write(tmp_path, WATCHER_LOG, "RuntimeError: CUDA unknown error\nSetting the available devices to be zero.\n")
    _write(tmp_path, R126_WATCHER_LOG, "GPU 5 probe success\nlaunching R121 smoke on GPU 5\n")

    payload = _load_stdout(_run(tmp_path))

    assert payload["state"] == "NEED_R121_SMOKE"
    reason_text = "\n".join(payload["reasons"])
    assert str(R126_WATCHER_LOG) in reason_text
    assert str(WATCHER_LOG) not in reason_text
    assert payload["paths"]["watcher_log"].endswith(str(R126_WATCHER_LOG))


def test_r126_probe_failure_blocks_even_when_r125_error_is_stale(tmp_path: Path) -> None:
    _write(tmp_path, WATCHER_LOG, "RuntimeError: CUDA unknown error\nSetting the available devices to be zero.\n")
    _write(tmp_path, R126_WATCHER_LOG, "GPU4-7 probe failed\nno probed GPU available\n")

    payload = _load_stdout(_run(tmp_path))

    assert payload["state"] == "BLOCKED_CUDA"
    reason_text = "\n".join(payload["reasons"])
    assert str(R126_WATCHER_LOG) in reason_text
    assert str(WATCHER_LOG) not in reason_text
    assert payload["paths"]["watcher_log"].endswith(str(R126_WATCHER_LOG))


def test_r121_success_takes_precedence_over_stale_r125_cuda_error(tmp_path: Path) -> None:
    _write(tmp_path, WATCHER_LOG, "RuntimeError: CUDA unknown error\nSetting the available devices to be zero.\n")
    _r121_passed(tmp_path)

    payload = _load_stdout(_run(tmp_path))

    assert payload["state"] == "NEED_R122_TRAIN"
    reason_text = "\n".join(payload["reasons"])
    assert "R121 smoke passed" in reason_text
    assert str(WATCHER_LOG) not in reason_text


def test_r121_retry_log_failure_reports_clear_reasons(tmp_path: Path) -> None:
    _write(
        tmp_path,
        Path(str(R121_DIR) + ".retry_20260518.tmux.log"),
        "CUDA out of memory\nNaN loss detected\ndepth sanity failed\nloss_depth_boundary missing\n",
    )

    payload = _load_stdout(_run(tmp_path))

    assert payload["state"] == "R121_FAILED"
    reason_text = "\n".join(payload["reasons"])
    assert "OOM" in reason_text
    assert "NaN" in reason_text
    assert "depth sanity failed" in reason_text
    assert "loss_depth_boundary missing" in reason_text


def test_r121_one_iter_with_depth_boundary_loss_needs_r122_train(tmp_path: Path) -> None:
    _r121_passed(tmp_path)

    payload = _load_stdout(_run(tmp_path))

    assert payload["state"] == "NEED_R122_TRAIN"
    assert any("R121" in reason and "passed" in reason for reason in payload["reasons"])


def test_missing_r122_checkpoint_needs_r122_train(tmp_path: Path) -> None:
    _r121_passed(tmp_path)
    (tmp_path / R122_DIR).mkdir(parents=True)
    _write(tmp_path, R122_DIR / "train.log", "iter: 99 loss_depth_boundary: 0.10\n")

    payload = _load_stdout(_run(tmp_path))

    assert payload["state"] == "NEED_R122_TRAIN"
    assert any("checkpoint_iter_0000099.pth" in reason for reason in payload["reasons"])


def test_r122_checkpoint_without_eval_metrics_needs_r122_eval(tmp_path: Path) -> None:
    _r121_passed(tmp_path)
    _r122_checkpoint(tmp_path)

    payload = _load_stdout(_run(tmp_path))

    assert payload["state"] == "NEED_R122_EVAL"
    reason_text = "\n".join(payload["reasons"])
    assert "remaining75" in reason_text
    assert "val28" in reason_text


def test_r122_checkpoint_without_eval_waits_when_matching_train_process_exists(tmp_path: Path) -> None:
    _r121_passed(tmp_path)
    _r122_checkpoint(tmp_path)
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_ps = fake_bin / "ps"
    fake_ps.write_text(
        "#!/bin/sh\n"
        "printf '%s\\n' \" 4242 torchrun --nproc_per_node=4 train.py --config configs/vc_suda/r122_depth_boundary_w001_r114warm_pseudo300.yaml --output output/vc_suda/r122_depth_boundary_w001_r114warm_pseudo300\"\n"
        "printf '%s\\n' \" 4243 bash tools/watch_r122_resume_state.sh output/vc_suda/r122_depth_boundary_w001_r114warm_pseudo300\"\n"
        "printf '%s\\n' \" 4244 python -m torch.distributed.run --nproc_per_node=4 train.py --config configs/vc_suda/r122_depth_boundary_w001_r114warm_pseudo300.yaml\"\n"
        "printf '%s\\n' \" 4245 python train.py --session r122_depth_boundary_w001_r114warm_pseudo300\"\n",
        encoding="utf-8",
    )
    fake_ps.chmod(0o755)
    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}:{env['PATH']}"
    output_json = tmp_path / "state.json"
    output_md = tmp_path / "state.md"

    payload = _load_stdout(_run_with_env(tmp_path, env, "--output-json", str(output_json), "--output-md", str(output_md)))

    assert payload["state"] == "R122_TRAINING"
    reason_text = "\n".join(payload["reasons"])
    assert "4242" in reason_text
    assert "4243" not in reason_text
    assert "4244" in reason_text
    assert "4245" in reason_text
    assert "wait" in payload["next_action"].lower()
    assert "training" in payload["next_action"].lower()
    assert [process["pid"] for process in payload["train_processes"]] == [4242, 4244, 4245]
    assert json.loads(output_json.read_text(encoding="utf-8"))["train_processes"][0]["pid"] == 4242
    md = output_md.read_text(encoding="utf-8")
    assert "Train Processes" in md
    assert "pid=4242" in md
    assert "pid=4243" not in md


def test_metrics_and_bucket_compare_without_go_no_go_needs_go_no_go(tmp_path: Path) -> None:
    _r121_passed(tmp_path)
    _r122_checkpoint(tmp_path)
    _r122_metrics(tmp_path)
    _bucket_compare(tmp_path)

    payload = _load_stdout(_run(tmp_path))

    assert payload["state"] == "NEED_GO_NO_GO"
    assert any("go_no_go.json" in reason for reason in payload["reasons"])


def test_go_no_go_json_ready_to_decide_reads_pass_fail(tmp_path: Path) -> None:
    _r121_passed(tmp_path)
    _r122_checkpoint(tmp_path)
    _r122_metrics(tmp_path)
    _bucket_compare(tmp_path)
    _write(tmp_path, GO_NO_GO, '{"decision": "PASS", "passed": true}\n')

    payload = _load_stdout(_run(tmp_path))

    assert payload["state"] == "READY_TO_DECIDE"
    assert payload["go_no_go"]["pass"] is True
    assert payload["go_no_go"]["decision"] == "PASS"


def test_go_no_go_json_ready_to_decide_reads_failure(tmp_path: Path) -> None:
    _r121_passed(tmp_path)
    _r122_checkpoint(tmp_path)
    _r122_metrics(tmp_path)
    _bucket_compare(tmp_path)
    _write(tmp_path, GO_NO_GO, '{"decision": "FAIL"}\n')

    payload = _load_stdout(_run(tmp_path))

    assert payload["state"] == "READY_TO_DECIDE"
    assert payload["go_no_go"]["pass"] is False
    assert payload["go_no_go"]["decision"] == "FAIL"


def test_missing_required_go_no_go_field_errors_without_fallback(tmp_path: Path) -> None:
    _r121_passed(tmp_path)
    _r122_checkpoint(tmp_path)
    _r122_metrics(tmp_path)
    _bucket_compare(tmp_path)
    _write(tmp_path, GO_NO_GO, '{"passed": true}\n')

    result = _run(tmp_path)

    assert result.returncode == 2
    assert "missing required field decision" in result.stderr


def test_duplicate_candidate_retry_logs_error_without_silent_fallback(tmp_path: Path) -> None:
    _write(tmp_path, Path(str(R121_DIR) + ".retry_a.tmux.log"), "CUDA out of memory\n")
    _write(tmp_path, Path(str(R121_DIR) + ".retry_b.tmux.log"), "CUDA out of memory\n")

    result = _run(tmp_path)

    assert result.returncode == 2
    assert "multiple R121 retry logs" in result.stderr


def test_markdown_output_lists_state_reasons_and_next_action(tmp_path: Path) -> None:
    output_md = tmp_path / "state.md"
    _write(tmp_path, WATCHER_LOG, "RuntimeError: CUDA unknown error\n")

    result = _run(tmp_path, "--output-md", str(output_md))
    assert result.returncode == 0, result.stderr
    md = output_md.read_text(encoding="utf-8")
    assert "State" in md
    assert "BLOCKED_CUDA" in md
    assert "Reasons" in md
    assert "Next Action" in md
