import os
import subprocess

from pathlib import Path


SCRIPT = Path("tools/start_vc_suda_watchers.sh")


def _script_text() -> str:
    return SCRIPT.read_text(encoding="utf-8")


def test_start_script_tracks_all_watcher_sessions() -> None:
    text = _script_text()

    for session_name in [
        "r126_cuda_resume_r121_watcher_g4567",
        "r127_gated_r122_launcher",
        "r128_gated_r122_evaluator",
        "r129_gated_go_no_go",
        "r130_final_goal_audit_watcher",
    ]:
        assert session_name in text


def test_start_script_uses_absolute_python_and_no_conda_activation() -> None:
    text = _script_text()

    assert "/home/hdd3/zhanghaonan/anaconda3/envs/magformer/bin/python" in text
    assert "conda activate" not in text
    assert " python " not in text


def test_start_script_delegates_tracked_r128_to_r130_runners() -> None:
    text = _script_text()

    assert "tools/run_r128_gated_r122_evaluator.sh" in text
    assert "tools/run_r129_gated_go_no_go.sh" in text
    assert "tools/run_r130_final_goal_audit_watcher.sh" in text


def test_start_script_contains_r126_r127_gates() -> None:
    text = _script_text()

    assert "configs/baseline_vc_suda_r121_depth_boundary_w001_smoke_b1_iter1_loss_only_fg00075.yaml" in text
    assert "configs/baseline_vc_suda_r122_depth_boundary_w001_pseudo300.yaml" in text
    assert "tools/check_r121_r122_resume_state.py" in text
    assert "NEED_R122_TRAIN" in text
    assert "CUDA_VISIBLE_DEVICES=4,5,6,7" in text


def test_start_script_uses_non_overwriting_r122_retry_log() -> None:
    text = _script_text()

    fixed_r122_log = "output/vc_suda/r122_depth_boundary_w001_r114warm_pseudo300.tmux.log"
    assert f'local r122_log="{fixed_r122_log}"' not in text
    assert "r122_depth_boundary_w001_r114warm_pseudo300.retry_$(date +%Y%m%d_%H%M%S).tmux.log" in text
    assert 'if [[ -e "${r122_log}" ]]; then' in text
    assert "refusing to overwrite existing R122 log" in text


def test_start_script_avoids_destructive_and_goal_update_commands() -> None:
    text = _script_text()

    forbidden_fragments = [
        "update_goal",
        "git reset",
        "git checkout",
        "rm -rf",
        "tmux kill-session",
    ]
    for fragment in forbidden_fragments:
        assert fragment not in text


def test_training_processes_reports_real_jobs_not_helper_or_watcher_shell(tmp_path: Path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_ps = fake_bin / "ps"
    fake_ps.write_text(
        "#!/bin/sh\n"
        "printf '%s\\n' ' 100 /home/hdd3/zhanghaonan/anaconda3/envs/magformer/bin/python -c helper mentions train.py torchrun torch.distributed.run'\n"
        "printf '%s\\n' ' 101 /bin/bash -c ps -u 1000 -o pid=,args= | /home/hdd3/zhanghaonan/anaconda3/envs/magformer/bin/python -c helper mentions train.py torchrun'\n"
        "printf '%s\\n' ' 102 bash /home/hdd3/zhanghaonan/magformer/tools/start_vc_suda_watchers.sh --r126-watcher train.py torchrun'\n"
        "printf '%s\\n' ' 200 /home/hdd3/zhanghaonan/anaconda3/envs/magformer/bin/python tools/train.py --config configs/vc_suda/example.yaml'\n"
        "printf '%s\\n' ' 201 /home/hdd3/zhanghaonan/anaconda3/envs/magformer/bin/python tools/evaluate_1024_backmap.py --config configs/vc_suda/example.yaml'\n"
        "printf '%s\\n' ' 202 torchrun --nproc_per_node=1 tools/train.py --config configs/vc_suda/example.yaml'\n"
        "printf '%s\\n' ' 203 /home/hdd3/zhanghaonan/anaconda3/envs/magformer/bin/python -m torch.distributed.run --nproc_per_node=1 tools/train.py'\n",
        encoding="utf-8",
    )
    fake_ps.chmod(0o755)
    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}:{env['PATH']}"

    result = subprocess.run(
        [
            "bash",
            "-c",
            "source <(sed '/^case \"\\${1:-}\" in/,$d' tools/start_vc_suda_watchers.sh); training_processes",
        ],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )

    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    assert [line.strip().split(maxsplit=1)[0] for line in lines] == ["200", "201", "202", "203"]
    assert all("python -c" not in line for line in lines)
    assert all("start_vc_suda_watchers.sh" not in line for line in lines)
