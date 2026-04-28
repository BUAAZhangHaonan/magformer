from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def _write_dataset(root: Path) -> None:
    (root / "annotations").mkdir(parents=True, exist_ok=True)
    for split in ("train", "val"):
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "depth" / "depth_npy" / split).mkdir(parents=True, exist_ok=True)
        image_name = f"{split}_000001.png"
        Image.new("RGB", (16, 16), color=(24, 48, 96)).save(root / "images" / split / image_name)
        np.save(root / "depth" / "depth_npy" / split / f"{split}_000001.npy", np.full((16, 16), 0.5, dtype=np.float32))
        payload = {
            "images": [{"id": 1, "file_name": image_name, "width": 16, "height": 16}],
            "annotations": [],
            "categories": [{"id": 1, "name": "component"}],
        }
        (root / "annotations" / f"instances_{split}.json").write_text(json.dumps(payload), encoding="utf-8")


def _write_fake_full19_repo(fake_repo: Path) -> None:
    experiments_dir = fake_repo / "scripts" / "experiments"
    analysis_dir = fake_repo / "scripts" / "analysis"
    bin_dir = fake_repo / "bin"
    experiments_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir.mkdir(parents=True, exist_ok=True)
    bin_dir.mkdir(parents=True, exist_ok=True)

    repo_root = Path(__file__).resolve().parents[1]
    shutil.copy2(
        repo_root / "scripts" / "experiments" / "run_20260321_20260318_1k_1566_full19.sh",
        experiments_dir / "run_20260321_20260318_1k_1566_full19.sh",
    )
    shutil.copy2(
        repo_root / "scripts" / "experiments" / "common_runner.sh",
        experiments_dir / "common_runner.sh",
    )

    (bin_dir / "conda").write_text(
        """#!/usr/bin/env bash
set -euo pipefail
if [[ "${1:-}" == "run" ]]; then
  shift
fi
if [[ "${1:-}" == "-n" ]]; then
  shift 2
fi
if [[ "${1:-}" == "python" ]]; then
  shift
  set -- python3 "$@"
fi
exec "$@"
""",
        encoding="utf-8",
    )
    os.chmod(bin_dir / "conda", 0o755)

    (experiments_dir / "dummy_model_cmd.py").write_text(
        """from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--consume-stdin", action="store_true")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--stamp-file")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.cocoeval.json").write_text(json.dumps({"segm_AP": 0.1}), encoding="utf-8")
    (out_dir / "metadata.json").write_text(json.dumps({"status": "done"}), encoding="utf-8")
    if args.stamp_file:
        (out_dir / args.stamp_file).write_text("ran\\n", encoding="utf-8")
    if args.consume_stdin:
        sys.stdin.read()
    fail_model = os.environ.get("FAKE_FULL19_FAIL_MODEL")
    if fail_model and fail_model == args.model_id:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
""",
        encoding="utf-8",
    )

    (experiments_dir / "full19_roster.py").write_text(
        """from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--format", required=True)
    parser.add_argument("--output-root", required=False, default="")
    parser.add_argument("--register", required=False)
    parser.add_argument("--dataset-root", required=False)
    parser.add_argument("--mode", required=False)
    parser.add_argument("--image-size", required=False)
    parser.add_argument("--single-gpu", action="store_true")
    args = parser.parse_args()
    scenario = os.environ.get("FAKE_FULL19_SCENARIO", "stdin")

    if args.format == "manifest":
        json.dump({"models": [{"id": "model_a"}, {"id": "model_b"}]}, sys.stdout)
        return 0

    output_root = Path(args.output_root)
    dummy = Path(__file__).with_name("dummy_model_cmd.py")
    if scenario == "recover":
        rows = [
            (
                "model_a",
                "model_a",
                f"python3 {dummy} --model-id model_a --out-dir {output_root / 'model_a'} --stamp-file reran.txt",
            ),
            (
                "model_b",
                "model_b",
                f"python3 {dummy} --model-id model_b --out-dir {output_root / 'model_b'}",
            ),
        ]
    else:
        rows = [
            (
                "model_a",
                "model_a",
                f"python3 {dummy} --model-id model_a --out-dir {output_root / 'model_a'} --consume-stdin",
            ),
            (
                "model_b",
                "model_b",
                f"python3 {dummy} --model-id model_b --out-dir {output_root / 'model_b'}",
            ),
        ]
    for row in rows:
        print("\\t".join(row))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
""",
        encoding="utf-8",
    )

    for rel_path, body in {
        "scripts/experiments/summarize_suite.py": """from __future__ import annotations
import argparse
import json
from pathlib import Path
parser = argparse.ArgumentParser()
parser.add_argument("--output-root", required=True)
parser.add_argument("--models-manifest", required=True)
parser.add_argument("--write", action="store_true")
args = parser.parse_args()
track = Path(args.output_root).name
(Path(args.output_root) / f"summary_{track}.json").write_text(json.dumps({"models": ["model_a", "model_b"]}), encoding="utf-8")
""",
        "scripts/analysis/benchmark_inference_suite.py": """from __future__ import annotations
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--output-root")
parser.add_argument("--dataset-root")
parser.add_argument("--summary")
parser.add_argument("--continue-on-error", action="store_true")
parser.parse_args()
""",
        "scripts/experiments/visualize_suite.py": """from __future__ import annotations
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--output-root")
parser.add_argument("--dataset-root")
parser.add_argument("--summary")
parser.add_argument("--num-images")
parser.parse_args()
""",
        "scripts/analysis/write_extended_metrics_table.py": """from __future__ import annotations
import argparse
from pathlib import Path
parser = argparse.ArgumentParser()
parser.add_argument("--summary")
parser.add_argument("--out-json", required=True)
parser.add_argument("--out-csv", required=True)
parser.add_argument("--out-md", required=True)
args = parser.parse_args()
Path(args.out_json).write_text("{}", encoding="utf-8")
Path(args.out_csv).write_text("model_id\\n", encoding="utf-8")
Path(args.out_md).write_text("# metrics\\n", encoding="utf-8")
""",
    }.items():
        path = fake_repo / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body, encoding="utf-8")


def _run_fake_full19(
    fake_repo: Path,
    tmp_path: Path,
    scenario: str,
    *,
    extra_env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    script = fake_repo / "scripts" / "experiments" / "run_20260321_20260318_1k_1566_full19.sh"
    dataset_root = tmp_path / "dataset"
    output_root = tmp_path / "out"
    _write_dataset(dataset_root)
    env = os.environ.copy()
    env["PATH"] = f"{fake_repo / 'bin'}:{env['PATH']}"
    env["PYTHONPATH"] = str(fake_repo)
    env["FAKE_FULL19_SCENARIO"] = scenario
    if extra_env:
        env.update(extra_env)
    return subprocess.run(
        [
            "bash",
            str(script),
            "--dataset-root",
            str(dataset_root),
            "--output-root",
            str(output_root),
            "--run",
        ],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )


def test_full19_roster_manifest_has_19_entries() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "experiments" / "full19_roster.py"

    res = subprocess.run(
        [sys.executable, str(script), "--format", "manifest"],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(res.stdout)
    assert len(payload["models"]) == 22
    ids = [item["id"] for item in payload["models"]]
    assert "mask2former" in ids
    assert "msmformer" in ids
    assert "uoais" in ids
    assert "unet_boundary_inst" in ids
    assert ids[-6:] == [
        "iaunet",
        "cellpose",
        "stardist",
        "unet_semantic_inst",
        "unet_boundary_inst",
        "unetpp_boundary_inst",
    ]


def test_full19_roster_uses_canonical_msmformer_source_output_name(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "experiments" / "full19_roster.py"
    dataset_root = tmp_path / "20260318_1K_1566"
    _write_dataset(dataset_root)

    res = subprocess.run(
        [
            sys.executable,
            str(script),
            "--format",
            "commands",
            "--register",
            "20260318_1K_1566",
            "--dataset-root",
            str(dataset_root),
            "--output-root",
            str(tmp_path / "out"),
            "--mode",
            "dry-run",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    msmformer_lines = [line for line in res.stdout.splitlines() if line.startswith("msmformer\t")]
    assert len(msmformer_lines) == 1
    assert "\tmsmformer\t" in msmformer_lines[0]
    assert "msmformer_scratch" not in msmformer_lines[0]


def test_full19_roster_commands_route_msmformer_to_canonical_output_name(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "experiments" / "full19_roster.py"
    dataset_root = tmp_path / "20260318_1K_1566"
    _write_dataset(dataset_root)

    res = subprocess.run(
        [
            sys.executable,
            str(script),
            "--format",
            "commands",
            "--register",
            "20260318_1K_1566",
            "--dataset-root",
            str(dataset_root),
            "--output-root",
            str(tmp_path / "out"),
            "--mode",
            "dry-run",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "\tmsmformer\t" in res.stdout
    assert "msmformer_scratch" not in res.stdout


def test_full19_suite_dry_run_lists_22_models(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "experiments" / "run_20260321_20260318_1k_1566_full19.sh"
    dataset_root = tmp_path / "20260318_1K_1566"
    _write_dataset(dataset_root)

    res = subprocess.run(
        [
            "bash",
            str(script),
            "--dataset-root",
            str(dataset_root),
            "--output-root",
            str(tmp_path / "out"),
            "--dry-run",
        ],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )

    start_ids = [
        line.split("[20260318-full19] START ", 1)[1]
        for line in res.stdout.splitlines()
        if "[20260318-full19] START " in line
    ]
    assert len(start_ids) == 22
    assert start_ids == [
        "magformer_nodpth_ref",
        "magformer_depthnorm_on",
        "magformer_lightdepth_convnextlite_spatialgate_edge_validhole",
        "magformer_lightdepth_mobilenetv3_sagate_edge_validhole",
        "magformer_lightdepth_mobilenetv3_spatialgate_edge_validhole",
        "mgm_mask2former_nodpth_ref",
        "mgm_mask2former_depthnorm_on",
        "mask2former",
        "maskrcnn",
        "yolov8_seg_n",
        "yolov8_seg_s",
        "yolov8_seg_m",
        "yolov8_seg_l",
        "yolov8_seg_x",
        "msmformer",
        "uoais",
        "iaunet",
        "cellpose",
        "stardist",
        "unet_semantic_inst",
        "unet_boundary_inst",
        "unetpp_boundary_inst",
    ]
    for model_id in [
        "magformer_depthnorm_on",
        "mgm_mask2former_depthnorm_on",
        "mask2former",
        "maskrcnn",
        "yolov8_seg_x",
        "msmformer",
        "uoais",
        "iaunet",
        "cellpose",
        "stardist",
        "unet_semantic_inst",
        "unetpp_boundary_inst",
    ]:
        assert model_id in start_ids
    assert "run_0831_1k_20ep_1024_revisit_magformer.sh" in res.stdout
    assert "run_0831_1k_20ep_1024_revisit_mgm_mask2former.sh" in res.stdout
    assert "run_0831_1k_20ep_scratch_yolov8_seg.sh" in res.stdout
    assert "--ddp --num-gpus 2" in res.stdout
    assert "mgm_mask2former_nodpth_ref" in res.stdout
    assert "mgm_mask2former_depthnorm_on" in res.stdout
    assert "msmformer_scratch" not in res.stdout
    assert "--variant nodpth_ref --num-gpus 1 --dry-run" in res.stdout
    assert "--variant depthnorm_on --num-gpus 1 --dry-run" in res.stdout
    assert "--device 0,1" in res.stdout
    assert "run_0831_1k_20ep_1024_revisit_iaunet_inst.sh" in res.stdout
    assert "run_0831_1k_20ep_1024_revisit_cellpose_inst.sh" in res.stdout
    assert "run_0831_1k_20ep_1024_revisit_stardist_inst.sh" in res.stdout


def test_full19_suite_dry_run_can_pass_multires_single_gpu_flags(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "experiments" / "run_20260321_20260318_1k_1566_full19.sh"
    dataset_root = tmp_path / "20260318_1K_1566_256"
    _write_dataset(dataset_root)

    res = subprocess.run(
        [
            "bash",
            str(script),
            "--register",
            "20260318_1K_1566_256",
            "--dataset-root",
            str(dataset_root),
            "--output-root",
            str(tmp_path / "out"),
            "--image-size",
            "256",
            "--single-gpu",
            "--dry-run",
        ],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "--image-size 256" in res.stdout
    assert "--ddp" not in res.stdout
    assert "--num-gpus 1" in res.stdout
    assert "--device 0" in res.stdout


def test_full19_suite_run_isolated_from_child_stdin(tmp_path: Path) -> None:
    fake_repo = tmp_path / "fake_repo"
    _write_fake_full19_repo(fake_repo)

    _run_fake_full19(fake_repo, tmp_path / "stdin_case", "stdin")

    output_root = tmp_path / "stdin_case" / "out"
    assert (output_root / "model_a" / "metrics.cocoeval.json").is_file()
    assert (output_root / "model_b" / "metrics.cocoeval.json").is_file()
    assert any(path.name.startswith("summary_") for path in output_root.glob("summary_*.json"))


def test_full19_suite_run_recovers_completed_staging_dir(tmp_path: Path) -> None:
    fake_repo = tmp_path / "fake_repo"
    _write_fake_full19_repo(fake_repo)

    case_root = tmp_path / "recover_case"
    output_root = case_root / "out"
    staged_dir = output_root / "_staging" / "model_a"
    staged_dir.mkdir(parents=True, exist_ok=True)
    (staged_dir / "metrics.cocoeval.json").write_text('{"segm_AP": 0.2}', encoding="utf-8")
    (staged_dir / "metadata.json").write_text('{"status": "done"}', encoding="utf-8")

    _run_fake_full19(fake_repo, case_root, "recover")

    assert (output_root / "model_a" / "metrics.cocoeval.json").is_file()
    assert not (output_root / "model_a" / "reran.txt").exists()
    assert (output_root / "model_b" / "metrics.cocoeval.json").is_file()


def test_full19_suite_run_purges_incomplete_staging_dir(tmp_path: Path) -> None:
    fake_repo = tmp_path / "fake_repo"
    _write_fake_full19_repo(fake_repo)

    case_root = tmp_path / "purge_case"
    output_root = case_root / "out"
    staged_dir = output_root / "_staging" / "model_a"
    staged_dir.mkdir(parents=True, exist_ok=True)
    (staged_dir / "stale.txt").write_text("stale\n", encoding="utf-8")

    _run_fake_full19(fake_repo, case_root, "recover")

    assert (output_root / "model_a" / "metrics.cocoeval.json").is_file()
    assert not (output_root / "model_a" / "stale.txt").exists()


def test_full19_suite_runs_after_model_failure(tmp_path: Path) -> None:
    fake_repo = tmp_path / "fake_repo"
    _write_fake_full19_repo(fake_repo)

    failure_root = tmp_path / "failure_case"
    res = _run_fake_full19(
        fake_repo,
        failure_root,
        "fail",
        extra_env={"FAKE_FULL19_FAIL_MODEL": "model_a"},
    )

    output_root = failure_root / "out"
    assert res.returncode != 0
    assert "[20260318-full19] START model_a" in res.stdout
    assert "[20260318-full19] START model_b" in res.stdout
    assert (output_root / "failed_models.tsv").is_file()
    assert "model_a" in (output_root / "failed_models.tsv").read_text(encoding="utf-8")
    assert (output_root / "model_b" / "metrics.cocoeval.json").is_file()
    assert any(path.name.startswith("summary_") for path in output_root.glob("summary_*.json"))


def test_common_runner_gpu_wait_respects_cuda_visible_devices(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    common_runner = repo_root / "scripts" / "experiments" / "common_runner.sh"
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)

    (bin_dir / "nvidia-smi").write_text(
        """#!/usr/bin/env bash
set -euo pipefail
gpu_id=""
for arg in "$@"; do
  case "$arg" in
    --id=*)
      gpu_id="${arg#--id=}"
      ;;
  esac
done
case "$gpu_id" in
  1)
    printf '50000\\n'
    ;;
  0)
    printf '1000\\n'
    ;;
  *)
    printf '1000\\n50000\\n'
    ;;
esac
""",
        encoding="utf-8",
    )
    os.chmod(bin_dir / "nvidia-smi", 0o755)

    env = os.environ.copy()
    env["PATH"] = f"{bin_dir}:{env['PATH']}"
    env["CUDA_VISIBLE_DEVICES"] = "1"

    res = subprocess.run(
        [
            "bash",
            "-lc",
            f"source '{common_runner}' && runner_gpu_free_mb",
        ],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )

    assert res.stdout.strip().splitlines()[-1] == "50000"


def test_common_runner_resource_guards_parse_meminfo_overrides(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    common_runner = repo_root / "scripts" / "experiments" / "common_runner.sh"
    meminfo = tmp_path / "meminfo"
    meminfo.write_text(
        "\n".join(
            [
                "MemTotal:       131072000 kB",
                "MemAvailable:   65536000 kB",
                "SwapTotal:       8388608 kB",
                "SwapFree:        6291456 kB",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    env = os.environ.copy()
    env["RUNNER_MEMINFO_PATH"] = str(meminfo)
    res = subprocess.run(
        [
            "bash",
            "-lc",
            f"source '{common_runner}' && echo mem=$(runner_mem_available_mb) && echo swap=$(runner_swap_used_mb)",
        ],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )

    assert "mem=64000" in res.stdout
    assert "swap=2048" in res.stdout


def test_common_runner_output_lock_creates_missing_parent(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    common_runner = repo_root / "scripts" / "experiments" / "common_runner.sh"
    run_log = tmp_path / "run.log"
    lock_dir = tmp_path / "missing_parent" / "metrics.cocoeval.json.lock"

    res = subprocess.run(
        [
            "bash",
            "-lc",
            (
                f"source '{common_runner}' && "
                f"runner_acquire_output_lock run '{run_log}' '{lock_dir}' 0 lock-test && "
                f"test -d '{lock_dir}' && "
                f"runner_release_output_lock run '{run_log}' '{lock_dir}' lock-test"
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=5,
    )

    assert "[output-lock] acquired lock-test" in res.stdout
    assert not lock_dir.exists()
