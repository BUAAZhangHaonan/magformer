from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def _render_commands(tmp_path: Path, image_size: int, *, single_gpu: bool = False) -> dict[str, tuple[str, str]]:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "experiments" / "full19_roster.py"
    dataset_root = tmp_path / f"20260318_1K_1566_{image_size}"
    output_root = tmp_path / f"out_{image_size}"

    cmd = [
        sys.executable,
        str(script),
        "--format",
        "commands",
        "--register",
        f"20260318_1K_1566_{image_size}",
        "--dataset-root",
        str(dataset_root),
        "--output-root",
        str(output_root),
        "--mode",
        "dry-run",
        "--image-size",
        str(image_size),
    ]
    if single_gpu:
        cmd.append("--single-gpu")

    res = subprocess.run(
        [
            *cmd,
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    rows: dict[str, tuple[str, str]] = {}
    for line in res.stdout.splitlines():
        model_id, source_name, command = line.split("\t", 2)
        rows[model_id] = (source_name, command)
    return rows


def test_full19_roster_can_render_multires_commands_for_512(tmp_path: Path) -> None:
    rows = _render_commands(tmp_path, image_size=512)

    assert len(rows) == 19
    assert rows["msmformer"][0] == "msmformer"
    assert "--dataset-root" in rows["msmformer"][1]
    assert "--image-size 512" in rows["msmformer"][1]
    assert "--image-size 512" in rows["mask2former"][1]
    assert "--image-size 512" in rows["maskrcnn"][1]
    assert "--image-size 512" in rows["yolov8_seg_n"][1]
    assert "--image-size 512" in rows["unet_semantic_inst"][1]
    assert "--image-size 512" in rows["magformer_nodpth_ref"][1]
    assert "--image-size 512" in rows["yolov8_seg_x"][1]


def test_full19_roster_can_render_multires_commands_for_256(tmp_path: Path) -> None:
    rows = _render_commands(tmp_path, image_size=256)

    assert rows["msmformer"][0] == "msmformer"
    assert "--image-size 256" in rows["msmformer"][1]
    assert "--image-size 256" in rows["uoais"][1]
    assert "--image-size 256" in rows["unet_boundary_inst"][1]
    assert "--image-size 256" in rows["unetpp_boundary_inst"][1]
    assert "--image-size 256" in rows["yolov8_seg_x"][1]


def test_full19_roster_can_render_single_gpu_multires_commands(tmp_path: Path) -> None:
    rows = _render_commands(tmp_path, image_size=256, single_gpu=True)

    assert "--ddp" not in rows["magformer_nodpth_ref"][1]
    assert "--num-gpus 1" in rows["magformer_nodpth_ref"][1]
    assert "--ddp" not in rows["magformer_lightdepth_mobilenetv3_sagate_edge_validhole"][1]
    assert "--num-gpus 1" in rows["magformer_lightdepth_mobilenetv3_sagate_edge_validhole"][1]
    assert "--device 0" in rows["yolov8_seg_n"][1]
    assert "--device 0" in rows["yolov8_seg_x"][1]
