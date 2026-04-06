from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest


def _write_min_coco_instances(path: Path, num_images: int, image_size: int) -> None:
    images = [
        {"id": i + 1, "file_name": f"{i + 1:06d}.png", "width": image_size, "height": image_size}
        for i in range(num_images)
    ]
    data = {"images": images, "annotations": [], "categories": [{"id": 1, "name": "component"}]}
    path.write_text(json.dumps(data), encoding="utf-8")


def _write_min_rgbd_dataset(root: Path, image_size: int) -> None:
    import numpy as np
    from PIL import Image

    (root / "images" / "train").mkdir(parents=True, exist_ok=True)
    (root / "images" / "val").mkdir(parents=True, exist_ok=True)
    (root / "depth" / "depth_npy" / "train").mkdir(parents=True, exist_ok=True)
    (root / "annotations").mkdir(parents=True, exist_ok=True)

    for split in ("train", "val"):
        image_name = f"{split}_000001.png"
        Image.new("RGB", (image_size, image_size), color=(12, 34, 56)).save(
            root / "images" / split / image_name
        )

    np.save(
        root / "depth" / "depth_npy" / "train" / "train_000001.npy",
        np.full((image_size, image_size), 0.5, dtype=np.float32),
    )

    _write_min_coco_instances(root / "annotations" / "instances_train.json", 1, image_size)
    _write_min_coco_instances(root / "annotations" / "instances_val.json", 1, image_size)


@pytest.mark.parametrize(
    "script_name",
    [
        "run_ecc_20ep_trackp_magformer.sh",
        "run_ecc_20ep_trackp_mgm_mask2former.sh",
        "run_ecc_20ep_trackp_msmformer.sh",
        "run_ecc_20ep_tracks_msmformer.sh",
        "run_ecc_20ep_trackp_uoais.sh",
        "run_ecc_20ep_trackp_ucn.sh",
        "run_ecc_20ep_trackp_official_mask2former.sh",
        "run_ecc_20ep_trackp_maskrcnn.sh",
        "run_ecc_20ep_trackp_yolov8_seg.sh",
    ],
)
def test_trackp_runner_dry_run_metadata_cmd_is_reproducible(tmp_path: Path, script_name: str) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "experiments" / script_name
    dataset_root = tmp_path / "0909_512_0.12K"
    _write_min_rgbd_dataset(dataset_root, 512)

    res = subprocess.run(
        [
            "bash",
            str(script),
            "--register",
            "0909",
            "--dataset-root",
            str(dataset_root),
            "--output-root",
            str(tmp_path / "out"),
            "--candidate-id",
            "C1",
            "--dry-run",
        ],
        cwd=str(tmp_path),
        check=True,
        capture_output=True,
        text=True,
    )

    assert "--dry-run" in res.stdout
    assert "--mode " not in res.stdout


@pytest.mark.parametrize(
    ("script_name", "extra_args", "expected_fragment"),
    [
        (
            "run_0831_1k_20ep_1024_revisit_magformer.sh",
            ["--variant", "nodpth_ref"],
            "magformer_nodpth_ref",
        ),
        (
            "run_0831_1k_20ep_1024_revisit_magformer.sh",
            ["--variant", "nodpth_ref_fair"],
            "magformer_nodpth_ref_fair",
        ),
        (
            "run_0831_1k_20ep_1024_revisit_mgm_mask2former.sh",
            ["--variant", "depthnorm_on", "--num-gpus", "2"],
            "DDP.FIND_UNUSED_PARAMETERS True",
        ),
        (
            "run_0831_1k_20ep_1024_revisit_unet_boundary_inst.sh",
            [],
            "unet_boundary_inst",
        ),
    ],
)
def test_0831_1024_revisit_runner_dry_run_metadata_cmd_is_reproducible(
    tmp_path: Path,
    script_name: str,
    extra_args: list[str],
    expected_fragment: str,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "experiments" / script_name
    dataset_root = tmp_path / "0831_1K"
    _write_min_rgbd_dataset(dataset_root, 1024)

    res = subprocess.run(
        [
            "bash",
            str(script),
            "--dataset-root",
            str(dataset_root),
            "--output-root",
            str(tmp_path / "out"),
            *extra_args,
            "--dry-run",
        ],
        cwd=str(tmp_path),
        check=True,
        capture_output=True,
        text=True,
    )

    assert "--dry-run" in res.stdout
    assert expected_fragment in res.stdout


def test_magformer_revisit_runner_prepares_fallback_warmstart_in_dry_run(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "experiments" / "run_0831_1k_20ep_1024_revisit_magformer.sh"
    dataset_root = tmp_path / "20260318_1K_1566_256"
    _write_min_rgbd_dataset(dataset_root, 256)

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
            "--variant",
            "nodpth_ref",
            "--image-size",
            "256",
            "--dry-run",
        ],
        cwd=str(tmp_path),
        check=True,
        capture_output=True,
        text=True,
    )

    assert "convert_mask2former_ckpt_to_magformer.py" in res.stdout
    assert "model.finetune_weights=" in res.stdout
    assert "mask2former2_swin_tiny_coco_instance_86143f_to_magformer" in res.stdout


def test_detectron2_mask2former_runner_sets_input_image_size_for_lsj_training(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "experiments" / "run_0831_1k_20ep_scratch_official_mask2former.sh"
    dataset_root = tmp_path / "20260318_1K_1566_512"
    _write_min_rgbd_dataset(dataset_root, 512)

    res = subprocess.run(
        [
            "bash",
            str(script),
            "--register",
            "20260318_1K_1566",
            "--dataset-root",
            str(dataset_root),
            "--output-root",
            str(tmp_path / "out"),
            "--candidate-id",
            "C1",
            "--run-tag",
            "final",
            "--image-size",
            "512",
            "--pretrained",
            "--dry-run",
        ],
        cwd=str(tmp_path),
        check=True,
        capture_output=True,
        text=True,
    )

    assert "INPUT.IMAGE_SIZE 512" in res.stdout


def test_detectron2_mgm_runner_sets_input_image_size_for_lsj_training(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "experiments" / "run_0831_1k_20ep_1024_revisit_mgm_mask2former.sh"
    dataset_root = tmp_path / "20260318_1K_1566_512"
    _write_min_rgbd_dataset(dataset_root, 512)

    res = subprocess.run(
        [
            "bash",
            str(script),
            "--register",
            "20260318_1K_1566",
            "--dataset-root",
            str(dataset_root),
            "--output-root",
            str(tmp_path / "out"),
            "--variant",
            "depthnorm_on",
            "--image-size",
            "512",
            "--dry-run",
        ],
        cwd=str(tmp_path),
        check=True,
        capture_output=True,
        text=True,
    )

    assert "INPUT.IMAGE_SIZE 512" in res.stdout


def test_magformer_revisit_runner_records_train_wall_time() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = (
        repo_root / "scripts" / "experiments" / "run_0831_1k_20ep_1024_revisit_magformer.sh"
    ).read_text(encoding="utf-8")

    assert "SECONDS=0" in script
    assert 'echo "${SECONDS}" > "${OUT}/wall_time_sec.txt"' in script
    assert script.index("SECONDS=0") < script.index('echo "${SECONDS}" > "${OUT}/wall_time_sec.txt"')


def test_maskrcnn_runner_divergence_fallback_halves_lr_with_python3() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = (
        repo_root / "scripts" / "experiments" / "run_0831_1k_20ep_scratch_maskrcnn.sh"
    )
    text = script.read_text(encoding="utf-8")

    assert 'retry_lr="$(python3 -c' in text
    assert "* 0.5" in text


def test_unet_revisit_runner_uses_dynamic_iters_per_epoch(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "experiments" / "run_0831_1k_20ep_1024_revisit_unet_boundary_inst.sh"
    dataset_root = tmp_path / "0831_1K"
    (dataset_root / "annotations").mkdir(parents=True)
    _write_min_coco_instances(dataset_root / "annotations" / "instances_train.json", 9, 1024)
    _write_min_coco_instances(dataset_root / "annotations" / "instances_val.json", 2, 1024)

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
        cwd=str(tmp_path),
        check=True,
        capture_output=True,
        text=True,
    )

    assert "--iters-per-epoch 3" in res.stdout
    assert "--max-iter 60" in res.stdout


def test_lightdepth_stage_a_runner_dry_run_metadata_cmd_is_reproducible(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "experiments" / "run_0831_1k_20ep_1024_lightdepth_stage_a_magformer.sh"
    dataset_root = tmp_path / "0831_1K"
    (dataset_root / "annotations").mkdir(parents=True)
    _write_min_coco_instances(dataset_root / "annotations" / "instances_train.json", 96, 512)
    _write_min_coco_instances(dataset_root / "annotations" / "instances_val.json", 12, 512)

    res = subprocess.run(
        [
            "bash",
            str(script),
            "--dataset-root",
            str(dataset_root),
            "--output-root",
            str(tmp_path / "out"),
            "--variant",
            "mobilenetv3_directadd_edge",
            "--dry-run",
        ],
        cwd=str(tmp_path),
        check=True,
        capture_output=True,
        text=True,
    )

    assert "--dry-run" in res.stdout
    assert "model.magformer.modality_fusion.mode=direct_add" in res.stdout
    assert "model.magformer.modality_fusion.priors=[edge]" in res.stdout


def test_lightdepth_stage_b_runner_dry_run_metadata_cmd_is_reproducible(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "experiments" / "run_0831_1k_20ep_1024_lightdepth_stage_b_magformer.sh"
    dataset_root = tmp_path / "0831_1K"
    (dataset_root / "annotations").mkdir(parents=True)
    _write_min_coco_instances(dataset_root / "annotations" / "instances_train.json", 96, 512)
    _write_min_coco_instances(dataset_root / "annotations" / "instances_val.json", 12, 512)

    res = subprocess.run(
        [
            "bash",
            str(script),
            "--dataset-root",
            str(dataset_root),
            "--output-root",
            str(tmp_path / "out"),
            "--variant",
            "mobilenetv3_crossattn_edge_validhole",
            "--dry-run",
        ],
        cwd=str(tmp_path),
        check=True,
        capture_output=True,
        text=True,
    )

    assert "--dry-run" in res.stdout
    assert "model.magformer.modality_fusion.mode=cross_attn" in res.stdout
    assert "model.magformer.modality_fusion.priors=[edge,valid-hole]" in res.stdout


def test_yolov8_runner_supports_non_nano_model_sizes_in_dry_run(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "experiments" / "run_0831_1k_20ep_scratch_yolov8_seg.sh"
    dataset_root = tmp_path / "0831_1K"
    (dataset_root / "annotations").mkdir(parents=True)
    _write_min_coco_instances(dataset_root / "annotations" / "instances_train.json", 96, 1024)
    _write_min_coco_instances(dataset_root / "annotations" / "instances_val.json", 12, 1024)

    res = subprocess.run(
        [
            "bash",
            str(script),
            "--dataset-root",
            str(dataset_root),
            "--output-root",
            str(tmp_path / "out"),
            "--image-size",
            "1024",
            "--model-size",
            "x",
            "--pretrained",
            "--dry-run",
        ],
        cwd=str(tmp_path),
        check=True,
        capture_output=True,
        text=True,
    )

    assert "--dry-run" in res.stdout
    assert "yolov8x-seg.pt" in res.stdout
    assert "yolov8_seg_x_pretrained" in res.stdout


def test_yolov8_runner_uses_standard_n_model_id_in_dry_run(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "experiments" / "run_0831_1k_20ep_scratch_yolov8_seg.sh"
    dataset_root = tmp_path / "0831_1K"
    (dataset_root / "annotations").mkdir(parents=True)
    _write_min_coco_instances(dataset_root / "annotations" / "instances_train.json", 96, 1024)
    _write_min_coco_instances(dataset_root / "annotations" / "instances_val.json", 12, 1024)

    res = subprocess.run(
        [
            "bash",
            str(script),
            "--dataset-root",
            str(dataset_root),
            "--output-root",
            str(tmp_path / "out"),
            "--image-size",
            "1024",
            "--model-size",
            "n",
            "--pretrained",
            "--dry-run",
        ],
        cwd=str(tmp_path),
        check=True,
        capture_output=True,
        text=True,
    )

    assert "--dry-run" in res.stdout
    assert "yolov8n-seg.pt" in res.stdout
    assert "yolov8_seg_n_pretrained" in res.stdout


def test_yolov8_runner_disables_plotting_in_dry_run(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "experiments" / "run_0831_1k_20ep_scratch_yolov8_seg.sh"
    dataset_root = tmp_path / "0831_1K"
    (dataset_root / "annotations").mkdir(parents=True)
    _write_min_coco_instances(dataset_root / "annotations" / "instances_train.json", 96, 1024)
    _write_min_coco_instances(dataset_root / "annotations" / "instances_val.json", 12, 1024)

    res = subprocess.run(
        [
            "bash",
            str(script),
            "--dataset-root",
            str(dataset_root),
            "--output-root",
            str(tmp_path / "out"),
            "--image-size",
            "1024",
            "--model-size",
            "n",
            "--pretrained",
            "--dry-run",
        ],
        cwd=str(tmp_path),
        check=True,
        capture_output=True,
        text=True,
    )

    assert "plots=False" in res.stdout


def test_tracks_runner_supports_custom_dataset_register_in_dry_run(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "experiments" / "run_ecc_20ep_tracks_maskrcnn.sh"
    dataset_root = tmp_path / "20260318_1K_1566"
    _write_min_rgbd_dataset(dataset_root, 1024)

    res = subprocess.run(
        [
            "bash",
            str(script),
            "--register",
            "20260318_1K_1566",
            "--dataset-root",
            str(dataset_root),
            "--output-root",
            str(tmp_path / "out"),
            "--candidate-id",
            "C1",
            "--dry-run",
        ],
        cwd=str(tmp_path),
        check=True,
        capture_output=True,
        text=True,
    )

    assert "--dry-run" in res.stdout
    assert "ecc20260318_1k_1566_train" in res.stdout
    assert "Unsupported --register" not in res.stdout


def test_maskrcnn_runner_prefers_local_cached_pretrained_weights(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "experiments" / "run_0831_1k_20ep_scratch_maskrcnn.sh"
    dataset_root = tmp_path / "0831_1K"
    _write_min_rgbd_dataset(dataset_root, 1024)

    pretrained_dir = repo_root / "output" / "pretrained"
    pretrained_dir.mkdir(parents=True, exist_ok=True)
    local_weights = pretrained_dir / "model_final_f10217.pkl"
    backup = local_weights.read_bytes() if local_weights.exists() else None
    local_weights.write_bytes(b"test-local-maskrcnn-weights")
    try:
        res = subprocess.run(
            [
                "bash",
                str(script),
                "--dataset-root",
                str(dataset_root),
                "--output-root",
                str(tmp_path / "out"),
                "--candidate-id",
                "C1",
                "--image-size",
                "1024",
                "--pretrained",
                "--dry-run",
            ],
            cwd=str(tmp_path),
            check=True,
            capture_output=True,
            text=True,
        )
    finally:
        if backup is None:
            local_weights.unlink(missing_ok=True)
        else:
            local_weights.write_bytes(backup)

    assert str(local_weights) in res.stdout
    assert "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl" not in res.stdout
    assert "INPUT.MASK_FORMAT bitmask" in res.stdout


def test_trackp_ucn_runner_uses_original_recipe_in_dry_run(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "experiments" / "run_ecc_20ep_trackp_ucn.sh"
    dataset_root = tmp_path / "0831_1K"
    _write_min_rgbd_dataset(dataset_root, 512)

    res = subprocess.run(
        [
            "bash",
            str(script),
            "--register",
            "0831",
            "--dataset-root",
            str(dataset_root),
            "--output-root",
            str(tmp_path / "out"),
            "--candidate-id",
            "C1",
            "--dry-run",
        ],
        cwd=str(tmp_path),
        check=True,
        capture_output=True,
        text=True,
    )

    assert "--lr 1e-05" in res.stdout or "--lr 0.00001" in res.stdout
    assert "--num-seeds 100" in res.stdout
    assert "--kappa 20" in res.stdout


def test_trackp_ucn_runner_supports_custom_image_size(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "experiments" / "run_ecc_20ep_trackp_ucn.sh"
    dataset_root = tmp_path / "20260318_1K_1566"
    _write_min_rgbd_dataset(dataset_root, 1024)

    res = subprocess.run(
        [
            "bash",
            str(script),
            "--register",
            "20260318_1K_1566",
            "--dataset-root",
            str(dataset_root),
            "--output-root",
            str(tmp_path / "out"),
            "--candidate-id",
            "C1",
            "--image-size",
            "1024",
            "--dry-run",
        ],
        cwd=str(tmp_path),
        check=True,
        capture_output=True,
        text=True,
    )

    assert "register=20260318_1k_1566" in res.stdout
    assert "--img-size 1024" in res.stdout


def test_0831_ucn_runner_uses_canonical_model_id_in_dry_run(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "experiments" / "run_0831_1k_20ep_scratch_ucn.sh"
    dataset_root = tmp_path / "0831_1K"
    _write_min_rgbd_dataset(dataset_root, 512)

    res = subprocess.run(
        [
            "bash",
            str(script),
            "--dataset-root",
            str(dataset_root),
            "--output-root",
            str(tmp_path / "out"),
            "--candidate-id",
            "C1",
            "--image-size",
            "512",
            "--dry-run",
        ],
        cwd=str(tmp_path),
        check=True,
        capture_output=True,
        text=True,
    )

    assert "/out/ucn" in res.stdout
    assert "ucn_scratch" not in res.stdout


def test_0831_ucn_runner_supports_custom_dataset_register_in_dry_run(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "experiments" / "run_0831_1k_20ep_scratch_ucn.sh"
    dataset_root = tmp_path / "20260318_1K_1566"
    _write_min_rgbd_dataset(dataset_root, 1024)

    res = subprocess.run(
        [
            "bash",
            str(script),
            "--register",
            "20260318_1K_1566",
            "--dataset-root",
            str(dataset_root),
            "--output-root",
            str(tmp_path / "out"),
            "--candidate-id",
            "C1",
            "--image-size",
            "1024",
            "--dry-run",
        ],
        cwd=str(tmp_path),
        check=True,
        capture_output=True,
        text=True,
    )

    assert "--register '20260318_1K_1566'" in res.stdout or "--register 20260318_1K_1566" in res.stdout
    assert "Unsupported --register" not in res.stdout


def test_trackp_msmformer_runner_prefers_local_cached_pretrained_weights(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "experiments" / "run_ecc_20ep_trackp_msmformer.sh"
    dataset_root = tmp_path / "0831_1K"
    _write_min_rgbd_dataset(dataset_root, 512)

    pretrained_dir = repo_root / "output" / "pretrained"
    pretrained_dir.mkdir(parents=True, exist_ok=True)
    local_weights = pretrained_dir / "norm_RGBD_pretrained.pth"
    created = False
    if not local_weights.exists():
        local_weights.write_bytes(b"test-local-msmformer-trackp-weights")
        created = True
    try:
        res = subprocess.run(
            [
                "bash",
                str(script),
                "--register",
                "0831",
                "--dataset-root",
                str(dataset_root),
                "--output-root",
                str(tmp_path / "out"),
                "--candidate-id",
                "C1",
                "--dry-run",
            ],
            cwd=str(tmp_path),
            check=True,
            capture_output=True,
            text=True,
        )
    finally:
        if created:
            local_weights.unlink(missing_ok=True)

    assert str(local_weights) in res.stdout


def test_0831_msmformer_runner_prefers_local_cached_pretrained_weights(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "experiments" / "run_0831_1k_20ep_scratch_msmformer.sh"
    dataset_root = tmp_path / "0831_1K"
    _write_min_rgbd_dataset(dataset_root, 512)

    pretrained_dir = repo_root / "output" / "pretrained"
    pretrained_dir.mkdir(parents=True, exist_ok=True)
    local_weights = pretrained_dir / "norm_RGBD_pretrained.pth"
    created = False
    if not local_weights.exists():
        local_weights.write_bytes(b"test-local-msmformer-0831-weights")
        created = True
    try:
        res = subprocess.run(
            [
                "bash",
                str(script),
                "--dataset-root",
                str(dataset_root),
                "--output-root",
                str(tmp_path / "out"),
                "--candidate-id",
                "C1",
                "--image-size",
                "512",
                "--dry-run",
            ],
            cwd=str(tmp_path),
            check=True,
            capture_output=True,
            text=True,
        )
    finally:
        if created:
            local_weights.unlink(missing_ok=True)

    assert str(local_weights) in res.stdout


def test_0831_msmformer_runner_respects_custom_image_size_in_dry_run(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "experiments" / "run_0831_1k_20ep_scratch_msmformer.sh"
    dataset_root = tmp_path / "20260318_1K_1566"
    _write_min_rgbd_dataset(dataset_root, 1024)

    res = subprocess.run(
        [
            "bash",
            str(script),
            "--register",
            "20260318_1K_1566",
            "--dataset-root",
            str(dataset_root),
            "--output-root",
            str(tmp_path / "out"),
            "--candidate-id",
            "C1",
            "--image-size",
            "1024",
            "--dry-run",
        ],
        cwd=str(tmp_path),
        check=True,
        capture_output=True,
        text=True,
    )

    assert "run_image_size=1024" in res.stdout
    assert "INPUT.MIN_SIZE_TRAIN '(1024,)'" in res.stdout
    assert "INPUT.MAX_SIZE_TEST 1024" in res.stdout


def test_0831_msmformer_runner_has_deep_oom_fallback_chain() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "experiments" / "run_0831_1k_20ep_scratch_msmformer.sh"
    text = script.read_text(encoding="utf-8")

    assert "retry with batch=4" in text
    assert 'run_train_cmd "${fallback_iter}" "${fallback_steps}" "${fallback_warmup}" "4"' in text
    assert "retry with batch=2" in text
    assert 'run_train_cmd "${second_iter}" "${second_steps}" "${second_warmup}" "2"' in text


def test_0831_msmformer_runner_uses_dataset_aligned_budget_in_dry_run(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "experiments" / "run_0831_1k_20ep_scratch_msmformer.sh"
    dataset_root = tmp_path / "20260318_1K_1566"
    _write_min_rgbd_dataset(dataset_root, 1024)

    res = subprocess.run(
        [
            "bash",
            str(script),
            "--register",
            "20260318_1K_1566",
            "--dataset-root",
            str(dataset_root),
            "--output-root",
            str(tmp_path / "out"),
            "--candidate-id",
            "C1",
            "--image-size",
            "1024",
            "--dry-run",
        ],
        cwd=str(tmp_path),
        check=True,
        capture_output=True,
        text=True,
    )

    assert "--iters-per-epoch 1" in res.stdout
    assert "--max-iter 20" in res.stdout
