from __future__ import annotations

from pathlib import Path

import pytest

from tools import evaluate_teacher_first50_1024_backmap as wrapper


def test_original_first50_defaults_to_finetune_config_and_fixed_data() -> None:
    args = wrapper.parse_args(["--weights", "teacher.pth", "--output-dir", "out"])

    assert args.base_config == "configs/finetune_1k_full_1024.yaml"
    assert args.dataset_root == "magformer_datasets/20260318_1K_1566"
    assert args.ann == "annotations/instances_all.json"
    assert args.split == "all"
    assert args.max_images == 50
    assert args.image_size == 1024
    assert args.iou_types == "bbox,segm"


def test_original_first50_protocol_accepts_only_original_depth_and_dataset() -> None:
    evidence = wrapper.validate_original_first50_protocol(
        base_config=Path("configs/finetune_1k_full_1024.yaml"),
        dataset_root=Path("magformer_datasets/20260318_1K_1566"),
        ann="annotations/instances_all.json",
        split="all",
        image_size=1024,
        max_images=50,
    )

    assert evidence["base_config"] == "configs/finetune_1k_full_1024.yaml"
    assert evidence["depth_clip_min"] == pytest.approx(1.0015300512313843)
    assert evidence["depth_clip_max"] == pytest.approx(2.095623016357422)


def test_original_first50_protocol_rejects_stage_b_pseudo_real_config() -> None:
    with pytest.raises(wrapper.ProtocolError) as excinfo:
        wrapper.validate_original_first50_protocol(
            base_config=Path("configs/vc_suda_stage_b_1024_teacher8499.yaml"),
            dataset_root=Path("magformer_datasets/20260318_1K_1566"),
            ann="annotations/instances_all.json",
            split="all",
            image_size=1024,
            max_images=50,
        )

    message = str(excinfo.value)
    assert "configs/finetune_1k_full_1024.yaml" in message
    assert "configs/vc_suda_stage_b_1024_teacher8499.yaml" in message
    assert "depth clip" in message
    assert "pseudo_real" in message
