from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from tools import check_eval_protocol as checker


def _write_yaml(path: Path, *, dataset_root: str, clip_min: float, clip_max: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(
            {
                "data": {
                    "dataset_root": dataset_root,
                    "image_size": 1024,
                    "depth": {
                        "clip_min": clip_min,
                        "clip_max": clip_max,
                        "norm": "minmax",
                        "per_sample_norm": True,
                    },
                }
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def _write_ann(path: Path, image_count: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"images": [{"id": idx, "file_name": f"{idx:06d}.png"} for idx in range(image_count)]}
    path.write_text(json.dumps(payload), encoding="utf-8")


@pytest.fixture()
def protocol_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(checker, "REPO_ROOT", tmp_path)
    _write_yaml(
        tmp_path / "configs/finetune_1k_full_1024.yaml",
        dataset_root="magformer_datasets/20260318_1K_1566",
        clip_min=1.0015300512313843,
        clip_max=2.095623016357422,
    )
    _write_yaml(
        tmp_path / "configs/eval_vc_suda_stage_b_1024_teacher8499_segm.yaml",
        dataset_root="magformer_datasets/pseudo_real_512",
        clip_min=0.0,
        clip_max=2.095623016357422,
    )
    _write_ann(tmp_path / "magformer_datasets/20260318_1K_1566/annotations/instances_all.json", 1566)
    _write_ann(tmp_path / "magformer_datasets/pseudo_real_512/annotations/instances_target_unlabeled.json", 200)
    (tmp_path / checker.ORIGINAL_TEACHER_WEIGHTS).parent.mkdir(parents=True)
    (tmp_path / checker.ORIGINAL_TEACHER_WEIGHTS).write_bytes(b"teacher")
    (tmp_path / "weights/r46.pth").parent.mkdir(parents=True)
    (tmp_path / "weights/r46.pth").write_bytes(b"r46")
    (tmp_path / "weights/other.pth").write_bytes(b"other")
    return tmp_path


def _check(argv: list[str]) -> dict[str, object]:
    return checker.run_check(checker.parse_args(argv))


def test_correct_original_first50_teacher_passes(protocol_repo: Path) -> None:
    summary = _check(
        [
            "original_first50_teacher",
            "--base-config",
            "configs/finetune_1k_full_1024.yaml",
            "--dataset-root",
            "magformer_datasets/20260318_1K_1566",
            "--ann",
            "annotations/instances_all.json",
            "--split",
            "all",
            "--weights",
            str(checker.ORIGINAL_TEACHER_WEIGHTS),
            "--image-size",
            "1024",
            "--max-images",
            "50",
            "--score-threshold",
            "0.05",
            "--mask-threshold",
            "0.5",
            "--iou-types",
            "bbox,segm",
            "--inference-topk",
            "100",
            "--max-dets",
            "100",
        ]
    )

    assert summary["protocol"] == "original_first50_teacher"
    assert summary["annotation"]["image_count"] == 1566
    assert summary["checks"]["status"] == "pass"


def test_original_first50_teacher_rejects_pseudo_real_base_config(protocol_repo: Path) -> None:
    with pytest.raises(checker.ProtocolError) as excinfo:
        _check(
            [
                "original_first50_teacher",
                "--base-config",
                "configs/eval_vc_suda_stage_b_1024_teacher8499_segm.yaml",
                "--dataset-root",
                "magformer_datasets/20260318_1K_1566",
                "--ann",
                "annotations/instances_all.json",
                "--split",
                "all",
                "--weights",
                str(checker.ORIGINAL_TEACHER_WEIGHTS),
                "--image-size",
                "1024",
                "--max-images",
                "50",
                "--score-threshold",
                "0.05",
                "--mask-threshold",
                "0.5",
                "--iou-types",
                "bbox,segm",
                "--inference-topk",
                "100",
                "--max-dets",
                "100",
            ]
        )

    message = str(excinfo.value)
    assert "original_first50_teacher" in message
    assert "base_config" in message
    assert "depth" in message
    assert "protocol" in message


def test_correct_pseudo_real_target_unlabeled200_passes(protocol_repo: Path) -> None:
    summary = _check(
        [
            "pseudo_real_target_unlabeled200",
            "--base-config",
            "configs/eval_vc_suda_stage_b_1024_teacher8499_segm.yaml",
            "--dataset-root",
            "magformer_datasets/pseudo_real_512",
            "--ann",
            "annotations/instances_target_unlabeled.json",
            "--split",
            "train",
            "--weights",
            "weights/r46.pth",
            "--image-size",
            "1024",
            "--score-threshold",
            "0.05",
            "--mask-threshold",
            "0.5",
            "--iou-types",
            "bbox,segm",
            "--inference-topk",
            "200",
            "--max-dets",
            "200",
        ]
    )

    assert summary["protocol"] == "pseudo_real_target_unlabeled200"
    assert summary["annotation"]["image_count"] == 200
    assert summary["checks"]["status"] == "pass"


def test_pseudo_real_target_rejects_wrong_topk_max_dets_or_max_images(protocol_repo: Path) -> None:
    base_args = [
        "pseudo_real_target_unlabeled200",
        "--base-config",
        "configs/eval_vc_suda_stage_b_1024_teacher8499_segm.yaml",
        "--dataset-root",
        "magformer_datasets/pseudo_real_512",
        "--ann",
        "annotations/instances_target_unlabeled.json",
        "--split",
        "train",
        "--weights",
        "weights/r46.pth",
        "--image-size",
        "1024",
        "--score-threshold",
        "0.05",
        "--mask-threshold",
        "0.5",
        "--iou-types",
        "bbox,segm",
    ]

    with pytest.raises(checker.ProtocolError, match="inference_topk"):
        _check(base_args + ["--inference-topk", "100", "--max-dets", "200"])
    with pytest.raises(checker.ProtocolError, match="max_dets"):
        _check(base_args + ["--inference-topk", "200", "--max-dets", "100"])
    with pytest.raises(checker.ProtocolError, match="max_images"):
        _check(base_args + ["--inference-topk", "200", "--max-dets", "200", "--max-images", "50"])


def test_missing_weights_fails(protocol_repo: Path) -> None:
    with pytest.raises(checker.ProtocolError, match="weights.*does not exist"):
        _check(
            [
                "pseudo_real_target_unlabeled200",
                "--base-config",
                "configs/eval_vc_suda_stage_b_1024_teacher8499_segm.yaml",
                "--dataset-root",
                "magformer_datasets/pseudo_real_512",
                "--ann",
                "annotations/instances_target_unlabeled.json",
                "--split",
                "train",
                "--weights",
                "weights/missing.pth",
                "--image-size",
                "1024",
                "--score-threshold",
                "0.05",
                "--mask-threshold",
                "0.5",
                "--iou-types",
                "bbox,segm",
                "--inference-topk",
                "200",
                "--max-dets",
                "200",
            ]
        )


def test_original_first50_teacher_can_allow_nondefault_existing_weights(protocol_repo: Path) -> None:
    with pytest.raises(checker.ProtocolError, match="default Teacher weights"):
        _check(
            [
                "original_first50_teacher",
                "--base-config",
                "configs/finetune_1k_full_1024.yaml",
                "--dataset-root",
                "magformer_datasets/20260318_1K_1566",
                "--ann",
                "annotations/instances_all.json",
                "--split",
                "all",
                "--weights",
                "weights/other.pth",
                "--image-size",
                "1024",
                "--max-images",
                "50",
                "--score-threshold",
                "0.05",
                "--mask-threshold",
                "0.5",
                "--iou-types",
                "bbox,segm",
                "--inference-topk",
                "100",
                "--max-dets",
                "100",
            ]
        )

    summary = _check(
        [
            "original_first50_teacher",
            "--base-config",
            "configs/finetune_1k_full_1024.yaml",
            "--dataset-root",
            "magformer_datasets/20260318_1K_1566",
            "--ann",
            "annotations/instances_all.json",
            "--split",
            "all",
            "--weights",
            "weights/other.pth",
            "--allow-nondefault-weights",
            "--image-size",
            "1024",
            "--max-images",
            "50",
            "--score-threshold",
            "0.05",
            "--mask-threshold",
            "0.5",
            "--iou-types",
            "bbox,segm",
            "--inference-topk",
            "100",
            "--max-dets",
            "100",
        ]
    )

    assert summary["weights"]["allow_nondefault"] is True
