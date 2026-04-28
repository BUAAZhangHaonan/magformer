from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from PIL import Image


def _load_module():
    repo_root = Path(__file__).resolve().parents[1]
    path = repo_root / "baselines" / "run_cellpose_instance_ecc.py"
    spec = importlib.util.spec_from_file_location("run_cellpose_instance_ecc", path)
    assert spec is not None
    assert spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _write_min_ecc_rgb_dataset(root: Path) -> None:
    (root / "annotations").mkdir(parents=True, exist_ok=True)
    (root / "images" / "train").mkdir(parents=True, exist_ok=True)
    (root / "images" / "val").mkdir(parents=True, exist_ok=True)

    train_name = "train_000001.png"
    val_name = "val_000001.png"
    Image.new("RGB", (16, 16), color=(12, 34, 56)).save(root / "images" / "train" / train_name)
    Image.new("RGB", (16, 16), color=(56, 34, 12)).save(root / "images" / "val" / val_name)
    payload = {
        "images": [
            {"id": 1, "file_name": train_name, "width": 16, "height": 16},
        ],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "segmentation": [[2, 2, 7, 2, 7, 7, 2, 7]],
                "area": 25,
                "bbox": [2, 2, 5, 5],
                "iscrowd": 0,
            }
        ],
        "categories": [{"id": 1, "name": "component"}],
    }
    (root / "annotations" / "instances_train.json").write_text(json.dumps(payload), encoding="utf-8")
    (root / "annotations" / "instances_val.json").write_text(json.dumps(payload), encoding="utf-8")


def test_run_experiment_writes_standard_artifacts_with_injected_predictions(tmp_path: Path) -> None:
    mod = _load_module()
    dataset_root = tmp_path / "ecc"
    _write_min_ecc_rgb_dataset(dataset_root)
    output_dir = tmp_path / "out"

    def fake_train_model(*_args, **_kwargs):
        return {"model": object(), "checkpoint": output_dir / "model_final.pth", "trainable_params": 1234}

    def fake_predict_records(*_args, **_kwargs):
        return [
            {
                "image_id": 1,
                "category_id": 0,
                "score": 0.9,
                "mask": __import__("numpy").ones((16, 16), dtype=__import__("numpy").uint8),
                "bbox": [0.0, 0.0, 16.0, 16.0],
            }
        ]

    def fake_eval_results(*_args, **_kwargs):
        return {"segm/AP": 1.0, "bbox/AP": 1.0}

    result = mod.run_experiment(
        dataset_root=str(dataset_root),
        output_dir=str(output_dir),
        image_size=16,
        epochs=1,
        batch=1,
        lr=1e-3,
        num_workers=0,
        min_area=5,
        device="cpu",
        max_train_steps=1,
        max_val_images=1,
        train_split="train",
        val_split="val",
        train_model_fn=fake_train_model,
        predict_records_fn=fake_predict_records,
        evaluate_results_fn=fake_eval_results,
    )

    assert result["metrics"]["segm/AP"] == 1.0
    assert (output_dir / "coco_instances_results.json").exists()
    assert (output_dir / "metrics.cocoeval.json").exists()
    assert (output_dir / "metadata.json").exists()
    assert (output_dir / "last_checkpoint").exists()
    assert (output_dir / "wall_time_sec.txt").exists()
    assert (output_dir / "params_trainable.txt").exists()


def test_cellpose_runner_precompute_only_writes_cache_summary(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    dataset_root = tmp_path / "ecc"
    _write_min_ecc_rgb_dataset(dataset_root)
    output_dir = tmp_path / "out"
    cache_dir = tmp_path / "cache"
    script = repo_root / "baselines" / "run_cellpose_instance_ecc.py"

    import subprocess
    import sys

    subprocess.run(
        [
            sys.executable,
            str(script),
            "--dataset-root",
            str(dataset_root),
            "--output-dir",
            str(output_dir),
            "--image-size",
            "512",
            "--target-cache-dir",
            str(cache_dir),
            "--precompute-targets-only",
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )

    assert (output_dir / "target_cache_summary.json").is_file()
    assert len(list(cache_dir.glob("*.npz"))) == 1
