from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from PIL import Image


def _write_split(root: Path, split: str, image_id: int) -> None:
    (root / "images" / split).mkdir(parents=True, exist_ok=True)
    image_name = f"{split}_{image_id:06d}.png"
    Image.new("RGB", (32, 32), color=(24, 80, 140)).save(root / "images" / split / image_name)
    payload = {
        "images": [{"id": image_id, "file_name": image_name, "width": 32, "height": 32}],
        "annotations": [
            {
                "id": image_id,
                "image_id": image_id,
                "category_id": 1,
                "segmentation": [[4, 4, 18, 4, 18, 18, 4, 18]],
                "area": 196,
                "bbox": [4, 4, 14, 14],
                "iscrowd": 0,
            }
        ],
        "categories": [{"id": 1, "name": "component"}],
    }
    (root / "annotations").mkdir(parents=True, exist_ok=True)
    (root / "annotations" / f"instances_{split}.json").write_text(json.dumps(payload), encoding="utf-8")


def _write_min_dataset(root: Path) -> None:
    _write_split(root, "train", 1)
    _write_split(root, "val", 2)


def test_iaunet_runner_smoke_writes_standard_artifacts(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    dataset_root = tmp_path / "ecc"
    output_dir = tmp_path / "iaunet_run"
    _write_min_dataset(dataset_root)

    script = repo_root / "baselines" / "run_iaunet_instance_ecc.py"
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
            "--epochs",
            "1",
            "--batch",
            "1",
            "--num-workers",
            "0",
            "--device",
            "cpu",
            "--max-train-steps",
            "1",
            "--max-val-images",
            "1",
            "--base-channels",
            "8",
            "--hidden-dim",
            "32",
            "--num-queries",
            "8",
            "--num-decoder-layers",
            "2",
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )

    assert (output_dir / "model_final.pth").is_file()
    assert (output_dir / "model_best.pth").is_file()
    assert (output_dir / "coco_instances_results.json").is_file()
    assert (output_dir / "metrics.cocoeval.json").is_file()
    assert (output_dir / "metadata.json").is_file()
    assert (output_dir / "last_checkpoint").is_file()
    assert (output_dir / "wall_time_sec.txt").is_file()
    assert (output_dir / "params_trainable.txt").is_file()

    metrics = json.loads((output_dir / "metrics.cocoeval.json").read_text(encoding="utf-8"))
    predictions = json.loads((output_dir / "coco_instances_results.json").read_text(encoding="utf-8"))
    metadata = json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))

    assert "segm/AP" in metrics
    assert isinstance(predictions, list)
    assert metadata["model_id"] == "iaunet"
    assert metadata["model_name"] == "iaunet"
    assert metadata["image_size"] == 512
