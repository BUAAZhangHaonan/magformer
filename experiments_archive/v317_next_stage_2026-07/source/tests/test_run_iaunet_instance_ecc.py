from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

from PIL import Image


def _load_module():
    repo_root = Path(__file__).resolve().parents[1]
    path = repo_root / "baselines" / "run_iaunet_instance_ecc.py"
    spec = importlib.util.spec_from_file_location("run_iaunet_instance_ecc", path)
    assert spec is not None
    assert spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


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


def test_iaunet_runner_resumes_from_best_epoch_when_only_best_checkpoint_exists(tmp_path: Path) -> None:
    output_dir = tmp_path / "iaunet_run"
    output_dir.mkdir(parents=True, exist_ok=True)
    for epoch, segm_ap in enumerate([0.10, 0.20, 0.30, 0.50, 0.35, 0.40], start=1):
        (output_dir / f"epoch_{epoch:04d}_results.json").write_text(
            json.dumps({"segm/AP": segm_ap}),
            encoding="utf-8",
        )
    (output_dir / "model_best.pth").write_bytes(b"best")

    mod = _load_module()
    resume_state = mod._resolve_resume_state(output_dir)
    assert resume_state["resume_epoch"] == 5
    assert resume_state["resume_checkpoint"] == output_dir / "model_best.pth"
    assert resume_state["best_epoch"] == 4
    assert resume_state["best_ap"] == 0.50
    assert [row["epoch"] for row in resume_state["existing_metrics"]] == [1, 2, 3, 4]
    assert resume_state["stale_epochs"] == [5, 6]


def test_iaunet_runner_resumes_from_final_checkpoint_when_available(tmp_path: Path) -> None:
    output_dir = tmp_path / "iaunet_run"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"epoch": 1, "segm/AP": 0.10}),
                json.dumps({"epoch": 2, "segm/AP": 0.20}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (output_dir / "model_final.pth").write_bytes(b"final")

    mod = _load_module()
    resume_state = mod._resolve_resume_state(output_dir)
    assert resume_state["resume_epoch"] == 3
    assert resume_state["resume_checkpoint"] == output_dir / "model_final.pth"
    assert resume_state["best_epoch"] == 2
    assert resume_state["stale_epochs"] == []


def test_iaunet_runner_prefers_trainer_state_for_optimizer_resume(tmp_path: Path) -> None:
    output_dir = tmp_path / "iaunet_run"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.jsonl").write_text(json.dumps({"epoch": 1, "segm/AP": 0.10}) + "\n", encoding="utf-8")
    (output_dir / "model_final.pth").write_bytes(b"final")
    trainer_state = output_dir / "trainer_state_latest.pth"
    trainer_state.write_bytes(b"trainer")

    mod = _load_module()
    resume_state = mod._resolve_resume_state(output_dir)

    assert resume_state["resume_trainer_state"] == trainer_state


def test_iaunet_eval_cadence_for_100_epochs_is_exactly_five_evals() -> None:
    mod = _load_module()
    eval_epochs = [
        epoch
        for epoch in range(1, 101)
        if mod._should_eval_epoch(epoch=epoch, epochs=100, eval_every=20)
    ]

    assert eval_epochs == [20, 40, 60, 80, 100]


def test_iaunet_build_loader_kwargs_enables_persistent_workers() -> None:
    mod = _load_module()
    kwargs = mod.build_loader_kwargs(num_workers=4, use_cuda=True)

    assert kwargs["num_workers"] == 4
    assert kwargs["pin_memory"] is True
    assert kwargs["persistent_workers"] is True
    assert kwargs["prefetch_factor"] == 1
    assert callable(kwargs["worker_init_fn"])

    single_process_kwargs = mod.build_loader_kwargs(num_workers=0, use_cuda=True)
    assert single_process_kwargs["persistent_workers"] is False
    assert "prefetch_factor" not in single_process_kwargs
    assert "worker_init_fn" not in single_process_kwargs


def test_iaunet_dataset_keeps_records_lightweight_and_decodes_masks_lazily(tmp_path: Path) -> None:
    mod = _load_module()
    dataset_root = tmp_path / "ecc"
    _write_min_dataset(dataset_root)

    dataset = mod.ECCIAUNetDataset(str(dataset_root), "train", 512, train=False)

    assert "annotations" in dataset.records[0]
    assert "annotation_targets" not in dataset.records[0]
    sample = dataset[0]
    assert sample["image"].shape == (3, 512, 512)
    assert sample["target"]["masks"].shape == (1, 512, 512)


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
            "4",
            "--transformer-blocks-per-stage",
            "1",
            "--grad-accum-steps",
            "2",
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )

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
            "2",
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
            "4",
            "--transformer-blocks-per-stage",
            "1",
            "--grad-accum-steps",
            "2",
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )

    assert (output_dir / "model_final.pth").is_file()
    assert (output_dir / "model_best.pth").is_file()
    assert not (output_dir / "trainer_state_latest.pth").exists()
    assert not (output_dir / "trainer_state_best.pth").exists()
    assert not (output_dir / "trainer_state_final.pth").exists()
    assert (output_dir / "coco_instances_results.json").is_file()
    assert (output_dir / "metrics.cocoeval.json").is_file()
    assert (output_dir / "metadata.json").is_file()
    assert (output_dir / "last_checkpoint").is_file()
    assert (output_dir / "wall_time_sec.txt").is_file()
    assert (output_dir / "params_trainable.txt").is_file()
    assert len((output_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()) == 2
    assert (output_dir / "epoch_0002_results.json").is_file()

    metrics = json.loads((output_dir / "metrics.cocoeval.json").read_text(encoding="utf-8"))
    predictions = json.loads((output_dir / "coco_instances_results.json").read_text(encoding="utf-8"))
    metadata = json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))

    assert "segm/AP" in metrics
    assert isinstance(predictions, list)
    assert metadata["model_id"] == "iaunet"
    assert metadata["model_name"] == "iaunet"
    assert metadata["image_size"] == 512
    assert metadata["grad_accum_steps"] == 2
