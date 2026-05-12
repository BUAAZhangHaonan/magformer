from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


def _write_annotations(dataset_root: Path) -> None:
    ann_dir = dataset_root / "annotations"
    ann_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "images": [{"id": 1, "file_name": "0001.png", "width": 16, "height": 16}],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "iscrowd": 0,
                "area": 16.0,
                "bbox": [2.0, 2.0, 4.0, 4.0],
                "segmentation": [[2.0, 2.0, 6.0, 2.0, 6.0, 6.0, 2.0, 6.0]],
            }
        ],
        "categories": [{"id": 1, "name": "component"}],
    }
    (ann_dir / "instances_val.json").write_text(json.dumps(payload), encoding="utf-8")


def test_build_full19_live_metrics_manifest_reads_live_artifacts_and_marks_missing_rows(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "analysis" / "build_full19_live_metrics_manifest.py"

    fake_repo = tmp_path / "repo"
    experiments_root = fake_repo / "output" / "experiments"
    dataset_root = fake_repo / "magformer_datasets" / "20260318_1K_1566"
    _write_annotations(dataset_root)

    model_dir = experiments_root / "20260318_1k_1566_20ep_1024_full19" / "mask2former"
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "metrics.cocoeval.json").write_text(
        json.dumps(
            {
                "iteration": 10,
                "segm/AP": 58.7554,
                "segm/AP50": 80.5301,
                "segm/AP75": 66.0488,
                "bbox/AP": 50.5990,
                "bbox/AP50": 77.8280,
                "bbox/AP75": 58.2970,
            }
        ),
        encoding="utf-8",
    )
    (model_dir / "coco_instances_results.json").write_text(
        json.dumps(
            [
                {
                    "image_id": 1,
                    "category_id": 1,
                    "score": 0.95,
                    "bbox": [2.0, 2.0, 4.0, 4.0],
                    "segmentation": [[2.0, 2.0, 6.0, 2.0, 6.0, 6.0, 2.0, 6.0]],
                }
            ]
        ),
        encoding="utf-8",
    )
    (model_dir / "metadata.json").write_text(
        json.dumps(
            {
                "dataset_root": str(dataset_root.resolve()),
                "command": "bash run_0831_1k_20ep_scratch_official_mask2former.sh --image-size 1024 --pretrained --run",
                "model_id": "official_mask2former_pretrained",
            }
        ),
        encoding="utf-8",
    )
    (model_dir / "wall_time_sec.txt").write_text("6944\n", encoding="utf-8")
    (model_dir / "peak_memory_mb.txt").write_text("35761\n", encoding="utf-8")
    (model_dir / "params_trainable.txt").write_text("44056196\n", encoding="utf-8")
    (model_dir / "inference_speed_clean.json").write_text(
        json.dumps(
            {
                "status": "ok",
                "latency_ms_mean": 66.5025,
                "throughput_fps": 15.0370,
                "inference_peak_memory_mb": 2711.4526,
            }
        ),
        encoding="utf-8",
    )

    out_manifest = tmp_path / "manifest.json"
    subprocess.run(
        [
            sys.executable,
            str(script),
            "--repo-root",
            str(fake_repo),
            "--output",
            str(out_manifest),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    rows = json.loads(out_manifest.read_text(encoding="utf-8"))
    row_1024 = next(row for row in rows if row["resolution"] == 1024 and row["model_id"] == "mask2former")
    row_512 = next(row for row in rows if row["resolution"] == 512 and row["model_id"] == "mask2former")

    assert row_1024["status"] == "ok"
    assert float(row_1024["segm_AP"]) == 58.7554
    assert float(row_1024["bbox_AP"]) == 50.5990
    assert float(row_1024["segm_f1_at_50"]) == pytest.approx(100.0)
    assert row_1024["training_mode"] == "fine-tuned"
    assert row_512["status"] == "missing"
    assert row_512["segm_AP"] is None
    assert ".worktrees" not in json.dumps(rows)


def test_build_full19_live_metrics_manifest_includes_new_external_unet_baselines(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "analysis" / "build_full19_live_metrics_manifest.py"

    fake_repo = tmp_path / "repo"
    out_manifest = tmp_path / "manifest.json"

    subprocess.run(
        [
            sys.executable,
            str(script),
            "--repo-root",
            str(fake_repo),
            "--output",
            str(out_manifest),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    rows = json.loads(out_manifest.read_text(encoding="utf-8"))
    model_ids = {row["model_id"] for row in rows}
    assert {"cellpose", "stardist", "iaunet"} <= model_ids
    for model_id in ["cellpose", "stardist", "iaunet"]:
        model_rows = [row for row in rows if row["model_id"] == model_id]
        assert len(model_rows) == 2
        assert all("implementation_fidelity" in row for row in model_rows)
        assert all("official_code_used" in row for row in model_rows)

    assert next(row for row in rows if row["model_id"] == "cellpose")["implementation_fidelity"] == "official-library"
    assert next(row for row in rows if row["model_id"] == "iaunet")["implementation_fidelity"] == "experimental-local-reimplementation"
    assert next(row for row in rows if row["model_id"] == "stardist")["implementation_fidelity"] == "official-library"


def test_build_full19_live_metrics_manifest_falls_back_to_metadata_timestamps_for_wall_time(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "analysis" / "build_full19_live_metrics_manifest.py"

    fake_repo = tmp_path / "repo"
    experiments_root = fake_repo / "output" / "experiments"
    dataset_root = fake_repo / "magformer_datasets" / "20260318_1K_1566"
    _write_annotations(dataset_root)

    model_dir = experiments_root / "20260318_1k_1566_20ep_1024_full19" / "magformer_depthnorm_on"
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "metrics.cocoeval.json").write_text(
        json.dumps(
            {
                "iteration": 10,
                "segm/AP": 68.4177,
                "segm/AP50": 87.9741,
                "segm/AP75": 77.7181,
                "bbox/AP": 62.2810,
                "bbox/AP50": 82.9465,
                "bbox/AP75": 69.2393,
            }
        ),
        encoding="utf-8",
    )
    (model_dir / "coco_instances_results.json").write_text("[]", encoding="utf-8")
    (model_dir / "metadata.json").write_text(
        json.dumps(
            {
                "dataset_root": str(dataset_root.resolve()),
                "start_time_iso": "2026-03-21T14:07:17+08:00",
                "end_time_iso": "2026-03-21T16:44:50+08:00",
                "wall_time_sec": None,
            }
        ),
        encoding="utf-8",
    )

    out_manifest = tmp_path / "manifest.json"
    subprocess.run(
        [
            sys.executable,
            str(script),
            "--repo-root",
            str(fake_repo),
            "--output",
            str(out_manifest),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    rows = json.loads(out_manifest.read_text(encoding="utf-8"))
    row = next(row for row in rows if row["resolution"] == 1024 and row["model_id"] == "magformer_depthnorm_on")
    assert float(row["train_wall_time_sec"]) == 9453.0


def test_build_full19_live_metrics_manifest_prefers_resolution_matching_candidate(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "analysis" / "build_full19_live_metrics_manifest.py"

    fake_repo = tmp_path / "repo"
    experiments_root = fake_repo / "output" / "experiments"
    dataset_root = fake_repo / "magformer_datasets" / "20260318_1K_1566"
    _write_annotations(dataset_root)

    top_level = experiments_root / "20260318_1k_1566_20ep_1024_full19" / "msmformer_scratch"
    backup = experiments_root / "20260318_1k_1566_20ep_1024_full19" / "_backup_fix_20260324_normrepair" / "msmformer"
    stray = experiments_root / "20260318_1k_1566_20ep_1024_full19" / "visualizations" / "msmformer"
    stray.mkdir(parents=True, exist_ok=True)
    for model_dir, image_size in ((top_level, 512), (backup, 1024)):
        model_dir.mkdir(parents=True, exist_ok=True)
        (model_dir / "metrics.cocoeval.json").write_text(
            json.dumps(
                {
                    "iteration": -1,
                    "segm/AP": 0.0,
                    "segm/AP50": 0.0,
                    "segm/AP75": 0.0,
                    "bbox/AP": 0.0,
                    "bbox/AP50": 0.0,
                    "bbox/AP75": 0.0,
                }
            ),
            encoding="utf-8",
        )
        (model_dir / "coco_instances_results.json").write_text("[]", encoding="utf-8")
        (model_dir / "metadata.json").write_text(
            json.dumps(
                {
                    "dataset_root": str(dataset_root.resolve()),
                    "command": (
                        "bash run_0831_1k_20ep_scratch_msmformer.sh "
                        f"--register 20260318_1K_1566 --dataset-root {dataset_root.resolve()} "
                        f"--output-root {experiments_root.resolve()} --candidate-id C1 --run-tag final "
                        f"--image-size {image_size} --run"
                    ),
                }
            ),
            encoding="utf-8",
        )

    out_manifest = tmp_path / "manifest.json"
    subprocess.run(
        [
            sys.executable,
            str(script),
            "--repo-root",
            str(fake_repo),
            "--output",
            str(out_manifest),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    rows = json.loads(out_manifest.read_text(encoding="utf-8"))
    row = next(row for row in rows if row["resolution"] == 1024 and row["model_id"] == "msmformer")
    assert row["status"] == "ok"
    assert row["output_dir"].endswith("_backup_fix_20260324_normrepair/msmformer")
    assert "multiple live artifact candidates found" in row["note"]


def test_build_full19_live_metrics_manifest_accepts_fair_magformer_nodpth_alias(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "analysis" / "build_full19_live_metrics_manifest.py"

    fake_repo = tmp_path / "repo"
    experiments_root = fake_repo / "output" / "experiments"
    dataset_root = fake_repo / "magformer_datasets" / "20260318_1K_1566"
    _write_annotations(dataset_root)

    model_dir = experiments_root / "20260406_1k_1566_20ep_1024_full19" / "magformer_nodpth_ref_fair"
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "metrics.cocoeval.json").write_text(
        json.dumps(
            {
                "iteration": 10,
                "segm/AP": 57.1234,
                "segm/AP50": 80.0,
                "segm/AP75": 63.0,
                "bbox/AP": 52.0,
                "bbox/AP50": 78.0,
                "bbox/AP75": 59.0,
            }
        ),
        encoding="utf-8",
    )
    (model_dir / "coco_instances_results.json").write_text("[]", encoding="utf-8")
    (model_dir / "metadata.json").write_text(
        json.dumps(
            {
                "dataset_root": str(dataset_root.resolve()),
                "command": (
                    "bash run_0831_1k_20ep_1024_revisit_magformer.sh "
                    f"--register 20260318_1K_1566 --dataset-root {dataset_root.resolve()} "
                    f"--output-root {experiments_root.resolve()} --variant nodpth_ref_fair --run"
                ),
                "model_id": "magformer_nodpth_ref_fair",
            }
        ),
        encoding="utf-8",
    )

    out_manifest = tmp_path / "manifest.json"
    subprocess.run(
        [
            sys.executable,
            str(script),
            "--repo-root",
            str(fake_repo),
            "--output",
            str(out_manifest),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    rows = json.loads(out_manifest.read_text(encoding="utf-8"))
    row = next(row for row in rows if row["resolution"] == 1024 and row["model_id"] == "magformer_nodpth_ref")
    assert row["status"] == "ok"
    assert float(row["segm_AP"]) == 57.1234
    assert row["output_dir"].endswith("magformer_nodpth_ref_fair")


def test_build_full19_live_metrics_manifest_prefers_latest_fair_magformer_nodpth_candidate(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "analysis" / "build_full19_live_metrics_manifest.py"

    fake_repo = tmp_path / "repo"
    experiments_root = fake_repo / "output" / "experiments"
    dataset_root = fake_repo / "magformer_datasets" / "20260318_1K_1566"
    _write_annotations(dataset_root)

    old_dir = experiments_root / "20260318_1k_1566_20ep_1024_full19" / "magformer_nodpth_ref"
    fair_dir = experiments_root / "20260406_1k_1566_20ep_1024_full19" / "magformer_nodpth_ref_fair"

    for model_dir, segm_ap in ((old_dir, 48.7786), (fair_dir, 59.5226)):
        model_dir.mkdir(parents=True, exist_ok=True)
        (model_dir / "metrics.cocoeval.json").write_text(
            json.dumps(
                {
                    "iteration": -1,
                    "segm/AP": segm_ap,
                    "segm/AP50": 80.0,
                    "segm/AP75": 60.0,
                    "bbox/AP": segm_ap - 5.0,
                    "bbox/AP50": 78.0,
                    "bbox/AP75": 55.0,
                }
            ),
            encoding="utf-8",
        )
        (model_dir / "coco_instances_results.json").write_text("[]", encoding="utf-8")
        (model_dir / "metadata.json").write_text(
            json.dumps(
                {
                    "dataset_root": str(dataset_root.resolve()),
                    "command": (
                        "bash run_0831_1k_20ep_1024_revisit_magformer.sh "
                        f"--register 20260318_1K_1566 --dataset-root {dataset_root.resolve()} "
                        f"--output-root {experiments_root.resolve()} --variant {model_dir.name} --run --image-size 1024"
                    ),
                    "model_id": model_dir.name,
                }
            ),
            encoding="utf-8",
        )

    out_manifest = tmp_path / "manifest.json"
    subprocess.run(
        [
            sys.executable,
            str(script),
            "--repo-root",
            str(fake_repo),
            "--output",
            str(out_manifest),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    rows = json.loads(out_manifest.read_text(encoding="utf-8"))
    row = next(row for row in rows if row["resolution"] == 1024 and row["model_id"] == "magformer_nodpth_ref")
    assert row["status"] == "ok"
    assert float(row["segm_AP"]) == 59.5226
    assert row["output_dir"].endswith("magformer_nodpth_ref_fair")
    assert "multiple live artifact candidates found" not in row["note"]
