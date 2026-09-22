from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_benchmark_inference_ucn_falls_back_to_run_log(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "analysis" / "benchmark_inference.py"
    out_dir = tmp_path / "ucn_scratch"
    out_dir.mkdir(parents=True)
    (out_dir / "metadata.json").write_text(json.dumps({"model_id": "ucn_scratch"}), encoding="utf-8")
    (out_dir / "run.log").write_text(
        "[2026-03-01 00:00:00] [ucn-eval] 110/110 images, elapsed=17.8s, results=1246\n",
        encoding="utf-8",
    )
    dataset_root = tmp_path / "ds"
    dataset_root.mkdir()

    subprocess.run(
        [
            sys.executable,
            str(script),
            "--out-dir",
            str(out_dir),
            "--dataset-root",
            str(dataset_root),
            "--device",
            "cpu",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads((out_dir / "inference_speed.json").read_text(encoding="utf-8"))
    assert payload["status"] == "ok"
    assert payload["source"] == "ucn_eval_from_log"
    assert payload["timed_images"] == 110


def test_benchmark_inference_ucn_falls_back_to_run_log_when_dataset_layout_breaks(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "analysis" / "benchmark_inference.py"
    out_dir = tmp_path / "ucn_scratch"
    out_dir.mkdir(parents=True)
    (out_dir / "metadata.json").write_text(json.dumps({"model_id": "ucn_scratch"}), encoding="utf-8")
    (out_dir / "run.log").write_text(
        "[2026-03-01 00:00:00] [ucn-eval] 149/149 images, elapsed=53.4s, results=7000\n",
        encoding="utf-8",
    )

    dataset_root = tmp_path / "ds"
    ann_dir = dataset_root / "annotations"
    image_dir = dataset_root / "images" / "val"
    depth_dir = dataset_root / "depth" / "depth_npy" / "val"
    ann_dir.mkdir(parents=True)
    image_dir.mkdir(parents=True)
    depth_dir.mkdir(parents=True)
    (ann_dir / "instances_val.json").write_text(
        json.dumps(
            {
                "images": [{"id": 1, "file_name": "0001.png", "width": 16, "height": 16}],
                "annotations": [],
                "categories": [{"id": 1, "name": "component"}],
            }
        ),
        encoding="utf-8",
    )

    subprocess.run(
        [
            sys.executable,
            str(script),
            "--out-dir",
            str(out_dir),
            "--dataset-root",
            str(dataset_root),
            "--device",
            "cpu",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads((out_dir / "inference_speed.json").read_text(encoding="utf-8"))
    assert payload["status"] == "ok"
    assert payload["source"] == "ucn_eval_from_log"
    assert payload["timed_images"] == 149
