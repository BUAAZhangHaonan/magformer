from __future__ import annotations

import json
import sys
from pathlib import Path

from scripts.analysis.backfill_live_inference_profiles import run_backfill, select_backfill_targets


def _write_metadata(out_dir: Path, dataset_root: Path, model_id: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metadata.json").write_text(
        json.dumps(
            {
                "dataset_root": str(dataset_root.resolve()),
                "model_id": model_id,
            }
        ),
        encoding="utf-8",
    )


def test_select_backfill_targets_filters_and_dedupes_realpaths(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()

    real_out = tmp_path / "suite" / "magformer_nodpth_ref_fair"
    _write_metadata(real_out, dataset_root, "magformer_nodpth_ref_fair")
    alias_out = tmp_path / "suite" / "magformer_nodpth_ref"
    alias_out.symlink_to(real_out, target_is_directory=True)

    ready_out = tmp_path / "suite" / "mask2former"
    _write_metadata(ready_out, dataset_root, "mask2former")
    (ready_out / "inference_speed.json").write_text(json.dumps({"status": "ok"}), encoding="utf-8")

    missing_out = tmp_path / "suite" / "ucn"

    rows = [
        {
            "resolution": 1024,
            "model_id": "magformer_nodpth_ref",
            "status": "ok",
            "output_dir": str(alias_out),
            "metadata_path": str(alias_out / "metadata.json"),
            "inference_latency_ms_mean": None,
            "inference_fps": None,
        },
        {
            "resolution": 1024,
            "model_id": "magformer_nodpth_ref_fair",
            "status": "ok",
            "output_dir": str(real_out),
            "metadata_path": str(real_out / "metadata.json"),
            "inference_latency_ms_mean": None,
            "inference_fps": None,
        },
        {
            "resolution": 512,
            "model_id": "mask2former",
            "status": "ok",
            "output_dir": str(ready_out),
            "metadata_path": str(ready_out / "metadata.json"),
            "inference_latency_ms_mean": None,
            "inference_fps": None,
        },
        {
            "resolution": 256,
            "model_id": "msmformer",
            "status": "ok",
            "output_dir": str(real_out),
            "metadata_path": str(real_out / "metadata.json"),
            "inference_latency_ms_mean": None,
            "inference_fps": None,
        },
        {
            "resolution": 1024,
            "model_id": "ucn",
            "status": "ok",
            "output_dir": str(missing_out),
            "metadata_path": str(missing_out / "metadata.json"),
            "inference_latency_ms_mean": None,
            "inference_fps": None,
        },
        {
            "resolution": 512,
            "model_id": "uoais",
            "status": "missing",
            "output_dir": str(real_out),
            "metadata_path": str(real_out / "metadata.json"),
            "inference_latency_ms_mean": None,
            "inference_fps": None,
        },
    ]

    targets = select_backfill_targets(rows)

    assert len(targets) == 1
    assert targets[0].model_id == "magformer_nodpth_ref"
    assert targets[0].output_dir == real_out.resolve()
    assert targets[0].dataset_root == dataset_root.resolve()


def test_run_backfill_continues_on_error_by_default(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()

    ok_out = tmp_path / "suite" / "ok_model"
    fail_out = tmp_path / "suite" / "fail_model"
    _write_metadata(ok_out, dataset_root, "ok_model")
    _write_metadata(fail_out, dataset_root, "fail_model")

    rows = [
        {
            "resolution": 1024,
            "model_id": "ok_model",
            "status": "ok",
            "output_dir": str(ok_out),
            "metadata_path": str(ok_out / "metadata.json"),
            "inference_latency_ms_mean": None,
            "inference_fps": None,
        },
        {
            "resolution": 512,
            "model_id": "fail_model",
            "status": "ok",
            "output_dir": str(fail_out),
            "metadata_path": str(fail_out / "metadata.json"),
            "inference_latency_ms_mean": None,
            "inference_fps": None,
        },
    ]

    fake_benchmark = tmp_path / "fake_benchmark.py"
    fake_benchmark.write_text(
        "\n".join(
            [
                "from __future__ import annotations",
                "import argparse",
                "import json",
                "from pathlib import Path",
                "",
                "parser = argparse.ArgumentParser()",
                "parser.add_argument('--out-dir', required=True)",
                "parser.add_argument('--dataset-root', required=True)",
                "parser.add_argument('--device', required=True)",
                "parser.add_argument('--warmup', required=True)",
                "parser.add_argument('--timed-images', required=True)",
                "parser.add_argument('--output-name', required=True)",
                "args = parser.parse_args()",
                "out_dir = Path(args.out_dir)",
                "if 'fail_model' in out_dir.name:",
                "    raise SystemExit(3)",
                "(out_dir / args.output_name).write_text(json.dumps({'status': 'ok'}), encoding='utf-8')",
                "print(json.dumps({'out_dir': str(out_dir), 'dataset_root': args.dataset_root}))",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    targets = select_backfill_targets(rows)
    results = run_backfill(
        targets,
        python_executable=sys.executable,
        benchmark_script=fake_benchmark,
        device="cpu",
        warmup=1,
        timed_images=1,
    )

    assert [result.returncode for result in results] == [0, 3]
    assert (ok_out / "inference_speed.json").exists()
    assert not (fail_out / "inference_speed.json").exists()
