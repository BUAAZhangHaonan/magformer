#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, List, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = REPO_ROOT / "output" / "analysis" / "2026-04-10-live-metrics-manifest-fresh.json"
BENCHMARK_SCRIPT = REPO_ROOT / "scripts" / "analysis" / "benchmark_inference.py"
ALLOWED_RESOLUTIONS = {1024, 512}


@dataclass(frozen=True)
class BackfillTarget:
    resolution: int
    model_id: str
    output_dir: Path
    dataset_root: Path
    metadata_path: Path


@dataclass(frozen=True)
class BackfillResult:
    target: BackfillTarget
    command: List[str]
    returncode: int
    stdout: str
    stderr: str


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _as_int(value: Any) -> int | None:
    try:
        return int(value)
    except Exception:
        return None


def _row_has_inference_metrics(row: dict[str, Any]) -> bool:
    return row.get("inference_latency_ms_mean") is not None and row.get("inference_fps") is not None


def _output_has_inference_artifact(output_dir: Path) -> bool:
    return (output_dir / "inference_speed.json").exists() or (output_dir / "inference_speed_clean.json").exists()


def _resolve_metadata_path(row: dict[str, Any], output_dir: Path) -> Path:
    metadata_path = row.get("metadata_path")
    if isinstance(metadata_path, str) and metadata_path.strip():
        return Path(metadata_path).resolve()
    return (output_dir / "metadata.json").resolve()


def _resolve_dataset_root(row: dict[str, Any], output_dir: Path) -> Path:
    metadata_path = _resolve_metadata_path(row, output_dir)
    payload = _load_json(metadata_path)
    dataset_root = payload.get("dataset_root")
    if not isinstance(dataset_root, str) or not dataset_root.strip():
        raise ValueError(f"Missing dataset_root in metadata: {metadata_path}")
    return Path(dataset_root).resolve()


def select_backfill_targets(
    rows: Sequence[dict[str, Any]],
    *,
    model_ids: Iterable[str] | None = None,
    limit: int = 0,
) -> List[BackfillTarget]:
    requested_model_ids = {model_id for model_id in (model_ids or []) if model_id}
    seen_output_dirs: set[Path] = set()
    targets: List[BackfillTarget] = []

    for row in rows:
        resolution = _as_int(row.get("resolution"))
        if resolution not in ALLOWED_RESOLUTIONS:
            continue
        if row.get("status") != "ok":
            continue
        model_id = str(row.get("model_id") or "").strip()
        if not model_id:
            continue
        if requested_model_ids and model_id not in requested_model_ids:
            continue
        if _row_has_inference_metrics(row):
            continue

        output_dir_raw = str(row.get("output_dir") or "").strip()
        if not output_dir_raw:
            continue
        output_dir = Path(output_dir_raw).resolve()
        if not output_dir.exists():
            continue
        if _output_has_inference_artifact(output_dir):
            continue
        if output_dir in seen_output_dirs:
            continue

        metadata_path = _resolve_metadata_path(row, output_dir)
        dataset_root = _resolve_dataset_root(row, output_dir)
        targets.append(
            BackfillTarget(
                resolution=resolution,
                model_id=model_id,
                output_dir=output_dir,
                dataset_root=dataset_root,
                metadata_path=metadata_path,
            )
        )
        seen_output_dirs.add(output_dir)
        if limit > 0 and len(targets) >= limit:
            break

    return targets


def run_backfill(
    targets: Sequence[BackfillTarget],
    *,
    python_executable: str = sys.executable,
    benchmark_script: Path = BENCHMARK_SCRIPT,
    device: str = "cuda",
    warmup: int = 10,
    timed_images: int = 50,
    output_name: str = "inference_speed.json",
    continue_on_error: bool = True,
) -> List[BackfillResult]:
    results: List[BackfillResult] = []
    benchmark_script = benchmark_script.resolve()

    for target in targets:
        command = [
            python_executable,
            str(benchmark_script),
            "--out-dir",
            str(target.output_dir),
            "--dataset-root",
            str(target.dataset_root),
            "--device",
            str(device),
            "--warmup",
            str(int(warmup)),
            "--timed-images",
            str(int(timed_images)),
            "--output-name",
            str(output_name),
        ]
        proc = subprocess.run(command, check=False, capture_output=True, text=True)
        result = BackfillResult(
            target=target,
            command=command,
            returncode=int(proc.returncode),
            stdout=proc.stdout,
            stderr=proc.stderr,
        )
        results.append(result)
        if proc.returncode != 0 and not continue_on_error:
            raise subprocess.CalledProcessError(
                proc.returncode,
                command,
                output=proc.stdout,
                stderr=proc.stderr,
            )

    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backfill live inference profiles from an existing manifest")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--model-id", action="append", default=[], help="Limit to one or more model ids")
    parser.add_argument("--limit", type=int, default=0, help="Maximum number of targets to process")
    parser.add_argument("--device", default="cuda", help="Device string passed to benchmark_inference.py")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup images for benchmark_inference.py")
    parser.add_argument("--timed-images", type=int, default=50, help="Timed images for benchmark_inference.py")
    parser.add_argument("--output-name", default="inference_speed.json", help="Benchmark output filename")
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop on the first benchmark failure instead of continuing",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest).resolve()
    rows = _load_json(manifest_path)
    if not isinstance(rows, list):
        raise SystemExit(f"Manifest must be a JSON list: {manifest_path}")

    targets = select_backfill_targets(rows, model_ids=args.model_id, limit=int(args.limit))
    print(f"[backfill] selected {len(targets)} targets from {manifest_path}")
    for target in targets:
        print(f"[backfill] target {target.resolution} {target.model_id} -> {target.output_dir}")

    results = run_backfill(
        targets,
        device=str(args.device),
        warmup=int(args.warmup),
        timed_images=int(args.timed_images),
        output_name=str(args.output_name),
        continue_on_error=not bool(args.fail_fast),
    )

    failures = 0
    for result in results:
        status = "ok" if result.returncode == 0 else f"failed({result.returncode})"
        print(f"[backfill] {status} {result.target.model_id} -> {result.target.output_dir}")
        if result.stdout.strip():
            print(result.stdout.strip())
        if result.stderr.strip():
            print(result.stderr.strip(), file=sys.stderr)
        if result.returncode != 0:
            failures += 1

    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
