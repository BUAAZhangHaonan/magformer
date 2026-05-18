#!/usr/bin/env python3
"""Resolve R122 formal-eval checkpoint and artifact paths."""

from __future__ import annotations

import argparse
import re
import shlex
import sys
from dataclasses import dataclass
from pathlib import Path


R122_DIR = Path("output/vc_suda/r122_depth_boundary_w001_r114warm_pseudo300")
DIAGNOSTICS_DIR = Path("output/diagnostics")
DEFAULT_ARTIFACT_DATE = "20260518"
SUPPORTED_LABELS = ("iter0099", "final", "latest")
_CHECKPOINT_RE = re.compile(r"^checkpoint_iter_(\d{7})\.pth$")


class R122CheckpointSelectionError(RuntimeError):
    """Raised when a requested R122 checkpoint cannot be selected."""


@dataclass(frozen=True)
class R122FormalEvalArtifacts:
    label: str
    checkpoint: Path
    remaining75_dir: Path
    val28_dir: Path
    bucket_compare_dir: Path
    go_no_go_json: Path

    @property
    def remaining75_metrics(self) -> Path:
        return self.remaining75_dir / "metrics.cocoeval.json"

    @property
    def val28_metrics(self) -> Path:
        return self.val28_dir / "metrics.cocoeval.json"

    @property
    def bucket_compare_csv(self) -> Path:
        return self.bucket_compare_dir / "bucket_compare.csv"

    @property
    def bucket_compare_summary(self) -> Path:
        return self.bucket_compare_dir / "summary.json"


def _checkpoint_for_iter(iteration: int) -> Path:
    return R122_DIR / f"checkpoint_iter_{iteration:07d}.pth"


def _known_checkpoint_iterations(repo_root: Path) -> list[int]:
    checkpoint_dir = repo_root / R122_DIR
    if not checkpoint_dir.exists():
        return []
    iterations: list[int] = []
    for path in checkpoint_dir.glob("checkpoint_iter_*.pth"):
        match = _CHECKPOINT_RE.match(path.name)
        if match:
            iterations.append(int(match.group(1)))
    return sorted(iterations)


def _resolve_checkpoint(repo_root: Path, label: str, checkpoint_path: Path | None, require_checkpoint: bool) -> Path:
    if checkpoint_path is not None:
        path = checkpoint_path if checkpoint_path.is_absolute() else repo_root / checkpoint_path
        if not path.exists():
            raise R122CheckpointSelectionError(f"missing requested R122 checkpoint: {path}")
        return path

    if label == "iter0099":
        path = repo_root / _checkpoint_for_iter(99)
        if require_checkpoint and not path.exists():
            raise R122CheckpointSelectionError(f"missing requested R122 checkpoint: {path}")
        return path

    if label in {"final", "latest"}:
        iterations = _known_checkpoint_iterations(repo_root)
        if not iterations:
            raise R122CheckpointSelectionError(f"no R122 checkpoints found in {repo_root / R122_DIR}")
        path = repo_root / _checkpoint_for_iter(iterations[-1])
        if require_checkpoint and not path.exists():
            raise R122CheckpointSelectionError(f"missing requested R122 checkpoint: {path}")
        return path

    supported = ", ".join(SUPPORTED_LABELS)
    raise R122CheckpointSelectionError(f"unknown R122 checkpoint label {label!r}; supported labels: {supported}")


def resolve_r122_formal_eval_artifacts(
    repo_root: Path,
    *,
    checkpoint_label: str = "iter0099",
    checkpoint_path: Path | None = None,
    artifact_date: str = DEFAULT_ARTIFACT_DATE,
    require_checkpoint: bool = True,
) -> R122FormalEvalArtifacts:
    repo_root = repo_root.resolve()
    label = checkpoint_label.strip().lower()
    checkpoint = _resolve_checkpoint(repo_root, label, checkpoint_path, require_checkpoint=require_checkpoint)
    prefix = f"r122_depth_boundary_w001_{label}"
    go_no_go_prefix = "r122_depth_boundary_w001" if label == "iter0099" else prefix
    return R122FormalEvalArtifacts(
        label=label,
        checkpoint=checkpoint,
        remaining75_dir=DIAGNOSTICS_DIR / f"{prefix}_remaining75_1024_backmap_topk200_{artifact_date}",
        val28_dir=DIAGNOSTICS_DIR / f"{prefix}_val28_1024_backmap_topk200_{artifact_date}",
        bucket_compare_dir=DIAGNOSTICS_DIR / f"{prefix}_bucket_compare_{artifact_date}",
        go_no_go_json=DIAGNOSTICS_DIR / f"{go_no_go_prefix}_go_no_go_{artifact_date}" / "go_no_go.json",
    )


def _shell_line(name: str, value: Path | str) -> str:
    return f"{name}={shlex.quote(str(value))}"


def render_shell_assignments(artifacts: R122FormalEvalArtifacts) -> str:
    lines = [
        _shell_line("R122_CHECKPOINT_LABEL", artifacts.label),
        _shell_line("R122_CHECKPOINT", artifacts.checkpoint),
        _shell_line("REMAINING75_DIR", artifacts.remaining75_dir),
        _shell_line("VAL28_DIR", artifacts.val28_dir),
        _shell_line("BUCKET_COMPARE_DIR", artifacts.bucket_compare_dir),
        _shell_line("BUCKET_COMPARE", artifacts.bucket_compare_csv),
        _shell_line("BUCKET_SUMMARY", artifacts.bucket_compare_summary),
        _shell_line("GO_NO_GO_JSON", artifacts.go_no_go_json),
        _shell_line("GO_NO_GO_DIR", artifacts.go_no_go_json.parent),
    ]
    return "\n".join(lines) + "\n"


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--checkpoint-label", default="iter0099", choices=SUPPORTED_LABELS)
    parser.add_argument("--checkpoint-path", type=Path)
    parser.add_argument("--artifact-date", default=DEFAULT_ARTIFACT_DATE)
    parser.add_argument("--require-checkpoint", action="store_true")
    parser.add_argument("--format", choices=("json", "shell"), default="json")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        artifacts = resolve_r122_formal_eval_artifacts(
            args.repo_root,
            checkpoint_label=args.checkpoint_label,
            checkpoint_path=args.checkpoint_path,
            artifact_date=args.artifact_date,
            require_checkpoint=args.require_checkpoint,
        )
    except R122CheckpointSelectionError as exc:
        sys.stderr.write(f"ERROR: {exc}\n")
        return 2

    if args.format == "shell":
        sys.stdout.write(render_shell_assignments(artifacts))
    else:
        import json

        payload = {
            "label": artifacts.label,
            "checkpoint": str(artifacts.checkpoint),
            "remaining75_dir": str(artifacts.remaining75_dir),
            "val28_dir": str(artifacts.val28_dir),
            "bucket_compare_dir": str(artifacts.bucket_compare_dir),
            "go_no_go_json": str(artifacts.go_no_go_json),
        }
        sys.stdout.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
