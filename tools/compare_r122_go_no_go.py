#!/usr/bin/env python3
"""Compare R114 baseline and R122 candidate bucket metrics for go/no-go gating."""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


class CompareError(RuntimeError):
    """Raised when bucket CSV inputs are missing required data."""


@dataclass(frozen=True)
class GateSpec:
    name: str
    bucket_type: str
    bucket: str
    metric: str
    source_column: str
    mode: str
    threshold: float


@dataclass(frozen=True)
class GateResult:
    name: str
    bucket_type: str
    bucket: str
    metric: str
    baseline: float
    candidate: float
    delta: float
    threshold: str
    passed: bool


METRIC_COLUMNS = {
    "boundary_f": "boundary_f_mean",
    "best_iou": "best_iou_mean",
    "r75": "recall75",
    "fp75": "fp75",
}
REQUIRED_COLUMNS = {"run", "bucket_type", "bucket", *METRIC_COLUMNS.values()}
KEY = tuple[str, str]


def _build_gate_specs(args: argparse.Namespace) -> list[GateSpec]:
    protection = args.protection_delta_min
    return [
        GateSpec(
            "remaining75 tiny boundary_f improvement",
            "area_bucket",
            "tiny_area_le_256",
            "boundary_f",
            METRIC_COLUMNS["boundary_f"],
            "delta_min",
            args.tiny_boundary_f_delta,
        ),
        GateSpec(
            "remaining75 tiny best_iou improvement",
            "area_bucket",
            "tiny_area_le_256",
            "best_iou",
            METRIC_COLUMNS["best_iou"],
            "delta_min",
            args.tiny_best_iou_delta,
        ),
        GateSpec(
            "remaining75 tiny r75 floor",
            "area_bucket",
            "tiny_area_le_256",
            "r75",
            METRIC_COLUMNS["r75"],
            "candidate_min",
            args.tiny_r75_min,
        ),
        GateSpec(
            "remaining75 high100 boundary_f improvement",
            "density_bucket",
            "high100",
            "boundary_f",
            METRIC_COLUMNS["boundary_f"],
            "delta_min",
            args.high100_boundary_f_delta,
        ),
        GateSpec(
            "remaining75 high100 r75 floor",
            "density_bucket",
            "high100",
            "r75",
            METRIC_COLUMNS["r75"],
            "candidate_min",
            args.high100_r75_min,
        ),
        GateSpec(
            "remaining75 high100 fp75 non-increase",
            "density_bucket",
            "high100",
            "fp75",
            METRIC_COLUMNS["fp75"],
            "delta_max",
            args.high100_fp75_delta_max,
        ),
        GateSpec(
            "remaining75 small boundary_f protection",
            "area_bucket",
            "small_257_1024",
            "boundary_f",
            METRIC_COLUMNS["boundary_f"],
            "delta_min",
            protection,
        ),
        GateSpec(
            "remaining75 small best_iou protection",
            "area_bucket",
            "small_257_1024",
            "best_iou",
            METRIC_COLUMNS["best_iou"],
            "delta_min",
            protection,
        ),
        GateSpec(
            "remaining75 small r75 protection",
            "area_bucket",
            "small_257_1024",
            "r75",
            METRIC_COLUMNS["r75"],
            "delta_min",
            protection,
        ),
        GateSpec(
            "remaining75 mid50 boundary_f protection",
            "density_bucket",
            "mid50",
            "boundary_f",
            METRIC_COLUMNS["boundary_f"],
            "delta_min",
            protection,
        ),
        GateSpec(
            "remaining75 mid50 best_iou protection",
            "density_bucket",
            "mid50",
            "best_iou",
            METRIC_COLUMNS["best_iou"],
            "delta_min",
            protection,
        ),
        GateSpec(
            "remaining75 mid50 r75 protection",
            "density_bucket",
            "mid50",
            "r75",
            METRIC_COLUMNS["r75"],
            "delta_min",
            protection,
        ),
    ]


def _read_bucket_csv(path: Path, run_name: str | None) -> dict[KEY, dict[str, str]]:
    if not path.exists():
        raise CompareError(f"missing input file: {path}")
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise CompareError(f"{path} is empty")
        columns = set(reader.fieldnames)
        for column in sorted(REQUIRED_COLUMNS - columns):
            raise CompareError(f"{path}: missing required column {column}")
        rows: dict[KEY, dict[str, str]] = {}
        seen_any = False
        for row_number, row in enumerate(reader, start=2):
            if run_name is not None and row["run"] != run_name:
                continue
            seen_any = True
            key = (row["bucket_type"], row["bucket"])
            if key in rows:
                raise CompareError(
                    f"{path}: duplicate bucket {key[0]}/{key[1]}"
                    + (f" for run {run_name}" if run_name else "")
                )
            for column in REQUIRED_COLUMNS:
                if row.get(column) in (None, ""):
                    raise CompareError(f"{path}:{row_number}: missing value for {column}")
            rows[key] = row
    if run_name is not None and not seen_any:
        raise CompareError(f"{path}: missing run {run_name}")
    return rows


def _parse_float(path: Path, row: dict[str, str], column: str) -> float:
    try:
        value = float(row[column])
    except ValueError as exc:
        raise CompareError(
            f"{path}: non-numeric value for {row['bucket_type']}/{row['bucket']} {column}: {row[column]}"
        ) from exc
    if not math.isfinite(value):
        raise CompareError(
            f"{path}: non-finite value for {row['bucket_type']}/{row['bucket']} {column}: {row[column]}"
        )
    return value


def _threshold_text(spec: GateSpec) -> str:
    if spec.mode == "delta_min":
        return f"delta >= {spec.threshold:.6f}"
    if spec.mode == "delta_max":
        return f"delta <= {spec.threshold:.6f}"
    if spec.mode == "candidate_min":
        return f"candidate >= {spec.threshold:.6f}"
    raise AssertionError(f"unknown gate mode: {spec.mode}")


def _passes(spec: GateSpec, candidate: float, delta: float) -> bool:
    if spec.mode == "delta_min":
        return delta >= spec.threshold
    if spec.mode == "delta_max":
        return delta <= spec.threshold
    if spec.mode == "candidate_min":
        return candidate >= spec.threshold
    raise AssertionError(f"unknown gate mode: {spec.mode}")


def compare(
    baseline_csv: Path,
    candidate_csv: Path,
    gate_specs: list[GateSpec],
    baseline_run: str | None = None,
    candidate_run: str | None = None,
) -> dict[str, object]:
    baseline_rows = _read_bucket_csv(baseline_csv, baseline_run)
    candidate_rows = _read_bucket_csv(candidate_csv, candidate_run)
    results: list[GateResult] = []
    for spec in gate_specs:
        key = (spec.bucket_type, spec.bucket)
        if key not in baseline_rows:
            raise CompareError(f"{baseline_csv}: missing required bucket {key[0]}/{key[1]}")
        if key not in candidate_rows:
            raise CompareError(f"{candidate_csv}: missing required bucket {key[0]}/{key[1]}")
        baseline = _parse_float(baseline_csv, baseline_rows[key], spec.source_column)
        candidate = _parse_float(candidate_csv, candidate_rows[key], spec.source_column)
        delta = round(candidate - baseline, 12)
        results.append(
            GateResult(
                name=spec.name,
                bucket_type=spec.bucket_type,
                bucket=spec.bucket,
                metric=spec.metric,
                baseline=baseline,
                candidate=candidate,
                delta=delta,
                threshold=_threshold_text(spec),
                passed=_passes(spec, candidate, delta),
            )
        )
    decision = "PASS" if all(result.passed for result in results) else "FAIL"
    return {
        "decision": decision,
        "baseline_bucket_csv": str(baseline_csv),
        "candidate_bucket_csv": str(candidate_csv),
        "baseline_run": baseline_run,
        "candidate_run": candidate_run,
        "gates": [result.__dict__ for result in results],
    }


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _fmt(value: object) -> str:
    if isinstance(value, bool):
        return "PASS" if value else "FAIL"
    if isinstance(value, float):
        return f"{value:.6f}"
    if value is None:
        return ""
    return str(value)


def _write_markdown(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    gates = payload["gates"]
    assert isinstance(gates, list)
    lines = [
        "# R122 Go/No-Go Comparison",
        "",
        "| field | value |",
        "| --- | --- |",
        f"| decision | {payload['decision']} |",
        f"| baseline_bucket_csv | {payload['baseline_bucket_csv']} |",
        f"| candidate_bucket_csv | {payload['candidate_bucket_csv']} |",
        f"| baseline_run | {_fmt(payload['baseline_run'])} |",
        f"| candidate_run | {_fmt(payload['candidate_run'])} |",
        "",
        "| gate | bucket_type | bucket | metric | baseline | candidate | delta | threshold | result |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | --- | --- |",
    ]
    for gate in gates:
        assert isinstance(gate, dict)
        lines.append(
            "| {name} | {bucket_type} | {bucket} | {metric} | {baseline} | {candidate} | "
            "{delta} | {threshold} | {passed} |".format(
                name=_fmt(gate["name"]),
                bucket_type=_fmt(gate["bucket_type"]),
                bucket=_fmt(gate["bucket"]),
                metric=_fmt(gate["metric"]),
                baseline=_fmt(gate["baseline"]),
                candidate=_fmt(gate["candidate"]),
                delta=_fmt(gate["delta"]),
                threshold=_fmt(gate["threshold"]),
                passed=_fmt(gate["passed"]),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be >= 0")
    return parsed


def _float(value: str) -> float:
    return float(value)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare R114 baseline and R122 candidate bucket_compare.csv gates."
    )
    parser.add_argument("--baseline-bucket-csv", required=True, type=Path)
    parser.add_argument("--candidate-bucket-csv", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-md", required=True, type=Path)
    parser.add_argument("--baseline-run", help="Optional run value to select from a multi-run CSV.")
    parser.add_argument("--candidate-run", help="Optional run value to select from a multi-run CSV.")
    parser.add_argument("--tiny-boundary-f-delta", type=_positive_float, default=0.02)
    parser.add_argument("--tiny-best-iou-delta", type=_positive_float, default=0.02)
    parser.add_argument("--tiny-r75-min", type=_positive_float, default=0.035)
    parser.add_argument("--high100-boundary-f-delta", type=_positive_float, default=0.02)
    parser.add_argument("--high100-r75-min", type=_positive_float, default=0.270)
    parser.add_argument("--high100-fp75-delta-max", type=_float, default=0.0)
    parser.add_argument(
        "--protection-delta-min",
        type=_float,
        default=0.0,
        help="Minimum allowed delta for small_257_1024 and mid50 protection gates.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        payload = compare(
            args.baseline_bucket_csv,
            args.candidate_bucket_csv,
            _build_gate_specs(args),
            baseline_run=args.baseline_run,
            candidate_run=args.candidate_run,
        )
    except CompareError as exc:
        parser.exit(2, f"error: {exc}\n")
    _write_json(args.output_json, payload)
    _write_markdown(args.output_md, payload)
    return 0 if payload["decision"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
