from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path


HEADER = [
    "run",
    "role",
    "bucket_type",
    "bucket",
    "images",
    "gt",
    "pred",
    "best_iou_mean",
    "boundary_f_mean",
    "recall75",
    "fp75",
]


BASELINE_ROWS = [
    {
        "run": "r114_remaining75",
        "role": "held-out",
        "bucket_type": "area_bucket",
        "bucket": "tiny_area_le_256",
        "images": 69,
        "gt": 1070,
        "pred": 2014,
        "best_iou_mean": 0.30,
        "boundary_f_mean": 0.46,
        "recall75": 0.020,
        "fp75": 1989,
    },
    {
        "run": "r114_remaining75",
        "role": "held-out",
        "bucket_type": "density_bucket",
        "bucket": "high100",
        "images": 17,
        "gt": 1691,
        "pred": 2876,
        "best_iou_mean": 0.50,
        "boundary_f_mean": 0.53,
        "recall75": 0.250,
        "fp75": 2459,
    },
    {
        "run": "r114_remaining75",
        "role": "held-out",
        "bucket_type": "area_bucket",
        "bucket": "small_257_1024",
        "images": 75,
        "gt": 3294,
        "pred": 5264,
        "best_iou_mean": 0.73,
        "boundary_f_mean": 0.68,
        "recall75": 0.580,
        "fp75": 3338,
    },
    {
        "run": "r114_remaining75",
        "role": "held-out",
        "bucket_type": "density_bucket",
        "bucket": "mid50",
        "images": 49,
        "gt": 2448,
        "pred": 4173,
        "best_iou_mean": 0.70,
        "boundary_f_mean": 0.67,
        "recall75": 0.540,
        "fp75": 2830,
    },
]


def _write_bucket_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=HEADER)
        writer.writeheader()
        writer.writerows(rows)


def _candidate_rows(**overrides: dict[str, object]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    defaults = {
        ("area_bucket", "tiny_area_le_256"): {
            "best_iou_mean": 0.33,
            "boundary_f_mean": 0.49,
            "recall75": 0.036,
            "fp75": 1980,
        },
        ("density_bucket", "high100"): {
            "best_iou_mean": 0.51,
            "boundary_f_mean": 0.56,
            "recall75": 0.275,
            "fp75": 2459,
        },
        ("area_bucket", "small_257_1024"): {
            "best_iou_mean": 0.73,
            "boundary_f_mean": 0.68,
            "recall75": 0.580,
            "fp75": 3300,
        },
        ("density_bucket", "mid50"): {
            "best_iou_mean": 0.70,
            "boundary_f_mean": 0.67,
            "recall75": 0.540,
            "fp75": 2800,
        },
    }
    for baseline in BASELINE_ROWS:
        key = (str(baseline["bucket_type"]), str(baseline["bucket"]))
        row = dict(baseline)
        row["run"] = "r122_remaining75"
        row.update(defaults[key])
        row.update(overrides.get(f"{key[0]}:{key[1]}", {}))
        rows.append(row)
    return rows


def _run_compare(tmp_path: Path, candidate_rows: list[dict[str, object]]) -> subprocess.CompletedProcess[str]:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "tools" / "compare_r122_go_no_go.py"
    baseline_csv = tmp_path / "baseline_bucket_compare.csv"
    candidate_csv = tmp_path / "candidate_bucket_compare.csv"
    output_json = tmp_path / "go_no_go.json"
    output_md = tmp_path / "go_no_go.md"
    _write_bucket_csv(baseline_csv, BASELINE_ROWS)
    _write_bucket_csv(candidate_csv, candidate_rows)
    return subprocess.run(
        [
            sys.executable,
            str(script),
            "--baseline-bucket-csv",
            str(baseline_csv),
            "--candidate-bucket-csv",
            str(candidate_csv),
            "--output-json",
            str(output_json),
            "--output-md",
            str(output_md),
        ],
        capture_output=True,
        text=True,
        check=False,
    )


def test_r122_go_no_go_passes_when_tiny_high100_and_protection_gates_pass(tmp_path: Path) -> None:
    result = _run_compare(tmp_path, _candidate_rows())

    assert result.returncode == 0, result.stderr
    payload = json.loads((tmp_path / "go_no_go.json").read_text(encoding="utf-8"))
    assert payload["decision"] == "PASS"
    assert all(gate["passed"] for gate in payload["gates"])
    tiny_iou = [
        gate
        for gate in payload["gates"]
        if gate["bucket"] == "tiny_area_le_256" and gate["metric"] == "best_iou"
    ][0]
    assert tiny_iou["baseline"] == 0.30
    assert tiny_iou["candidate"] == 0.33
    assert tiny_iou["delta"] == 0.03
    assert tiny_iou["threshold"] == "delta >= 0.020000"
    assert "| decision | PASS |" in (tmp_path / "go_no_go.md").read_text(encoding="utf-8")


def test_r122_go_no_go_fails_when_high100_fp75_rises(tmp_path: Path) -> None:
    result = _run_compare(
        tmp_path,
        _candidate_rows(**{"density_bucket:high100": {"fp75": 2460}}),
    )

    assert result.returncode == 1
    payload = json.loads((tmp_path / "go_no_go.json").read_text(encoding="utf-8"))
    assert payload["decision"] == "FAIL"
    high_fp = [
        gate
        for gate in payload["gates"]
        if gate["bucket"] == "high100" and gate["metric"] == "fp75"
    ][0]
    assert high_fp["passed"] is False
    assert high_fp["threshold"] == "delta <= 0.000000"


def test_r122_go_no_go_fails_when_tiny_gates_do_not_clear_threshold(tmp_path: Path) -> None:
    result = _run_compare(
        tmp_path,
        _candidate_rows(
            **{
                "area_bucket:tiny_area_le_256": {
                    "best_iou_mean": 0.31,
                    "boundary_f_mean": 0.47,
                    "recall75": 0.034,
                }
            }
        ),
    )

    assert result.returncode == 1
    payload = json.loads((tmp_path / "go_no_go.json").read_text(encoding="utf-8"))
    failed = [gate for gate in payload["gates"] if not gate["passed"]]
    assert {gate["metric"] for gate in failed} >= {"boundary_f", "best_iou", "r75"}


def test_r122_go_no_go_errors_on_missing_required_bucket(tmp_path: Path) -> None:
    result = _run_compare(
        tmp_path,
        [
            row
            for row in _candidate_rows()
            if not (row["bucket_type"] == "density_bucket" and row["bucket"] == "high100")
        ],
    )

    assert result.returncode == 2
    assert "missing required bucket density_bucket/high100" in result.stderr


def test_r122_go_no_go_errors_on_missing_required_metric(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "tools" / "compare_r122_go_no_go.py"
    baseline_csv = tmp_path / "baseline_bucket_compare.csv"
    candidate_csv = tmp_path / "candidate_bucket_compare.csv"
    output_json = tmp_path / "go_no_go.json"
    output_md = tmp_path / "go_no_go.md"
    _write_bucket_csv(baseline_csv, BASELINE_ROWS)
    with candidate_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[column for column in HEADER if column != "recall75"])
        writer.writeheader()
        for row in _candidate_rows():
            item = dict(row)
            item.pop("recall75")
            writer.writerow(item)

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--baseline-bucket-csv",
            str(baseline_csv),
            "--candidate-bucket-csv",
            str(candidate_csv),
            "--output-json",
            str(output_json),
            "--output-md",
            str(output_md),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    assert "missing required column recall75" in result.stderr


def test_r122_go_no_go_errors_on_duplicate_bucket(tmp_path: Path) -> None:
    result = _run_compare(tmp_path, _candidate_rows() + [_candidate_rows()[0]])

    assert result.returncode == 2
    assert "duplicate bucket area_bucket/tiny_area_le_256" in result.stderr
