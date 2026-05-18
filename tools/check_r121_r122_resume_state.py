#!/usr/bin/env python3
"""Check the readonly R121/R122 resume state from documented runbook artifacts.

Default R122 evaluation artifact paths:
- remaining75 metrics: output/diagnostics/r122_depth_boundary_w001_iter0099_remaining75_1024_backmap_topk200_20260518/metrics.cocoeval.json
- val28 metrics: output/diagnostics/r122_depth_boundary_w001_iter0099_val28_1024_backmap_topk200_20260518/metrics.cocoeval.json
- bucket compare: output/diagnostics/r122_depth_boundary_w001_iter0099_bucket_compare_20260518/bucket_compare.csv
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


WATCHER_LOG = Path("output/diagnostics/r125_cuda_resume_r121_watcher_20260518.log")
R121_DIR = Path("output/vc_suda/r121_depth_boundary_w001_smoke_b1_iter1_loss_only_fg00075")
R121_TRAIN_LOG = R121_DIR / "train.log"
R121_RETRY_GLOB = "output/vc_suda/r121_depth_boundary_w001_smoke_b1_iter1_loss_only_fg00075.retry_*.tmux.log"
R122_DIR = Path("output/vc_suda/r122_depth_boundary_w001_r114warm_pseudo300")
R122_CHECKPOINT = R122_DIR / "checkpoint_iter_0000099.pth"
R122_REMAINING75_METRICS = (
    Path("output/diagnostics")
    / "r122_depth_boundary_w001_iter0099_remaining75_1024_backmap_topk200_20260518"
    / "metrics.cocoeval.json"
)
R122_VAL28_METRICS = (
    Path("output/diagnostics")
    / "r122_depth_boundary_w001_iter0099_val28_1024_backmap_topk200_20260518"
    / "metrics.cocoeval.json"
)
R122_BUCKET_COMPARE = (
    Path("output/diagnostics")
    / "r122_depth_boundary_w001_iter0099_bucket_compare_20260518"
    / "bucket_compare.csv"
)
GO_NO_GO_JSON = Path("output/diagnostics/r122_depth_boundary_w001_go_no_go_20260518/go_no_go.json")
R122_PROCESS_KEYWORDS = (
    "r122_depth_boundary_w001_r114warm_pseudo300",
    "configs/vc_suda/r122_depth_boundary_w001_r114warm_pseudo300",
    str(R122_DIR),
)


class ResumeStateError(RuntimeError):
    """Raised for invalid or ambiguous inspection inputs."""


@dataclass(frozen=True)
class TrainProcess:
    pid: int
    command: str

    def to_jsonable(self) -> dict[str, Any]:
        return {"pid": self.pid, "command": self.command}

    def summary(self) -> str:
        command = self.command
        if len(command) > 240:
            command = command[:237] + "..."
        return f"pid={self.pid} cmd={command}"


@dataclass(frozen=True)
class Inspection:
    state: str
    reasons: list[str]
    next_action: str
    paths: dict[str, str]
    go_no_go: dict[str, Any] | None = None
    train_processes: list[TrainProcess] | None = None

    def to_jsonable(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "state": self.state,
            "reasons": self.reasons,
            "next_action": self.next_action,
            "paths": self.paths,
        }
        if self.go_no_go is not None:
            payload["go_no_go"] = self.go_no_go
        if self.train_processes:
            payload["train_processes"] = [process.to_jsonable() for process in self.train_processes]
        return payload


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        raise ResumeStateError(f"failed to read {path}: {exc}") from exc


def _has_cuda_unknown(text: str) -> bool:
    lowered = text.lower()
    return "cuda unknown error" in lowered or ("unknown error" in lowered and "cuda" in lowered)


def _detect_failure_reasons(text: str) -> list[str]:
    lowered = text.lower()
    reasons: list[str] = []
    if "outofmemory" in lowered or "cuda out of memory" in lowered or "out of memory" in lowered or "oom" in lowered:
        reasons.append("OOM detected in R121 log")
    if re.search(r"(^|[^a-z])nan([^a-z]|$)", lowered):
        reasons.append("NaN detected in R121 log")
    if "depth sanity failed" in lowered or "depth sanity fail" in lowered:
        reasons.append("depth sanity failed in R121 log")
    if "loss_depth_boundary missing" in lowered or "missing loss_depth_boundary" in lowered:
        reasons.append("loss_depth_boundary missing in R121 log")
    if _has_cuda_unknown(text):
        reasons.append("CUDA unknown error detected in R121 log")
    return reasons


def _r121_passed(text: str) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if "loss_depth_boundary" not in text:
        reasons.append("loss_depth_boundary missing in R121 train.log")
    lowered = text.lower()
    has_one_iter = bool(re.search(r"iter\s*[:=]\s*0\s*/\s*1", text)) or bool(
        re.search(r"max_iter\s*[:=]\s*1", lowered)
    )
    has_completion = "total training time" in lowered or "max_iter" in lowered or "completed 1 iter" in lowered
    if not (has_one_iter and has_completion):
        reasons.append("R121 train.log does not show a completed 1-iter run")
    return not reasons, reasons


def _single_retry_log(repo_root: Path) -> Path | None:
    matches = sorted(repo_root.glob(R121_RETRY_GLOB))
    if len(matches) > 1:
        names = ", ".join(str(path.relative_to(repo_root)) for path in matches)
        raise ResumeStateError(f"multiple R121 retry logs match {R121_RETRY_GLOB}: {names}")
    return matches[0] if matches else None


def _load_go_no_go(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(_read_text(path))
    except json.JSONDecodeError as exc:
        raise ResumeStateError(f"{path}: invalid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ResumeStateError(f"{path}: expected JSON object")
    if "decision" not in payload:
        raise ResumeStateError(f"{path}: missing required field decision")
    decision = str(payload["decision"]).upper()
    if decision not in {"PASS", "FAIL"}:
        raise ResumeStateError(f"{path}: decision must be PASS or FAIL, got {payload['decision']!r}")
    if "passed" in payload and not isinstance(payload["passed"], bool):
        raise ResumeStateError(f"{path}: passed must be boolean when present")
    passed = bool(payload["passed"]) if "passed" in payload else decision == "PASS"
    return {"decision": decision, "pass": passed, "path": str(path)}


def _current_user_processes() -> list[TrainProcess]:
    try:
        result = subprocess.run(
            ["ps", "-u", str(os.getuid()), "-o", "pid=,args="],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as exc:
        raise ResumeStateError(f"failed to inspect current user processes with ps: {exc}") from exc
    if result.returncode != 0:
        stderr = result.stderr.strip()
        raise ResumeStateError(f"failed to inspect current user processes with ps: {stderr or 'exit ' + str(result.returncode)}")

    processes: list[TrainProcess] = []
    for line in result.stdout.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        match = re.match(r"^(\d+)\s+(.*)$", stripped)
        if not match:
            continue
        processes.append(TrainProcess(pid=int(match.group(1)), command=match.group(2)))
    return processes


def _is_training_command(command: str) -> bool:
    lowered = command.lower()
    return "train.py" in lowered or "torchrun" in lowered or "torch.distributed.run" in lowered


def _is_r122_training_process(process: TrainProcess) -> bool:
    lowered = process.command.lower()
    if not _is_training_command(lowered):
        return False
    return any(keyword.lower() in lowered for keyword in R122_PROCESS_KEYWORDS)


def _r122_training_processes() -> list[TrainProcess]:
    return [process for process in _current_user_processes() if _is_r122_training_process(process)]


def inspect_resume_state(repo_root: Path) -> Inspection:
    repo_root = repo_root.resolve()
    paths = {
        "watcher_log": str(repo_root / WATCHER_LOG),
        "r121_output_dir": str(repo_root / R121_DIR),
        "r121_train_log": str(repo_root / R121_TRAIN_LOG),
        "r121_retry_glob": str(repo_root / R121_RETRY_GLOB),
        "r122_train_dir": str(repo_root / R122_DIR),
        "r122_checkpoint": str(repo_root / R122_CHECKPOINT),
        "remaining75_metrics": str(repo_root / R122_REMAINING75_METRICS),
        "val28_metrics": str(repo_root / R122_VAL28_METRICS),
        "bucket_compare": str(repo_root / R122_BUCKET_COMPARE),
        "go_no_go_json": str(repo_root / GO_NO_GO_JSON),
    }

    retry_log = _single_retry_log(repo_root)
    if retry_log is not None:
        retry_text = _read_text(retry_log)
        failure_reasons = _detect_failure_reasons(retry_text)
        if not failure_reasons:
            raise ResumeStateError(f"{retry_log}: retry log has no recognized pass/fail signal")
        return Inspection(
            state="R121_FAILED",
            reasons=[f"{reason}: {retry_log}" for reason in failure_reasons],
            next_action="Stop before R122. Read the R121 retry log and resolve the listed R121/CUDA failure.",
            paths=paths,
        )

    watcher_path = repo_root / WATCHER_LOG
    r121_dir = repo_root / R121_DIR
    if watcher_path.exists() and not r121_dir.exists():
        watcher_text = _read_text(watcher_path)
        if _has_cuda_unknown(watcher_text):
            return Inspection(
                state="BLOCKED_CUDA",
                reasons=[f"CUDA unknown error found in watcher log: {watcher_path}", f"R121 output dir is absent: {r121_dir}"],
                next_action="Do not start R121/R122. Wait for NVIDIA driver or host CUDA recovery, then rerun R124 CUDA probes.",
                paths=paths,
            )

    r121_train_log = repo_root / R121_TRAIN_LOG
    if not r121_train_log.exists():
        return Inspection(
            state="NEED_R121_TRAIN",
            reasons=[f"R121 train.log is missing: {r121_train_log}"],
            next_action="Run only the R121 smoke after the CUDA probes pass.",
            paths=paths,
        )

    r121_text = _read_text(r121_train_log)
    failure_reasons = _detect_failure_reasons(r121_text)
    if failure_reasons:
        return Inspection(
            state="R121_FAILED",
            reasons=[f"{reason}: {r121_train_log}" for reason in failure_reasons],
            next_action="Stop before R122. Fix the documented R121 failure condition first.",
            paths=paths,
        )
    r121_ok, missing_reasons = _r121_passed(r121_text)
    if not r121_ok:
        return Inspection(
            state="R121_FAILED",
            reasons=missing_reasons,
            next_action="Stop before R122. R121 smoke must finish 1 iter and print loss_depth_boundary.",
            paths=paths,
        )

    checkpoint = repo_root / R122_CHECKPOINT
    if not checkpoint.exists():
        return Inspection(
            state="NEED_R122_TRAIN",
            reasons=["R121 smoke passed with loss_depth_boundary", f"R122 checkpoint is missing: {checkpoint}"],
            next_action="Run R122 300iter only after CUDA probes still pass.",
            paths=paths,
        )

    train_processes = _r122_training_processes()
    if train_processes:
        return Inspection(
            state="R122_TRAINING",
            reasons=[
                f"R122 checkpoint exists, but matching R122 training process is still running: {process.summary()}"
                for process in train_processes
            ],
            next_action="Wait for the R122 training process to exit before running remaining75/val28 evaluation.",
            paths=paths,
            train_processes=train_processes,
        )

    missing_eval = [
        label
        for label, relative in [
            ("remaining75 metrics", R122_REMAINING75_METRICS),
            ("val28 metrics", R122_VAL28_METRICS),
        ]
        if not (repo_root / relative).exists()
    ]
    if missing_eval:
        return Inspection(
            state="NEED_R122_EVAL",
            reasons=[f"missing {label}" for label in missing_eval],
            next_action="Run the documented R122 remaining75 and val28 evaluations for checkpoint_iter_0000099.pth.",
            paths=paths,
        )

    bucket_compare = repo_root / R122_BUCKET_COMPARE
    if not bucket_compare.exists():
        return Inspection(
            state="NEED_R122_EVAL",
            reasons=[f"missing bucket_compare.csv: {bucket_compare}"],
            next_action="Generate the R120-atlas-style bucket_compare.csv before go/no-go.",
            paths=paths,
        )

    go_no_go_path = repo_root / GO_NO_GO_JSON
    if not go_no_go_path.exists():
        return Inspection(
            state="NEED_GO_NO_GO",
            reasons=[f"go_no_go.json is missing: {go_no_go_path}"],
            next_action="Run tools/compare_r122_go_no_go.py with the documented R120 baseline and R122 bucket_compare.csv.",
            paths=paths,
        )

    go_no_go = _load_go_no_go(go_no_go_path)
    return Inspection(
        state="READY_TO_DECIDE",
        reasons=[f"go/no-go decision is available: {go_no_go_path}"],
        next_action="Read the go/no-go report and decide whether to continue or stop.",
        paths=paths,
        go_no_go=go_no_go,
    )


def _render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# R121/R122 Resume State",
        "",
        "| Field | Value |",
        "| --- | --- |",
        f"| State | {payload['state']} |",
        "",
        "## Reasons",
    ]
    lines.extend(f"- {reason}" for reason in payload["reasons"])
    lines.extend(["", "## Next Action", "", str(payload["next_action"])])
    if "go_no_go" in payload:
        lines.extend(
            [
                "",
                "## Go/No-Go",
                "",
                f"- decision: {payload['go_no_go']['decision']}",
                f"- pass: {str(payload['go_no_go']['pass']).lower()}",
            ]
        )
    if "train_processes" in payload:
        lines.extend(["", "## Train Processes"])
        for process in payload["train_processes"]:
            lines.append(f"- pid={process['pid']} cmd=`{process['command']}`")
    lines.extend(["", "## Checked Paths"])
    for key, value in payload["paths"].items():
        lines.append(f"- {key}: `{value}`")
    lines.append("")
    return "\n".join(lines)


def _write_output(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Readonly CPU-only R121/R122 resume state checker. Default eval paths are "
            "output/diagnostics/r122_depth_boundary_w001_iter0099_remaining75_1024_backmap_topk200_20260518/metrics.cocoeval.json, "
            "output/diagnostics/r122_depth_boundary_w001_iter0099_val28_1024_backmap_topk200_20260518/metrics.cocoeval.json, "
            "and output/diagnostics/r122_depth_boundary_w001_iter0099_bucket_compare_20260518/bucket_compare.csv."
        )
    )
    parser.add_argument("--repo-root", default=".", type=Path, help="Repository root to inspect. Defaults to current directory.")
    parser.add_argument("--output-json", type=Path, help="Optional path for the JSON report.")
    parser.add_argument("--output-md", type=Path, help="Optional path for the Markdown report.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    args = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        inspection = inspect_resume_state(args.repo_root)
        payload = inspection.to_jsonable()
        json_text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
        if args.output_json:
            _write_output(args.output_json, json_text)
        if args.output_md:
            _write_output(args.output_md, _render_markdown(payload))
        sys.stdout.write(json_text)
        return 0
    except ResumeStateError as exc:
        sys.stderr.write(f"ERROR: {exc}\n")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
