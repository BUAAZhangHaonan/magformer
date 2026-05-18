#!/usr/bin/env python3
"""Audit VC SUDA final goal evidence without running experiments."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
from pathlib import Path
from typing import Callable, Iterable

SEGM_AP_THRESHOLD = 0.61
FIRST50_METRICS = Path(
    "output/diagnostics/r44_teacher_first50_protocol_fix_20260516/metrics.cocoeval.json"
)
R123_DIR = Path("output/diagnostics/r123_pseudo_real_validity_20260518")
R123_STATS = R123_DIR / "annotation_stats.json"
R123_CONTACT_SHEETS = [
    R123_DIR / "contact_random24.png",
    R123_DIR / "contact_low_count24.png",
    R123_DIR / "contact_tiny24.png",
    R123_DIR / "contact_high_density24.png",
]
R122_GO_NO_GO = Path("output/diagnostics/r122_depth_boundary_w001_go_no_go_20260518/go_no_go.json")
R122_REMAINING75_METRICS = Path(
    "output/diagnostics/r122_depth_boundary_w001_iter0099_remaining75_1024_backmap_topk200_20260518/metrics.cocoeval.json"
)
R122_VAL28_METRICS = Path(
    "output/diagnostics/r122_depth_boundary_w001_iter0099_val28_1024_backmap_topk200_20260518/metrics.cocoeval.json"
)

GitRunner = Callable[[list[str], Path], subprocess.CompletedProcess[str]]
ProcessLister = Callable[[], list[dict[str, str]]]

TRAIN_SCRIPT = "tools/train.py"
EVAL_1024_BACKMAP_SCRIPT = "tools/evaluate_1024_backmap.py"
TORCH_LAUNCHERS = {"torchrun", "torch.distributed.run"}
EXCLUDED_PROCESS_NAMES = {
    "grep",
    "egrep",
    "pytest",
    "py.test",
    "bash",
    "sh",
    "zsh",
    "dash",
}
EXCLUDED_PROCESS_SCRIPTS = {
    "tools/audit_vc_suda_goal.py",
    "tools/check_r121_r122_resume_state.py",
    "tools/run_r128_gated_r122_evaluator.sh",
}


def _rel(path: Path) -> str:
    return path.as_posix()


def _item(item_id: str, label: str, status: str, detail: str, evidence: dict[str, object] | None = None) -> dict[str, object]:
    return {
        "id": item_id,
        "label": label,
        "status": status,
        "detail": detail,
        "evidence": evidence or {},
    }


def _read_json(repo_root: Path, relpath: Path) -> tuple[dict[str, object] | None, str | None]:
    path = repo_root / relpath
    if not path.exists():
        return None, "missing"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return None, f"invalid json: {exc}"
    if not isinstance(payload, dict):
        return None, "json root is not an object"
    return payload, None


def _segm_ap(metrics: dict[str, object] | None) -> float | None:
    if metrics is None:
        return None
    value = metrics.get("segm_AP")
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _run_git(args: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=False)


def _list_processes() -> list[dict[str, str]]:
    result = subprocess.run(["ps", "-eo", "pid=,args="], capture_output=True, text=True, check=False)
    if result.returncode != 0:
        return [{"pid": "", "command": f"PS_ERROR: {result.stderr.strip()}"}]
    processes: list[dict[str, str]] = []
    for line in result.stdout.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        pid, _, command = stripped.partition(" ")
        processes.append({"pid": pid, "command": command.strip()})
    return processes


def _append_missing(missing_inputs: list[str], relpaths: Iterable[Path]) -> None:
    for relpath in relpaths:
        item = _rel(relpath)
        if item not in missing_inputs:
            missing_inputs.append(item)


def _audit_first50(repo_root: Path, missing_inputs: list[str], summary: dict[str, object]) -> dict[str, object]:
    metrics, error = _read_json(repo_root, FIRST50_METRICS)
    if error:
        _append_missing(missing_inputs, [FIRST50_METRICS])
        return _item(
            "first50_protocol_fix",
            "first50 protocol fix segm AP >= 0.61",
            "FAIL",
            f"{_rel(FIRST50_METRICS)}: {error}",
        )
    ap = _segm_ap(metrics)
    summary["first50_segm_AP"] = ap
    if ap is None:
        return _item("first50_protocol_fix", "first50 protocol fix segm AP >= 0.61", "FAIL", "segm_AP missing")
    status = "PASS" if ap >= SEGM_AP_THRESHOLD else "FAIL"
    return _item(
        "first50_protocol_fix",
        "first50 protocol fix segm AP >= 0.61",
        status,
        f"segm_AP={ap:.6f}; threshold={SEGM_AP_THRESHOLD:.2f}",
        {"path": _rel(FIRST50_METRICS), "segm_AP": ap, "threshold": SEGM_AP_THRESHOLD},
    )


def _audit_r123(repo_root: Path, missing_inputs: list[str]) -> dict[str, object]:
    required = [R123_STATS, *R123_CONTACT_SHEETS]
    missing = [relpath for relpath in required if not (repo_root / relpath).exists()]
    if missing:
        _append_missing(missing_inputs, missing)
        return _item(
            "r123_validity_artifacts",
            "R123 validity stats and contact sheets exist",
            "FAIL",
            "missing: " + ", ".join(_rel(path) for path in missing),
            {"required": [_rel(path) for path in required]},
        )
    return _item(
        "r123_validity_artifacts",
        "R123 validity stats and contact sheets exist",
        "PASS",
        "annotation_stats.json and four contact sheets found",
        {"required": [_rel(path) for path in required]},
    )


def _audit_r122_inputs(repo_root: Path, missing_inputs: list[str], summary: dict[str, object]) -> list[dict[str, object]]:
    items: list[dict[str, object]] = []
    go_no_go, go_error = _read_json(repo_root, R122_GO_NO_GO)
    if go_error:
        _append_missing(missing_inputs, [R122_GO_NO_GO])
        items.append(_item("r122_go_no_go", "R122 formal go/no-go exists", "FAIL", f"{_rel(R122_GO_NO_GO)}: {go_error}"))
    else:
        decision = str(go_no_go.get("decision", "")) if go_no_go else ""
        summary["r122_go_no_go_decision"] = decision
        status = "PASS" if decision == "PASS" else "FAIL"
        items.append(
            _item(
                "r122_go_no_go",
                "R122 formal go/no-go decision PASS",
                status,
                f"decision={decision or 'MISSING'}",
                {"path": _rel(R122_GO_NO_GO), "decision": decision},
            )
        )

    for item_id, label, relpath in [
        ("r122_remaining75_metrics", "R122 formal remaining75 metrics exist", R122_REMAINING75_METRICS),
        ("r122_val28_metrics", "R122 formal val28 metrics exist", R122_VAL28_METRICS),
    ]:
        metrics, error = _read_json(repo_root, relpath)
        if error:
            _append_missing(missing_inputs, [relpath])
            items.append(_item(item_id, label, "FAIL", f"{_rel(relpath)}: {error}"))
            continue
        ap = _segm_ap(metrics)
        summary[f"{item_id}_segm_AP"] = ap
        if ap is None:
            items.append(_item(item_id, label, "FAIL", "segm_AP missing", {"path": _rel(relpath)}))
        else:
            items.append(_item(item_id, label, "PASS", f"segm_AP={ap:.6f}", {"path": _rel(relpath), "segm_AP": ap}))
    return items


def _audit_final_target(summary: dict[str, object]) -> dict[str, object]:
    decision = summary.get("r122_go_no_go_decision")
    ap = summary.get("r122_remaining75_metrics_segm_AP")
    passed = decision == "PASS" and isinstance(ap, float) and ap >= SEGM_AP_THRESHOLD
    detail_ap = "missing" if not isinstance(ap, float) else f"{ap:.6f}"
    return _item(
        "final_r122_segm_target",
        "Final R122 remaining75 segm target",
        "PASS" if passed else "FAIL",
        f"go_no_go={decision or 'missing'}; remaining75 segm_AP={detail_ap}; threshold={SEGM_AP_THRESHOLD:.2f}",
        {"required_decision": "PASS", "segm_AP": ap, "threshold": SEGM_AP_THRESHOLD},
    )


def _audit_git(repo_root: Path, git_runner: GitRunner, warnings: list[str]) -> list[dict[str, object]]:
    items: list[dict[str, object]] = []
    dirty = git_runner(["status", "--porcelain"], repo_root)
    if dirty.returncode != 0:
        msg = f"git status failed: {dirty.stderr.strip() or dirty.stdout.strip()}"
        warnings.append(msg)
        items.append(_item("git_dirty", "Git worktree clean", "WARNING", msg))
    elif dirty.stdout.strip():
        changed = dirty.stdout.strip().splitlines()
        items.append(_item("git_dirty", "Git worktree clean", "FAIL", "dirty paths present", {"paths": changed}))
    else:
        items.append(_item("git_dirty", "Git worktree clean", "PASS", "no dirty paths"))

    upstream = git_runner(["rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"], repo_root)
    if upstream.returncode != 0:
        msg = "git upstream not available; unpushed commit count not checked"
        warnings.append(msg)
        items.append(_item("git_unpushed_commits", "No unpushed commits", "WARNING", msg))
        return items
    unpushed = git_runner(["rev-list", "--count", "@{u}..HEAD"], repo_root)
    if unpushed.returncode != 0:
        msg = f"git rev-list failed: {unpushed.stderr.strip() or unpushed.stdout.strip()}"
        warnings.append(msg)
        items.append(_item("git_unpushed_commits", "No unpushed commits", "WARNING", msg))
        return items
    try:
        count = int(unpushed.stdout.strip() or "0")
    except ValueError:
        msg = f"git rev-list returned non-integer count: {unpushed.stdout.strip()}"
        warnings.append(msg)
        items.append(_item("git_unpushed_commits", "No unpushed commits", "WARNING", msg))
        return items
    status = "PASS" if count == 0 else "WARNING"
    if count:
        warnings.append(f"{count} unpushed commit(s) relative to {upstream.stdout.strip()}")
    items.append(
        _item(
            "git_unpushed_commits",
            "No unpushed commits",
            status,
            f"unpushed_commits={count}; upstream={upstream.stdout.strip()}",
            {"count": count, "upstream": upstream.stdout.strip()},
        )
    )
    return items


def _command_tokens(command: str) -> list[str]:
    try:
        return shlex.split(command)
    except ValueError:
        return command.split()


def _token_name(token: str) -> str:
    return Path(token).name


def _is_script_token(token: str, relpath: str) -> bool:
    return token == relpath or token.endswith(f"/{relpath}")


def _has_script_token(tokens: list[str], relpath: str) -> bool:
    return any(_is_script_token(token, relpath) for token in tokens)


def _is_python_executable(token: str) -> bool:
    name = _token_name(token)
    return name == "python" or name.startswith("python")


def _script_invoked_directly(tokens: list[str], relpath: str) -> bool:
    for index, token in enumerate(tokens):
        if not _is_script_token(token, relpath):
            continue
        if index == 0 or _is_python_executable(tokens[index - 1]):
            return True
    return False


def _has_torch_launcher(tokens: list[str]) -> bool:
    return any(_token_name(token) in TORCH_LAUNCHERS or token in TORCH_LAUNCHERS for token in tokens)


def _is_excluded_process(tokens: list[str]) -> bool:
    if not tokens:
        return True
    if _token_name(tokens[0]) in EXCLUDED_PROCESS_NAMES:
        return True
    for token in tokens:
        if token in EXCLUDED_PROCESS_NAMES or _token_name(token) in EXCLUDED_PROCESS_NAMES:
            return True
        if any(_is_script_token(token, script) for script in EXCLUDED_PROCESS_SCRIPTS):
            return True
    return False


def _matches_repo_process(_repo_root: Path, command: str) -> bool:
    if command.startswith("PS_ERROR:"):
        return True
    tokens = _command_tokens(command)
    if _is_excluded_process(tokens):
        return False
    train_invoked = _script_invoked_directly(tokens, TRAIN_SCRIPT)
    eval_invoked = _script_invoked_directly(tokens, EVAL_1024_BACKMAP_SCRIPT)
    train_launched = _has_torch_launcher(tokens) and _has_script_token(tokens, TRAIN_SCRIPT)
    return eval_invoked or train_invoked or train_launched


def _audit_processes(repo_root: Path, process_lister: ProcessLister, warnings: list[str]) -> dict[str, object]:
    processes = process_lister()
    matches = [proc for proc in processes if _matches_repo_process(repo_root, proc.get("command", ""))]
    ps_errors = [proc for proc in matches if proc.get("command", "").startswith("PS_ERROR:")]
    if ps_errors:
        msg = ps_errors[0]["command"]
        warnings.append(msg)
        return _item("repo_train_eval_processes", "No repo train/eval process running", "WARNING", msg)
    if matches:
        detail = f"{len(matches)} train/eval process(es) found"
        return _item("repo_train_eval_processes", "No repo train/eval process running", "FAIL", detail, {"processes": matches})
    return _item("repo_train_eval_processes", "No repo train/eval process running", "PASS", "none found")


def build_audit(
    repo_root: Path,
    git_runner: GitRunner | None = None,
    process_lister: ProcessLister | None = None,
) -> dict[str, object]:
    repo_root = repo_root.resolve()
    git_runner = git_runner or _run_git
    process_lister = process_lister or _list_processes
    missing_inputs: list[str] = []
    warnings: list[str] = []
    summary: dict[str, object] = {"segm_AP_threshold": SEGM_AP_THRESHOLD, "repo_root": str(repo_root)}
    checklist: list[dict[str, object]] = []

    checklist.append(_audit_first50(repo_root, missing_inputs, summary))
    checklist.append(_audit_r123(repo_root, missing_inputs))
    checklist.extend(_audit_r122_inputs(repo_root, missing_inputs, summary))
    checklist.append(_audit_final_target(summary))
    checklist.extend(_audit_git(repo_root, git_runner, warnings))
    checklist.append(_audit_processes(repo_root, process_lister, warnings))

    status = "NOT_COMPLETE" if any(item["status"] == "FAIL" for item in checklist) else "COMPLETE"
    summary["failed_items"] = [item["id"] for item in checklist if item["status"] == "FAIL"]
    summary["warning_items"] = [item["id"] for item in checklist if item["status"] == "WARNING"]
    return {
        "status": status,
        "checklist": checklist,
        "missing_inputs": missing_inputs,
        "warnings": warnings,
        "summary": summary,
    }


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_markdown(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# VC SUDA Final Goal Audit",
        "",
        f"Status: `{payload['status']}`",
        "",
        "## Checklist",
        "",
        "| id | status | detail |",
        "| --- | --- | --- |",
    ]
    for item in payload["checklist"]:
        assert isinstance(item, dict)
        detail = str(item["detail"]).replace("|", "\\|")
        lines.append(f"| {item['id']} | {item['status']} | {detail} |")
    lines.extend(["", "## Missing Inputs", ""])
    missing = payload["missing_inputs"]
    assert isinstance(missing, list)
    if missing:
        lines.extend(f"- `{path}`" for path in missing)
    else:
        lines.append("- none")
    lines.extend(["", "## Warnings", ""])
    warnings = payload["warnings"]
    assert isinstance(warnings, list)
    if warnings:
        lines.extend(f"- {warning}" for warning in warnings)
    else:
        lines.append("- none")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit VC SUDA final goal evidence without running experiments.")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-md", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = build_audit(args.repo_root)
    if args.output_json:
        _write_json(args.output_json, payload)
    if args.output_md:
        _write_markdown(args.output_md, payload)
    if not args.output_json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"status {payload['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
