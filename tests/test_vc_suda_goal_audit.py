from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from tools.audit_vc_suda_goal import build_audit


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


class FakeGit:
    def __init__(self, outputs: dict[tuple[str, ...], subprocess.CompletedProcess[str]] | None = None) -> None:
        self.outputs = outputs or {}

    def __call__(self, args: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
        return self.outputs.get(tuple(args), subprocess.CompletedProcess(args, 0, "", ""))


def _write_json(repo_root: Path, relpath: Path, payload: dict[str, object]) -> None:
    path = repo_root / relpath
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_metrics(repo_root: Path, relpath: Path, segm_ap: float) -> None:
    _write_json(repo_root, relpath, {"segm_AP": segm_ap, "bbox_AP": segm_ap})


def _write_complete_inputs(repo_root: Path, *, first50_ap: float = 0.62, r122_ap: float = 0.62) -> None:
    _write_metrics(repo_root, FIRST50_METRICS, first50_ap)
    _write_json(repo_root, R123_STATS, {"audit": "r123_pseudo_real_validity_20260518"})
    for relpath in R123_CONTACT_SHEETS:
        path = repo_root / relpath
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"png")
    _write_json(repo_root, R122_GO_NO_GO, {"decision": "PASS"})
    _write_metrics(repo_root, R122_REMAINING75_METRICS, r122_ap)
    _write_metrics(repo_root, R122_VAL28_METRICS, 0.50)


def _item(payload: dict[str, object], item_id: str) -> dict[str, object]:
    return next(item for item in payload["checklist"] if item["id"] == item_id)


def test_missing_r122_go_no_go_and_metrics_are_not_complete(tmp_path: Path) -> None:
    _write_metrics(tmp_path, FIRST50_METRICS, 0.62)
    _write_json(tmp_path, R123_STATS, {"audit": "r123_pseudo_real_validity_20260518"})
    for relpath in R123_CONTACT_SHEETS:
        path = tmp_path / relpath
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"png")

    payload = build_audit(tmp_path, git_runner=FakeGit(), process_lister=lambda: [])

    assert payload["status"] == "NOT_COMPLETE"
    assert str(R122_GO_NO_GO) in payload["missing_inputs"]
    assert str(R122_REMAINING75_METRICS) in payload["missing_inputs"]
    assert str(R122_VAL28_METRICS) in payload["missing_inputs"]


def test_first50_protocol_fix_passes_only_when_segm_ap_clears_threshold(tmp_path: Path) -> None:
    _write_complete_inputs(tmp_path, first50_ap=0.61)
    passing = build_audit(tmp_path, git_runner=FakeGit(), process_lister=lambda: [])
    assert _item(passing, "first50_protocol_fix")["status"] == "PASS"

    _write_metrics(tmp_path, FIRST50_METRICS, 0.609)
    failing = build_audit(tmp_path, git_runner=FakeGit(), process_lister=lambda: [])
    assert _item(failing, "first50_protocol_fix")["status"] == "FAIL"


def test_r123_validity_requires_stats_and_all_contact_sheets(tmp_path: Path) -> None:
    _write_complete_inputs(tmp_path)
    passing = build_audit(tmp_path, git_runner=FakeGit(), process_lister=lambda: [])
    assert _item(passing, "r123_validity_artifacts")["status"] == "PASS"

    (tmp_path / R123_CONTACT_SHEETS[-1]).unlink()
    failing = build_audit(tmp_path, git_runner=FakeGit(), process_lister=lambda: [])
    assert _item(failing, "r123_validity_artifacts")["status"] == "FAIL"
    assert str(R123_CONTACT_SHEETS[-1]) in failing["missing_inputs"]


def test_final_segm_target_requires_go_no_go_pass_and_r122_remaining75_threshold(tmp_path: Path) -> None:
    _write_complete_inputs(tmp_path, r122_ap=0.61)
    passing = build_audit(tmp_path, git_runner=FakeGit(), process_lister=lambda: [])
    assert _item(passing, "final_r122_segm_target")["status"] == "PASS"
    assert passing["status"] == "COMPLETE"

    _write_json(tmp_path, R122_GO_NO_GO, {"decision": "FAIL"})
    failed_decision = build_audit(tmp_path, git_runner=FakeGit(), process_lister=lambda: [])
    assert _item(failed_decision, "final_r122_segm_target")["status"] == "FAIL"
    assert failed_decision["status"] == "NOT_COMPLETE"

    _write_json(tmp_path, R122_GO_NO_GO, {"decision": "PASS"})
    _write_metrics(tmp_path, R122_REMAINING75_METRICS, 0.609)
    failed_metric = build_audit(tmp_path, git_runner=FakeGit(), process_lister=lambda: [])
    assert _item(failed_metric, "final_r122_segm_target")["status"] == "FAIL"
    assert failed_metric["status"] == "NOT_COMPLETE"


def test_git_and_repo_process_state_are_not_silent(tmp_path: Path) -> None:
    _write_complete_inputs(tmp_path)
    git_runner = FakeGit(
        {
            ("status", "--porcelain"): subprocess.CompletedProcess([], 0, " M tools/train.py\n", ""),
            ("rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"): subprocess.CompletedProcess(
                [], 0, "origin/feature/vc-suda-sim2real\n", ""
            ),
            ("rev-list", "--count", "@{u}..HEAD"): subprocess.CompletedProcess([], 0, "2\n", ""),
        }
    )
    processes = [
        {"pid": "1234", "command": f"python tools/train.py --output-dir {tmp_path}/output/vc_suda/r122"}
    ]

    payload = build_audit(tmp_path, git_runner=git_runner, process_lister=lambda: processes)

    assert _item(payload, "git_dirty")["status"] == "FAIL"
    assert _item(payload, "git_unpushed_commits")["status"] == "WARNING"
    assert _item(payload, "repo_train_eval_processes")["status"] == "FAIL"
    assert payload["status"] == "NOT_COMPLETE"


def test_repo_process_audit_ignores_wrappers_and_counts_real_train_eval_only(tmp_path: Path) -> None:
    _write_complete_inputs(tmp_path)
    processes = [
        {
            "pid": "1111",
            "command": (
                "bash -lc 'cd /home/hdd3/zhanghaonan/magformer && "
                "python tools/audit_vc_suda_goal.py --note tools/evaluate_1024_backmap.py'"
            ),
        },
        {
            "pid": "2222",
            "command": "/home/hdd3/zhanghaonan/anaconda3/envs/magformer/bin/python tools/evaluate_1024_backmap.py --config r122",
        },
        {
            "pid": "3333",
            "command": "torchrun --nproc_per_node 1 tools/train.py --config configs/r122.py --work-dir /home/hdd3/zhanghaonan/magformer/output/r122",
        },
        {
            "pid": "4444",
            "command": "bash tools/run_r128_gated_r122_evaluator.sh",
        },
        {
            "pid": "5555",
            "command": "grep tools/evaluate_1024_backmap.py /home/hdd3/zhanghaonan/magformer/logs/audit.log",
        },
        {
            "pid": "6666",
            "command": (
                "/home/hdd3/zhanghaonan/anaconda3/envs/magformer/bin/python -m pytest "
                "/home/hdd3/zhanghaonan/magformer/tests/test_vc_suda_goal_audit.py -k tools/train.py"
            ),
        },
        {
            "pid": "7777",
            "command": (
                "/home/hdd3/zhanghaonan/anaconda3/envs/magformer/bin/python "
                "tools/check_r121_r122_resume_state.py --note tools/train.py"
            ),
        },
    ]

    payload = build_audit(tmp_path, git_runner=FakeGit(), process_lister=lambda: processes)

    process_item = _item(payload, "repo_train_eval_processes")
    assert process_item["status"] == "FAIL"
    assert [proc["pid"] for proc in process_item["evidence"]["processes"]] == ["2222", "3333"]


def test_cli_writes_json_and_markdown(tmp_path: Path) -> None:
    _write_complete_inputs(tmp_path)
    repo_root = Path(__file__).resolve().parents[1]
    output_json = tmp_path / "audit.json"
    output_md = tmp_path / "audit.md"

    result = subprocess.run(
        [
            sys.executable,
            str(repo_root / "tools" / "audit_vc_suda_goal.py"),
            "--repo-root",
            str(tmp_path),
            "--output-json",
            str(output_json),
            "--output-md",
            str(output_md),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(output_json.read_text(encoding="utf-8"))["status"] == "COMPLETE"
    assert "# VC SUDA Final Goal Audit" in output_md.read_text(encoding="utf-8")
