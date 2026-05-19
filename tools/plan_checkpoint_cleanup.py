#!/usr/bin/env python3
"""Plan checkpoint cleanup without deleting files."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Iterable

DEFAULT_MANIFEST = Path("docs/results/checkpoint_cleanup_manifest_20260519.md")
DEFAULT_REFERENCE_ROOTS = ("docs/results", "configs", "scripts", "tools")
CHECKPOINT_ITER_RE = re.compile(r"^checkpoint_iter_\d{7}\.pth$")
MODEL_ITER_RE = re.compile(r"^model_\d+\.pth$")
TRAINER_STATE_ITER_RE = re.compile(r"^trainer_state_\d+\.pth$")
REFERENCE_PATH_RE = re.compile(r"(?P<path>(?:\./)?output/[^\s\"'`()<>]+?\.pth)")
GENERATED_CLEANUP_DOC_RE = re.compile(r"^checkpoint_cleanup_(?:manifest|plan)_\d{8}\.(?:md|json)$")

EXACT_KEEP_PATHS = {
    "output/vc_suda/stage_c_r12_32k_source_r8b_ckpt999_continue_1024_teacher8499/checkpoint_iter_0000499.pth",
}

RUN_KEEP_RULES = (
    ("final/gate:r139-r141-iter2000", ("r139", "r141"), {"checkpoint_iter_0002000.pth"}),
    (
        "final/gate:r114-iters0999-1499-2000",
        ("r114",),
        {
            "checkpoint_iter_0000999.pth",
            "checkpoint_iter_0001499.pth",
            "checkpoint_iter_0002000.pth",
        },
    ),
    (
        "final/gate:r115b-iters0999-2000",
        ("r115b",),
        {"checkpoint_iter_0000999.pth", "checkpoint_iter_0002000.pth"},
    ),
    (
        "final/gate:r111-r112-iters0499-0799-1000",
        ("r111", "r112"),
        {
            "checkpoint_iter_0000499.pth",
            "checkpoint_iter_0000799.pth",
            "checkpoint_iter_0001000.pth",
        },
    ),
    (
        "final/gate:r113-iters1499-2000",
        ("r113",),
        {"checkpoint_iter_0001499.pth", "checkpoint_iter_0002000.pth"},
    ),
    (
        "warm-start:r12-r98-iter0499",
        ("r12", "r98"),
        {"checkpoint_iter_0000499.pth"},
    ),
)


def _rel(path: Path, repo_root: Path) -> str:
    return path.relative_to(repo_root).as_posix()


def _human_size(num_bytes: int) -> str:
    value = float(num_bytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024.0 or unit == "TiB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024.0
    return f"{num_bytes} B"


def _is_checkpoint_file(path: Path) -> bool:
    name = path.name
    return bool(
        CHECKPOINT_ITER_RE.match(name)
        or MODEL_ITER_RE.match(name)
        or TRAINER_STATE_ITER_RE.match(name)
    )


def _run_id_in_path(rel_path: str, run_id: str) -> bool:
    pattern = rf"(?<![a-z0-9]){re.escape(run_id)}(?![a-z0-9])"
    return re.search(pattern, rel_path.lower()) is not None


def _iter_checkpoints(repo_root: Path, output_root: Path) -> list[Path]:
    root = output_root if output_root.is_absolute() else repo_root / output_root
    if not root.exists():
        return []
    return sorted(path for path in root.rglob("*.pth") if path.is_file() and _is_checkpoint_file(path))


def _iter_reference_files(repo_root: Path, reference_roots: Iterable[str]) -> list[Path]:
    files: list[Path] = []
    for rel_root in reference_roots:
        root = repo_root / rel_root
        if not root.exists():
            continue
        if root.is_file():
            if not GENERATED_CLEANUP_DOC_RE.match(root.name):
                files.append(root)
            continue
        files.extend(
            sorted(
                path
                for path in root.rglob("*")
                if path.is_file() and not GENERATED_CLEANUP_DOC_RE.match(path.name)
            )
        )
    return files


def _read_text_lossy(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return ""


def _referenced_checkpoint_paths(
    repo_root: Path, checkpoints: list[Path], reference_roots: Iterable[str]
) -> dict[str, set[str]]:
    by_rel = {_rel(path, repo_root): path for path in checkpoints}
    by_resolved: dict[Path, str] = {}
    for rel_path, path in by_rel.items():
        try:
            by_resolved[path.resolve()] = rel_path
        except OSError:
            continue

    references: dict[str, set[str]] = {rel_path: set() for rel_path in by_rel}
    pending = set(by_rel)
    for text_file in _iter_reference_files(repo_root, reference_roots):
        text = _read_text_lossy(text_file)
        if not text:
            continue
        source = _rel(text_file, repo_root)

        for match in REFERENCE_PATH_RE.finditer(text):
            token = match.group("path").rstrip(".,;:")
            token = token[2:] if token.startswith("./") else token
            direct = by_rel.get(token)
            if direct is not None:
                references[token].add(source)
                pending.discard(token)
                continue
            ref_path = repo_root / token
            try:
                resolved_rel = by_resolved.get(ref_path.resolve())
            except OSError:
                resolved_rel = None
            if resolved_rel is not None:
                references[resolved_rel].add(source)
                pending.discard(resolved_rel)

        for rel_path in list(pending):
            if rel_path in text or f"./{rel_path}" in text or str(repo_root / rel_path) in text:
                references[rel_path].add(source)
                pending.discard(rel_path)

    return {path: sources for path, sources in references.items() if sources}


def _keep_reasons(rel_path: str, path: Path, references: dict[str, set[str]]) -> list[str]:
    reasons: list[str] = []
    if rel_path in references:
        sources = ", ".join(sorted(references[rel_path]))
        reasons.append(f"referenced by {sources}")
    if rel_path in EXACT_KEEP_PATHS:
        reasons.append("explicit keep path")
    for rule_name, run_ids, names in RUN_KEEP_RULES:
        if path.name not in names:
            continue
        if any(_run_id_in_path(rel_path, run_id) for run_id in run_ids):
            reasons.append(rule_name)
    return reasons


def _checkpoint_record(path: Path, repo_root: Path, reason: str) -> dict[str, object]:
    size_bytes = path.stat().st_size
    return {
        "path": _rel(path, repo_root),
        "size_bytes": size_bytes,
        "size_human": _human_size(size_bytes),
        "reason": reason,
    }


def build_cleanup_plan(
    repo_root: Path,
    *,
    output_root: Path | str = Path("output"),
    reference_roots: Iterable[str] = DEFAULT_REFERENCE_ROOTS,
) -> dict[str, object]:
    repo_root = repo_root.resolve()
    output_root = Path(output_root)
    checkpoints = _iter_checkpoints(repo_root, output_root)
    references = _referenced_checkpoint_paths(repo_root, checkpoints, reference_roots)

    kept: list[dict[str, object]] = []
    candidates: list[dict[str, object]] = []
    for path in checkpoints:
        rel_path = _rel(path, repo_root)
        reasons = _keep_reasons(rel_path, path, references)
        if reasons:
            kept.append(_checkpoint_record(path, repo_root, "; ".join(reasons)))
        else:
            candidates.append(_checkpoint_record(path, repo_root, "unreferenced checkpoint"))

    candidate_total = sum(int(item["size_bytes"]) for item in candidates)
    first_batch = [
        item
        for item in candidates
        if _run_id_in_path(str(item["path"]), "r139") or _run_id_in_path(str(item["path"]), "r141")
    ]
    first_batch_total = sum(int(item["size_bytes"]) for item in first_batch)

    return {
        "repo_root": str(repo_root),
        "output_root": output_root.as_posix(),
        "reference_roots": list(reference_roots),
        "checkpoint_count": len(checkpoints),
        "kept_count": len(kept),
        "delete_candidate_count": len(candidates),
        "estimated_reclaim_bytes": candidate_total,
        "estimated_reclaim_human": _human_size(candidate_total),
        "first_batch_r139_r141_count": len(first_batch),
        "first_batch_r139_r141_bytes": first_batch_total,
        "first_batch_r139_r141_human": _human_size(first_batch_total),
        "kept_checkpoints": kept,
        "delete_candidates": candidates,
    }


def _render_markdown(plan: dict[str, object]) -> str:
    lines = [
        "# Checkpoint Cleanup Manifest - 2026-05-19",
        "",
        "This is a dry-run manifest. It does not delete files.",
        "",
        "## Summary",
        "",
        f"- Checkpoints scanned: {plan['checkpoint_count']}",
        f"- Kept checkpoints: {plan['kept_count']}",
        f"- Delete candidates: {plan['delete_candidate_count']}",
        f"- Estimated reclaim: {plan['estimated_reclaim_human']} ({plan['estimated_reclaim_bytes']} bytes)",
        "- First batch recommendation: only R139/R141 intermediate checkpoints after manual review.",
        f"- First batch R139/R141 reclaim: {plan['first_batch_r139_r141_human']} ({plan['first_batch_r139_r141_bytes']} bytes)",
        "",
        "## Safety Rules",
        "",
        "- The planner never deletes files.",
        "- It only lists checkpoint files, not run directories.",
        "- Logs, metrics, configs, and JSON files are never listed as delete candidates.",
        "- Checkpoints referenced from docs/results, configs, scripts, or tools are kept.",
        "- Final/gate and warm-start checkpoints from the manual keep rules are kept.",
        "",
        "## Delete Candidates",
        "",
        "| Path | Size | Reason |",
        "| --- | ---: | --- |",
    ]
    for item in plan["delete_candidates"]:
        lines.append(f"| `{item['path']}` | {item['size_human']} | {item['reason']} |")

    lines.extend(["", "## Kept Checkpoints", "", "| Path | Size | Reason |", "| --- | ---: | --- |"])
    for item in plan["kept_checkpoints"]:
        lines.append(f"| `{item['path']}` | {item['size_human']} | {item['reason']} |")
    lines.append("")
    return "\n".join(lines)


def write_manifest(plan: dict[str, object], manifest: Path) -> None:
    manifest.parent.mkdir(parents=True, exist_ok=True)
    if manifest.suffix.lower() == ".json":
        manifest.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    else:
        manifest.write_text(_render_markdown(plan), encoding="utf-8")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--output-root", type=Path, default=Path("output"))
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    repo_root = args.repo_root.resolve()
    manifest = args.manifest if args.manifest.is_absolute() else repo_root / args.manifest
    plan = build_cleanup_plan(repo_root, output_root=args.output_root)
    write_manifest(plan, manifest)
    print(f"manifest={manifest}")
    print(f"delete_candidates={plan['delete_candidate_count']}")
    print(f"estimated_reclaim={plan['estimated_reclaim_human']}")
    print(f"first_batch_r139_r141_reclaim={plan['first_batch_r139_r141_human']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
