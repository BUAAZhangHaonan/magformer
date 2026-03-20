#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shlex
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]
ROSTER_PATH = REPO_ROOT / "configs" / "experiments" / "full_20260318_1k_1566_roster.json"


def load_roster() -> dict[str, Any]:
    return json.loads(ROSTER_PATH.read_text(encoding="utf-8"))


def iter_entries() -> Iterable[dict[str, Any]]:
    yield from load_roster().get("entries", [])


def render_command(
    entry: dict[str, Any],
    *,
    script_dir: Path,
    register: str,
    dataset_root: str,
    output_root: str,
    mode: str,
) -> str:
    args = [
        "bash",
        str((script_dir / entry["runner"]).resolve()),
        "--register",
        str(register),
        "--dataset-root",
        str(dataset_root),
        "--output-root",
        str(output_root),
    ]
    args.extend(str(x) for x in entry.get("args", []))
    args.append(f"--{mode}")
    return shlex.join(args)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect or render the full 19-model 20260318 roster.")
    parser.add_argument("--format", choices=["commands", "manifest"], default="manifest")
    parser.add_argument("--register", default="")
    parser.add_argument("--dataset-root", default="")
    parser.add_argument("--output-root", default="")
    parser.add_argument("--mode", choices=["run", "dry-run"], default="dry-run")
    args = parser.parse_args()

    if args.format == "manifest":
        payload = {
            "models": [
                {"id": str(entry["model_id"]), "framework": str(entry["runner"])}
                for entry in iter_entries()
            ]
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    if not args.register or not args.dataset_root or not args.output_root:
        raise SystemExit("--register, --dataset-root and --output-root are required when --format=commands")

    script_dir = REPO_ROOT / "scripts" / "experiments"
    for entry in iter_entries():
        command = render_command(
            entry,
            script_dir=script_dir,
            register=args.register,
            dataset_root=args.dataset_root,
            output_root=args.output_root,
            mode=args.mode,
        )
        print(f"{entry['model_id']}\t{entry['source_output_name']}\t{command}")


if __name__ == "__main__":
    main()

