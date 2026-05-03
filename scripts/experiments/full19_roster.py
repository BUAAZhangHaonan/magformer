#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shlex
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]
ROSTER_PATH = REPO_ROOT / "configs" / "experiments" / "full_20260318_1k_1566_roster.json"
RUNNERS_SUPPORT_IMAGE_SIZE = {
    "run_0831_1k_20ep_1024_revisit_magformer.sh",
    "run_0831_1k_20ep_1024_lightdepth_stage_a_magformer.sh",
    "run_0831_1k_20ep_1024_revisit_mgm_mask2former.sh",
    "run_0831_1k_20ep_scratch_official_mask2former.sh",
    "run_0831_1k_20ep_scratch_maskrcnn.sh",
    "run_0831_1k_20ep_scratch_yolov8_seg.sh",
    "run_0831_1k_20ep_scratch_msmformer.sh",
    "run_0831_1k_20ep_scratch_ucn.sh",
    "run_0831_1k_20ep_scratch_uoais.sh",
    "run_0831_1k_20ep_1024_revisit_iaunet_inst.sh",
    "run_0831_1k_20ep_1024_revisit_cellpose_inst.sh",
    "run_0831_1k_20ep_1024_revisit_stardist_inst.sh",
    "run_0831_1k_20ep_1024_revisit_unet_semantic_inst.sh",
    "run_0831_1k_20ep_1024_revisit_unet_boundary_inst.sh",
    "run_0831_1k_20ep_1024_revisit_unetpp_boundary_inst.sh",
}


def load_roster() -> dict[str, Any]:
    return json.loads(ROSTER_PATH.read_text(encoding="utf-8"))


def iter_entries() -> Iterable[dict[str, Any]]:
    yield from load_roster().get("entries", [])


def _string_args(entry: dict[str, Any]) -> list[str]:
    return [str(x) for x in entry.get("args", [])]


def _apply_image_size(entry: dict[str, Any], args: list[str], image_size: int | None) -> list[str]:
    if image_size is None:
        return args
    size = str(int(image_size))
    out = list(args)
    if "--image-size" in out:
        idx = out.index("--image-size")
        if idx + 1 < len(out):
            out[idx + 1] = size
        else:
            out.append(size)
        return out
    if str(entry.get("runner", "")) in RUNNERS_SUPPORT_IMAGE_SIZE:
        out.extend(["--image-size", size])
    return out


def _apply_single_gpu(args: list[str], single_gpu: bool) -> list[str]:
    if not single_gpu:
        return list(args)

    out: list[str] = []
    i = 0
    while i < len(args):
        token = args[i]
        if token == "--ddp":
            i += 1
            continue
        if token == "--num-gpus":
            out.extend(["--num-gpus", "1"])
            i += 2
            continue
        if token == "--device":
            out.extend(["--device", "0"])
            i += 2
            continue
        out.append(token)
        i += 1
    return out


def _entries_for_commands() -> list[dict[str, Any]]:
    return [deepcopy(entry) for entry in iter_entries()]


def render_command(
    entry: dict[str, Any],
    *,
    script_dir: Path,
    register: str,
    dataset_root: str,
    output_root: str,
    mode: str,
    image_size: int | None,
    single_gpu: bool,
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
    runner_args = _apply_image_size(entry, _string_args(entry), image_size)
    runner_args = _apply_single_gpu(runner_args, single_gpu)
    args.extend(runner_args)
    args.append(f"--{mode}")
    return shlex.join(args)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect or render the 20260318 full roster.")
    parser.add_argument("--format", choices=["commands", "manifest"], default="manifest")
    parser.add_argument("--register", default="")
    parser.add_argument("--dataset-root", default="")
    parser.add_argument("--output-root", default="")
    parser.add_argument("--mode", choices=["run", "dry-run"], default="dry-run")
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--single-gpu", action="store_true")
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
    for entry in _entries_for_commands():
        command = render_command(
            entry,
            script_dir=script_dir,
            register=args.register,
            dataset_root=args.dataset_root,
            output_root=args.output_root,
            mode=args.mode,
            image_size=args.image_size,
            single_gpu=bool(args.single_gpu),
        )
        print(f"{entry['model_id']}\t{entry['source_output_name']}\t{command}")


if __name__ == "__main__":
    main()
