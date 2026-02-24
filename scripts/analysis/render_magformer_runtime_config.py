#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import yaml


def _load_yaml(path: Path) -> Dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _dump_yaml(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")


def _parse_steps(raw: str) -> List[int]:
    vals = [int(x.strip()) for x in raw.split(",") if x.strip()]
    if len(vals) != 2:
        raise ValueError(f"--steps expects exactly 2 integers, got: {raw}")
    return vals


def _parse_override(raw: str) -> Tuple[Sequence[str], Any]:
    if "=" not in raw:
        raise ValueError(f"--override expects KEY=VALUE, got: {raw}")
    key, value_str = raw.split("=", 1)
    key = key.strip()
    if not key:
        raise ValueError(f"--override expects non-empty KEY=VALUE, got: {raw}")
    try:
        value = yaml.safe_load(value_str)
    except Exception:
        # Fallback to raw string for edge cases (e.g. unquoted paths with ':')
        value = value_str
    return tuple(k for k in key.split(".") if k), value


def _set_by_dotted_path(cfg: Dict[str, Any], path: Sequence[str], value: Any) -> None:
    cur: Any = cfg
    for key in path[:-1]:
        if not isinstance(cur, dict):
            raise TypeError(f"Cannot set {'.'.join(path)} on non-dict at {key}: {type(cur)}")
        nxt = cur.get(key)
        if nxt is None:
            nxt = {}
            cur[key] = nxt
        elif not isinstance(nxt, dict):
            raise TypeError(f"Cannot set {'.'.join(path)}; {key} is {type(nxt)}, expected dict")
        cur = nxt
    last = path[-1]
    if not isinstance(cur, dict):
        raise TypeError(f"Cannot set {'.'.join(path)} on non-dict parent: {type(cur)}")
    cur[last] = value


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-config", type=str, required=True)
    ap.add_argument("--out-config", type=str, required=True)
    ap.add_argument("--output-dir", type=str, required=True)
    ap.add_argument("--run-name", type=str, required=True)
    ap.add_argument("--base-lr", type=float, required=True)
    ap.add_argument("--max-iter", type=int, required=True)
    ap.add_argument("--steps", type=str, required=True, help="Comma-separated milestones, e.g. 1776,1998")
    ap.add_argument("--warmup-iters", type=int, required=True)
    ap.add_argument("--ims-per-batch", type=int, required=True)
    ap.add_argument("--eval-period", type=int, required=True)
    ap.add_argument("--checkpoint-period", type=int, required=True)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument(
        "--override",
        action="append",
        default=[],
        help="Extra config overrides in dotted form KEY=VALUE (VALUE parsed as YAML).",
    )
    args = ap.parse_args()

    base_cfg = Path(args.base_config).resolve()
    out_cfg = Path(args.out_config).resolve()
    output_dir = str(Path(args.output_dir).resolve())
    run_name = str(args.run_name)

    cfg = _load_yaml(base_cfg)
    steps = _parse_steps(args.steps)

    cfg["name"] = run_name
    cfg["solver"]["base_lr"] = float(args.base_lr)
    cfg["solver"]["max_iter"] = int(args.max_iter)
    cfg["solver"]["steps"] = steps
    cfg["solver"]["warmup_iters"] = int(args.warmup_iters)
    cfg["solver"]["ims_per_batch"] = int(args.ims_per_batch)

    cfg["runtime"]["output_dir"] = output_dir
    cfg["runtime"]["num_workers"] = int(args.num_workers)
    cfg["runtime"]["eval_period"] = int(args.eval_period)
    cfg["runtime"]["checkpoint_period"] = int(args.checkpoint_period)
    cfg["runtime"]["logger"]["run_name"] = run_name
    cfg["runtime"]["logger"]["log_dir"] = str(Path(output_dir) / "logs")

    for raw in args.override:
        path, value = _parse_override(raw)
        if not path:
            raise ValueError(f"Invalid --override (empty key): {raw}")
        _set_by_dotted_path(cfg, path, value)

    _dump_yaml(out_cfg, cfg)
    print(f"[render-magformer-config] wrote: {out_cfg}")


if __name__ == "__main__":
    main()
