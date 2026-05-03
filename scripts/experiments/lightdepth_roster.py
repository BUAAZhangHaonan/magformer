#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]
ROSTER_PATH = REPO_ROOT / "configs" / "experiments" / "lightdepth_roster.json"


def load_roster() -> dict:
    return json.loads(ROSTER_PATH.read_text(encoding="utf-8"))


def iter_entries(status: str | None = None) -> Iterable[dict]:
    payload = load_roster()
    if status is None:
        for key in ("active", "archived"):
            yield from payload.get(key, [])
        return
    if status not in {"active", "archived"}:
        raise ValueError(f"Unsupported status: {status}")
    yield from payload.get(status, [])


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect the lightdepth active/archive roster.")
    parser.add_argument("--status", choices=["active", "archived"], default=None)
    parser.add_argument("--field", default="variant", choices=["variant", "model_id", "runner", "family", "reason", "status"])
    args = parser.parse_args()

    for entry in iter_entries(args.status):
        print(entry[args.field])


if __name__ == "__main__":
    main()
