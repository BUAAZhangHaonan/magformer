from __future__ import annotations

import importlib
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
VENDORED_DETECTRON2_ROOT = REPO_ROOT / "baselines" / "detectron2"


class _StorageWithoutTimeHistory:
    def __init__(self, iteration: int) -> None:
        self.iter = iteration

    def history(self, _name: str):
        raise KeyError("time")

    def put_scalar(self, *_args, **_kwargs) -> None:
        return None


def test_uoais_wrapper_patches_common_metric_printer_for_short_runs() -> None:
    sys.path.insert(0, str(VENDORED_DETECTRON2_ROOT))
    try:
        events = importlib.import_module("detectron2.utils.events")
        wrapper = importlib.import_module("baselines.run_uoais_ecc")
        wrapper._patch_short_run_common_metric_printer()

        printer = events.CommonMetricPrinter(max_iter=2)
        printer._last_write = (1, time.perf_counter())

        eta = printer._get_eta(_StorageWithoutTimeHistory(iteration=1))

        assert eta is None
        assert printer._last_write is not None
        assert printer._last_write[0] == 1
    finally:
        try:
            sys.path.remove(str(VENDORED_DETECTRON2_ROOT))
        except ValueError:
            pass
