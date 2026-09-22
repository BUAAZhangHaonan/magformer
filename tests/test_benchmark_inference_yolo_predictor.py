from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from scripts.analysis.benchmark_inference import _make_yolo_infer_fn


class _FakePredictor:
    def __init__(self, overrides=None, _callbacks=None):
        self.overrides = dict(overrides or {})
        self.callbacks = _callbacks
        self.args = SimpleNamespace(device=self.overrides.get("device"))
        self.setup_calls = []

    def setup_model(self, model=None, verbose=False):
        self.setup_calls.append((model, verbose))


class _FakeYOLO:
    def __init__(self):
        self.overrides = {"existing": "keep"}
        self.callbacks = object()
        self.model = object()
        self.predictor = None
        self.calls = []

    def predict(self, **kwargs):
        assert "rgb_mean" not in kwargs
        assert "rgb_std" not in kwargs
        assert "predictor" not in kwargs
        self.calls.append(kwargs)
        return {"ok": True}


def test_make_yolo_infer_fn_configures_predictor_once_and_omits_rgb_kwargs() -> None:
    model = _FakeYOLO()
    infer_fn = _make_yolo_infer_fn(
        model=model,
        predictor_cls=_FakePredictor,
        yolo_device=0,
        imgsz=1024,
        rgb_mean=[1.0, 2.0, 3.0],
        rgb_std=[4.0, 5.0, 6.0],
    )

    infer_fn(np.zeros((8, 8, 3), dtype=np.uint8))
    infer_fn(np.zeros((8, 8, 3), dtype=np.uint8))

    assert isinstance(model.predictor, _FakePredictor)
    assert model.predictor.overrides["rgb_mean"] == [1.0, 2.0, 3.0]
    assert model.predictor.overrides["rgb_std"] == [4.0, 5.0, 6.0]
    assert len(model.predictor.setup_calls) == 1
    assert len(model.calls) == 2
