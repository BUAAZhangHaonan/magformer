from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from scripts.analysis import benchmark_inference as benchmark_mod


@pytest.mark.parametrize(
    ("model_id", "expected_family"),
    [
        ("cellpose", "cellpose"),
        ("stardist", "stardist"),
        ("iaunet", "iaunet"),
        ("unet_semantic_inst", "unet"),
    ],
)
def test_detect_family_recognizes_new_external_baselines(tmp_path: Path, model_id: str, expected_family: str) -> None:
    assert benchmark_mod._detect_family(tmp_path, model_id) == expected_family


def test_infer_image_size_uses_metadata_command_then_path_hint(tmp_path: Path) -> None:
    out_dir = tmp_path / "20260406_1k_1566_20ep_512_full19" / "cellpose"
    out_dir.mkdir(parents=True)

    assert benchmark_mod._infer_image_size({"command": "python run.py --image-size 512"}, out_dir) == 512
    assert benchmark_mod._infer_image_size({}, out_dir) == 512
    assert benchmark_mod._infer_image_size({"image_size": "1024"}, out_dir) == 1024


def test_benchmark_output_dir_routes_new_families(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def _fake_result(name: str):
        def _fn(*args, **kwargs):
            calls.append(name)
            return {"framework": name, "image_size": 512}

        return _fn

    monkeypatch.setattr(benchmark_mod, "_benchmark_cellpose", _fake_result("cellpose"))
    monkeypatch.setattr(benchmark_mod, "_benchmark_stardist", _fake_result("stardist"))
    monkeypatch.setattr(benchmark_mod, "_benchmark_iaunet", _fake_result("iaunet"))

    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    device = torch.device("cpu")

    for model_id, expected_family in [("cellpose", "cellpose"), ("stardist", "stardist"), ("iaunet", "iaunet")]:
        out_dir = tmp_path / model_id
        out_dir.mkdir()
        (out_dir / "metadata.json").write_text(
            json.dumps({"model_id": model_id, "command": "python run.py --image-size 512"}),
            encoding="utf-8",
        )

        result = benchmark_mod.benchmark_output_dir(
            out_dir=out_dir,
            dataset_root=dataset_root,
            device=device,
            warmup=0,
            timed_images=1,
        )
        assert result["family"] == expected_family
        assert result["model_id"] == model_id
        assert result["image_size"] == 512

    assert calls == ["cellpose", "stardist", "iaunet"]
