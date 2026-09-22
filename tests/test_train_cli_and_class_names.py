from __future__ import annotations

from types import SimpleNamespace

import pytest


def test_train_main_forwards_finetune_weights_cli_override(monkeypatch) -> None:
    from tools import train as train_tool

    captured = {}

    class _Stop(Exception):
        pass

    args = SimpleNamespace(
        config="configs/magformer_aligned_comparison.yaml",
        dataset_root=None,
        weights=None,
        finetune_weights="warm_start.pth",
        output_dir=None,
        resume=None,
        gpus=None,
        num_workers=None,
        seed=None,
    )

    def _fake_load_config(config_file, overrides=None):
        captured["config_file"] = config_file
        captured["overrides"] = overrides
        raise _Stop

    monkeypatch.setattr(train_tool, "parse_args", lambda: args)
    monkeypatch.setattr(train_tool, "load_config", _fake_load_config)

    with pytest.raises(_Stop):
        train_tool.main()

    assert captured["overrides"]["model"]["finetune_weights"] == "warm_start.pth"


def test_resolve_class_names_prefers_dataset_metadata() -> None:
    from magformer.utils.labels import resolve_class_names

    config = SimpleNamespace(data=SimpleNamespace(class_names=["from_config"]))
    dataset = SimpleNamespace(class_names=["from_dataset"])

    assert resolve_class_names(config=config, dataset=dataset) == ["from_dataset"]


def test_resolve_class_names_falls_back_to_config() -> None:
    from magformer.utils.labels import resolve_class_names

    config = SimpleNamespace(data=SimpleNamespace(class_names=["from_config"]))

    assert resolve_class_names(config=config, dataset=None) == ["from_config"]
