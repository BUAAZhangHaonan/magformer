from __future__ import annotations

from typing import Any, List, Optional


def resolve_class_names(
    *,
    config: Optional[Any] = None,
    dataset: Optional[Any] = None,
    default: Optional[List[str]] = None,
) -> List[str]:
    if dataset is not None:
        dataset_names = getattr(dataset, "class_names", None)
        if dataset_names:
            return [str(name) for name in dataset_names]

    if config is not None:
        if isinstance(config, dict):
            data_cfg = config.get("data", {})
            config_names = data_cfg.get("class_names") if isinstance(data_cfg, dict) else None
        else:
            data_cfg = getattr(config, "data", None)
            config_names = getattr(data_cfg, "class_names", None)
        if config_names:
            return [str(name) for name in config_names]

    return list(default or ["component"])
