from __future__ import annotations

import importlib.util
import os
import sys
import warnings
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
os.environ["PATH"] = f"{REPO_ROOT}:{os.environ.get('PATH', '')}"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

VENDORED_IMPORT_ROOTS = (
    REPO_ROOT / "baselines" / "detectron2",
    REPO_ROOT / "baselines" / "msmformer" / "MSMFormer",
    REPO_ROOT / "baselines" / "uoais",
    REPO_ROOT / "baselines" / "unseen_object_clustering",
)

for vendored_root in VENDORED_IMPORT_ROOTS:
    if vendored_root.exists():
        vendored_root_str = str(vendored_root)
        if vendored_root_str not in sys.path:
            sys.path.insert(0, vendored_root_str)


OPTIONAL_DEPENDENCY_TESTS = {
    "pycocotools": {
        "test_benchmark_inference_phases.py",
        "test_benchmark_inference_register_id.py",
        "test_benchmark_inference_yolo_predictor.py",
        "test_coco_eval_results_bbox_fallback.py",
        "test_coco_export_mask_fallback_policy.py",
        "test_coco_export_parity.py",
        "test_depth_backbone_builder.py",
    },
    "cv2": {
        "test_depth_sanity_helpers.py",
        "test_optimizer_groups.py",
        "test_reference_bank_loading.py",
        "test_scheduler_parity.py",
    },
    "fvcore": {
        "test_detectron2_ddp_cfg.py",
        "test_msmformer_optimizer_alignment.py",
    },
    "detectron2": {
        "test_detectron2_ddp_cfg.py",
        "test_detectron2_common_metric_printer_short_run.py",
        "test_ecc_datasets.py",
        "test_ecc_datasets_script_imports.py",
        "test_mgm_repo_pathing.py",
        "test_msmformer_dataset_mapper_normalization.py",
        "test_msmformer_imports.py",
        "test_msmformer_optimizer_alignment.py",
        "test_msmformer_recipe_alignment.py",
        "test_uoais_modal_coco_masks.py",
    },
    "torchvision_runtime": {
        "test_convert_mask2former_ckpt_to_magformer_class_embed.py",
        "test_detectron2_ddp_cfg.py",
        "test_detectron2_common_metric_printer_short_run.py",
        "test_ecc_datasets.py",
        "test_ecc_datasets_script_imports.py",
        "test_mgm_repo_pathing.py",
        "test_msmformer_dataset_mapper_normalization.py",
        "test_msmformer_imports.py",
        "test_msmformer_optimizer_alignment.py",
        "test_msmformer_recipe_alignment.py",
        "test_uoais_modal_coco_masks.py",
        "test_yolo_stats_norm.py",
    },
}

OPTIONAL_DEPENDENCY_NODEIDS = {
    "torchvision_runtime": {
        "tests/test_convert_mask2former_ckpt_to_magformer_class_embed.py::test_convert_mask2former_ckpt_maps_class_embed_to_1class",
        "tests/test_convert_mask2former_ckpt_to_magformer_class_embed.py::test_convert_mask2former_ckpt_maps_pixel_decoder_adapters_and_static_query",
        "tests/test_unet_instance_models.py::test_build_instance_model_supports_modern_smp_variant",
    },
}


def _has_working_torchvision_runtime() -> bool:
    try:
        import torchvision  # noqa: F401
    except Exception:
        return False
    return True


def _dependency_is_available(dependency: str) -> bool:
    if dependency == "torchvision_runtime":
        return _has_working_torchvision_runtime()
    return importlib.util.find_spec(dependency) is not None


def _missing_dependency_reason(path: Path) -> str | None:
    test_name = path.name
    missing = sorted(
        dependency
        for dependency, test_names in OPTIONAL_DEPENDENCY_TESTS.items()
        if test_name in test_names and not _dependency_is_available(dependency)
    )
    if not missing:
        return None
    dependencies = ", ".join(missing)
    return f"requires optional dependency/dependencies not installed in this environment: {dependencies}"


def pytest_ignore_collect(collection_path, config) -> bool:  # type: ignore[override]
    path = Path(str(collection_path))
    if path.suffix != ".py":
        return False
    reason = _missing_dependency_reason(path)
    if reason is None:
        return False
    warnings.warn(f"Skipping {path.name}: {reason}", UserWarning)
    return True


def pytest_collection_modifyitems(config, items) -> None:  # type: ignore[override]
    for dependency, nodeids in OPTIONAL_DEPENDENCY_NODEIDS.items():
        if _dependency_is_available(dependency):
            continue
        reason = f"requires optional dependency/dependencies not installed in this environment: {dependency}"
        marker = pytest.mark.skip(reason=reason)
        for item in items:
            if item.nodeid in nodeids:
                item.add_marker(marker)
