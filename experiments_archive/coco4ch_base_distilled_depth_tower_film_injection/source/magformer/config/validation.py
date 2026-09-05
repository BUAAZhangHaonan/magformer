# -*- coding: utf-8 -*-
"""
MAGFormer Configuration Validation

Validates configuration for common issues that could cause training failure.
"""

import copy
import warnings
from typing import Any, List, Tuple, Mapping, Dict


# Legacy key registry (all emit DeprecationWarning):
#   dpe_enabled                 -> model.magformer.dpe.enabled
#   dpe_beta                    -> model.magformer.dpe.beta


def normalize_legacy_config_dict(config_dict: Any) -> Any:
    """Normalize legacy flat config keys to their canonical nested paths."""
    if not isinstance(config_dict, Mapping):
        return config_dict

    normalized: Dict[str, Any] = copy.deepcopy(dict(config_dict))
    legacy_key_map = {
        "dpe_enabled": ["model", "magformer", "dpe", "enabled"],
        "dpe_beta": ["model", "magformer", "dpe", "beta"],
    }

    for legacy_key, nested_path in legacy_key_map.items():
        if legacy_key not in normalized:
            continue

        legacy_value = normalized[legacy_key]
        warnings.warn(
            f"Legacy config key '{legacy_key}' is deprecated; use '{'.'.join(nested_path)}' instead.",
            DeprecationWarning,
            stacklevel=3,
        )

        current = normalized
        for key in nested_path[:-1]:
            value = current.get(key)
            if not isinstance(value, dict):
                value = {}
                current[key] = value
            current = value
        if nested_path[-1] not in current or current[nested_path[-1]] is None:
            current[nested_path[-1]] = legacy_value

    return normalized


class ConfigValidator:
    """
    Configuration validator for MAGFormer.

    Detects common configuration issues that could cause training failure,
    especially the critical depth normalization issue.
    """

    CRITICAL_CHECKS = [
        "depth_normalization",
        "dpe_configuration",
    ]

    def validate(self, config: Any) -> Tuple[bool, List[str]]:
        """
        Run all validation checks.

        Args:
            config: MAGFormer configuration object

        Returns:
            (is_valid, issues): Tuple of validation result and list of issue descriptions
        """
        issues = []

        # Run all checks
        issues.extend(self._check_depth_normalization(config))
        issues.extend(self._check_dpe_configuration(config))
        issues.extend(self._check_loss_weights(config))
        issues.extend(self._check_training_params(config))
        issues.extend(self._check_single_class_contract(config))
        issues.extend(self._check_normalization(config))
        issues.extend(self._check_swin_config(config))
        issues.extend(self._check_fusion_config(config))
        issues.extend(self._check_solver_recipe(config))

        # Separate critical issues
        critical_issues = [i for i in issues if i.startswith("CRITICAL")]
        is_valid = len(critical_issues) == 0

        return is_valid, issues

    def _check_depth_normalization(self, config: Any) -> List[str]:
        """
        Check for depth normalization issues.

        CRITICAL: When clip_min=0.0, clip_max=1.0, and per_sample_norm=False,
        depth values will NOT be normalized. This is the root cause of AP<1
        when depth values are already in a narrow range like [0.93, 0.96].
        """
        issues = []

        try:
            depth_cfg = config.data.depth
            clip_min = getattr(depth_cfg, "clip_min", 0.0)
            clip_max = getattr(depth_cfg, "clip_max", 1.0)
            per_sample_norm = getattr(depth_cfg, "per_sample_norm", True)

            rgb_concat_depth = bool(getattr(config.model.magformer, "rgb_concat_depth", False))
            if rgb_concat_depth:
                # 4ch concat mode: depth keeps metric scale and is normalized by the
                # 4-channel pixel_mean/std, matching the detectron2 rgbd_concat baseline.
                pass
            elif clip_min == 0.0 and clip_max == 1.0 and not per_sample_norm:
                issues.append(
                    "CRITICAL: depth.clip_min=0.0, clip_max=1.0 with per_sample_norm=False. "
                    "Depth values will NOT be normalized to [0,1]! "
                    "If your depth data is in a narrow range (e.g., [0.93, 0.96]), "
                    "set per_sample_norm=true to spread values to full [0,1] range."
                )
            elif clip_min == 0.0 and clip_max == 1.0 and per_sample_norm:
                issues.append(
                    "INFO: depth.per_sample_norm=true - depth values will be normalized "
                    "per-sample to [0,1] range. This is the recommended setting for "
                    "pre-normalized depth data."
                )
        except AttributeError as e:
            issues.append(f"WARNING: Could not access depth config: {e}")

        return issues

    def _check_dpe_configuration(self, config: Any) -> List[str]:
        """
        Check DPE (Depth Position Encoding) configuration.
        """
        issues = []

        try:
            dpe_cfg_nested = getattr(
                getattr(config.model, "magformer", object()), "dpe", None)
            dpe_enabled_nested = bool(
                getattr(dpe_cfg_nested, "enabled", False))

            # Check if pixel decoder has DPE enabled. Legacy flat keys are
            # normalized before this point, so the canonical nested path is the
            # only source of truth here.
            try:
                sem_seg_head = config.model.magformer.sem_seg_head
                pixel_decoder = getattr(sem_seg_head, "pixel_decoder_name", "")
                transformer_enc_layers = getattr(
                    sem_seg_head, "transformer_enc_layers", 0)

                if pixel_decoder == "MSDeformAttnPixelDecoder" and transformer_enc_layers > 0:
                    if not dpe_enabled_nested:
                        issues.append(
                            "INFO: MSDeformAttnPixelDecoder with transformer_enc_layers>0 "
                            "but model.magformer.dpe.enabled=False. Consider enabling DPE for better "
                            "depth-aware position encoding."
                        )
                    else:
                        issues.append(
                            "INFO: DPE (Depth Position Encoding) is enabled - "
                            "depth information will modulate position encoding."
                        )
            except AttributeError:
                pass

        except AttributeError as e:
            issues.append(f"WARNING: Could not access DPE config: {e}")

        return issues

    def _check_loss_weights(self, config: Any) -> List[str]:
        """
        Check loss weight configuration.
        """
        issues = []

        try:
            mf = config.model.magformer.mask_former
            class_weight = getattr(mf, "class_weight", 1.0)
            dice_weight = getattr(mf, "dice_weight", 1.0)
            mask_weight = getattr(mf, "mask_weight", 5.0)
            no_object_weight = getattr(mf, "no_object_weight", 0.1)

            # Reference implementation uses class_weight=2.0, dice_weight=5.0, mask_weight=5.0
            if class_weight < 1.0:
                issues.append(
                    f"WARNING: class_weight={class_weight} < 1.0 may cause weak classification. "
                    "Reference implementation uses class_weight=2.0."
                )

            if dice_weight < 3.0:
                issues.append(
                    f"WARNING: dice_weight={dice_weight} < 3.0 may cause poor mask quality. "
                    "Reference implementation uses dice_weight=5.0."
                )

            if no_object_weight > 0.5:
                issues.append(
                    f"WARNING: no_object_weight={no_object_weight} > 0.5 may suppress "
                    "foreground predictions. Reference uses 0.1."
                )

        except AttributeError as e:
            issues.append(f"WARNING: Could not access loss weight config: {e}")

        return issues

    def _check_training_params(self, config: Any) -> List[str]:
        """
        Check training hyperparameters.
        """
        issues = []

        try:
            solver = config.solver

            # Check learning rate
            base_lr = getattr(solver, "base_lr", 0.0001)
            if base_lr > 0.001:
                issues.append(
                    f"WARNING: base_lr={base_lr} > 0.001 may cause training instability. "
                    "Consider using a lower learning rate."
                )

            # Check gradient clipping
            clip_enabled = getattr(solver, "clip_gradients", True)
            clip_value = getattr(solver, "clip_value", 0.01)
            if not clip_enabled:
                issues.append(
                    "WARNING: Gradient clipping is disabled. "
                    "Reference implementation uses clip_value=1.0."
                )
            elif clip_value > 5.0:
                issues.append(
                    f"WARNING: clip_value={clip_value} > 5.0 may be too high. "
                    "Reference implementation uses clip_value=1.0."
                )

        except AttributeError as e:
            issues.append(f"WARNING: Could not access training config: {e}")

        return issues

    def _check_single_class_contract(self, config: Any) -> List[str]:
        """Check that the shipped MAGFormer path stays single-class."""
        issues = []

        try:
            meta_arch = str(getattr(config.model, "meta_architecture", "")).lower()
            if meta_arch != "magformer":
                return issues

            sem_seg_head = getattr(getattr(config.model, "magformer", object()), "sem_seg_head", None)
            num_classes = int(getattr(sem_seg_head, "num_classes", 1))
            if num_classes != 1:
                issues.append(
                    "CRITICAL: This project currently supports exactly one foreground class. "
                    "Multi-class is not implemented. "
                    f"Set model.magformer.sem_seg_head.num_classes=1 (got {num_classes})."
                )

            class_names = list(getattr(config.data, "class_names", ["component"]))
            if len(class_names) != 1:
                issues.append(
                    "WARNING: This project currently supports exactly one foreground class. Multi-class is not implemented. "
                    "Set data.class_names to a single label for the shipped single-class path."
                )
        except AttributeError as e:
            issues.append(f"WARNING: Could not access single-class config contract: {e}")

        return issues


    # ------------------------------------------------------------------
    # New validation methods
    # ------------------------------------------------------------------

    def _check_normalization(self, config: Any) -> List[str]:
        """
        Check pixel_mean / pixel_std are in reasonable ranges.

        Also warns if normalization is set at the model: top level rather
        than the canonical model.magformer: level.
        """
        issues = []

        try:
            # Try model.magformer first (canonical location)
            mf = getattr(config.model, "magformer", None)
            pixel_mean = getattr(mf, "pixel_mean", None) if mf else None
            pixel_std = getattr(mf, "pixel_std", None) if mf else None

            # Fallback: check model: level
            if pixel_mean is None:
                pixel_mean = getattr(config.model, "pixel_mean", None)
                if pixel_mean is not None:
                    issues.append(
                        "WARNING: pixel_mean is set at model: level. "
                        "The canonical location is model.magformer.pixel_mean."
                    )
            if pixel_std is None:
                pixel_std = getattr(config.model, "pixel_std", None)
                if pixel_std is not None:
                    issues.append(
                        "WARNING: pixel_std is set at model: level. "
                        "The canonical location is model.magformer.pixel_std."
                    )

            # Range checks
            if pixel_mean is not None:
                try:
                    mean_values = list(pixel_mean)
                    for i, v in enumerate(mean_values):
                        if v < 50 or v > 200:
                            issues.append(
                                f"WARNING: pixel_mean[{i}]={v} is outside the "
                                f"reasonable range [50, 200]. Check normalization config."
                            )
                except (TypeError, ValueError):
                    pass

            if pixel_std is not None:
                try:
                    std_values = list(pixel_std)
                    for i, v in enumerate(std_values):
                        if v < 10 or v > 100:
                            issues.append(
                                f"WARNING: pixel_std[{i}]={v} is outside the "
                                f"reasonable range [10, 100]. Check normalization config."
                            )
                except (TypeError, ValueError):
                    pass

        except AttributeError as e:
            issues.append(f"WARNING: Could not access normalization config: {e}")

        return issues

    def _check_swin_config(self, config: Any) -> List[str]:
        """
        Check Swin backbone configuration.

        Validates that num_heads[i] divides (embed_dim * 2^i) for each stage,
        and that depths[i] > 0.
        """
        issues = []

        try:
            swin = getattr(config.model.magformer, "swin", None)
            if swin is None:
                # No Swin backbone configured -- nothing to check
                return issues

            embed_dim = getattr(swin, "embed_dim", 96)
            depths = getattr(swin, "depths", [2, 2, 6, 2])
            num_heads = getattr(swin, "num_heads", [3, 6, 12, 24])

            if len(depths) != len(num_heads):
                issues.append(
                    f"WARNING: Swin depths (len={len(depths)}) and num_heads "
                    f"(len={len(num_heads)}) must have the same length."
                )
                return issues

            for i, (d, h) in enumerate(zip(depths, num_heads)):
                # depths[i] should not be 0
                if d == 0:
                    issues.append(
                        f"WARNING: Swin depths[{i}]=0. Stage {i} will be skipped, "
                        "which is unusual."
                    )

                # num_heads[i] must divide embed_dim * 2^i
                stage_dim = embed_dim * (2 ** i)
                if h <= 0:
                    issues.append(
                        f"WARNING: Swin num_heads[{i}]={h} must be positive."
                    )
                elif stage_dim % h != 0:
                    issues.append(
                        f"WARNING: Swin num_heads[{i}]={h} does not divide "
                        f"stage dim {stage_dim} (embed_dim*2^{i}={embed_dim}*{2**i}). "
                        "Each head dimension must be an integer."
                    )

        except AttributeError as e:
            issues.append(f"WARNING: Could not access Swin config: {e}")

        return issues

    def _check_fusion_config(self, config: Any) -> List[str]:
        """
        Check fusion module configuration.

        Validates that feature_dims is monotonically increasing and warns
        if it differs from the standard [96, 192, 384, 768].
        """
        issues = []

        try:
            mf = getattr(config.model, "magformer", None)
            if mf is None:
                return issues

            fusion = getattr(mf, "modality_fusion", None)
            if fusion is None:
                return issues

            feature_dims = getattr(fusion, "feature_dims", None)
            if feature_dims is None:
                return issues

            dims = list(feature_dims)
            standard_dims = [96, 192, 384, 768]

            # Check monotonically increasing
            for i in range(1, len(dims)):
                if dims[i] <= dims[i - 1]:
                    issues.append(
                        f"WARNING: fusion.feature_dims is not monotonically increasing: "
                        f"{dims}. Stage {i} dim ({dims[i]}) <= stage {i-1} dim ({dims[i-1]})."
                    )

            # Warn if differs from standard
            if dims != standard_dims:
                issues.append(
                    f"WARNING: fusion.feature_dims={dims} differs from standard "
                    f"{standard_dims}. Make sure backbone output dims match."
                )

        except AttributeError as e:
            issues.append(f"WARNING: Could not access fusion config: {e}")

        return issues

    def _check_solver_recipe(self, config: Any) -> List[str]:
        """
        Check solver against the golden training recipe.

        Warns if warm-start uses a high base_lr or if poly scheduler is
        used with very long training.
        """
        issues = []

        try:
            solver = config.solver

            base_lr = float(getattr(solver, "base_lr", 0.0001))

            # Check if warm-starting from a checkpoint
            weight_init = getattr(solver, "weight_init", None)
            has_checkpoint = (
                weight_init is not None
                or getattr(solver, "model_checkpoint", None) is not None
            )

            if has_checkpoint and base_lr > 5e-5:
                issues.append(
                    f"WARNING: Warm-start (finetune) with base_lr={base_lr} > 5e-5. "
                    "Golden recipe uses base_lr=1.5e-5 for warm-start. "
                    "High LR may overwrite pretrained weights too aggressively."
                )

            # Check poly scheduler with long training
            lr_scheduler = getattr(solver, "lr_scheduler", None)
            max_iter = int(getattr(solver, "max_iter", 0))

            if lr_scheduler is not None:
                scheduler_name = str(lr_scheduler).lower()
                if "poly" in scheduler_name and max_iter > 200000:
                    issues.append(
                        f"WARNING: Poly LR scheduler with max_iter={max_iter}. "
                        "Poly decay with very long training drives LR extremely low. "
                        "Consider using cosine schedule or reducing max_iter."
                    )

        except AttributeError as e:
            issues.append(f"WARNING: Could not access solver recipe config: {e}")

        return issues


def validate_config(config: Any, strict: bool = False) -> bool:
    """
    Validate configuration and print issues.

    Args:
        config: MAGFormer configuration object
        strict: If True, exit on critical issues

    Returns:
        True if configuration is valid, False otherwise
    """
    validator = ConfigValidator()
    is_valid, issues = validator.validate(config)

    if issues:
        print("\n=== Configuration Validation ===")
        for issue in issues:
            if issue.startswith("CRITICAL"):
                print(f"  [CRITICAL] {issue[10:]}")
            elif issue.startswith("WARNING"):
                print(f"  [WARNING]  {issue[9:]}")
            elif issue.startswith("INFO"):
                print(f"  [INFO]     {issue[6:]}")
            else:
                print(f"  {issue}")
        print(f"  Overall: {'VALID' if is_valid else 'INVALID'}\n")

    if not is_valid and strict:
        print("ERROR: Critical configuration issues detected!")
        return False

    return is_valid
