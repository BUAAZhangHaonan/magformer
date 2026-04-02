# -*- coding: utf-8 -*-
"""
MAGFormer Configuration Validation

Validates configuration for common issues that could cause training failure.
"""

from typing import Any, List, Tuple
import logging

logger = logging.getLogger(__name__)


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

            if clip_min == 0.0 and clip_max == 1.0 and not per_sample_norm:
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
            dpe_enabled_root = bool(getattr(config, "dpe_enabled", False))
            dpe_cfg_nested = getattr(
                getattr(config.model, "magformer", object()), "dpe", None)
            dpe_enabled_nested = bool(
                getattr(dpe_cfg_nested, "enabled", False))

            if dpe_enabled_root and not dpe_enabled_nested:
                issues.append(
                    "INFO: Using legacy root-level dpe_enabled. "
                    "Prefer model.magformer.dpe.enabled for new configs."
                )

            # Check if pixel decoder has DPE enabled
            try:
                sem_seg_head = config.model.magformer.sem_seg_head
                pixel_decoder = getattr(sem_seg_head, "pixel_decoder_name", "")
                transformer_enc_layers = getattr(
                    sem_seg_head, "transformer_enc_layers", 0)

                if pixel_decoder == "MSDeformAttnPixelDecoder" and transformer_enc_layers > 0:
                    if not dpe_enabled_root and not dpe_enabled_nested:
                        issues.append(
                            "INFO: MSDeformAttnPixelDecoder with transformer_enc_layers>0 "
                            "but dpe_enabled=False. Consider enabling DPE for better "
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
                    "CRITICAL: MAGFormer currently supports single-class RGB-D instance segmentation only. "
                    f"Set model.magformer.sem_seg_head.num_classes=1 (got {num_classes})."
                )

            class_names = list(getattr(config.data, "class_names", ["component"]))
            if len(class_names) != 1:
                issues.append(
                    "WARNING: data.class_names should contain exactly one label for the shipped single-class path."
                )
        except AttributeError as e:
            issues.append(f"WARNING: Could not access single-class config contract: {e}")

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
