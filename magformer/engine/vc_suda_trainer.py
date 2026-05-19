# -*- coding: utf-8 -*-
"""VC-SUDA Trainer: dual forward-pass training for sim2real domain adaptation.

Extends the base Trainer with:
- EMA teacher pseudo-label generation (Stage C+)
- Quality-weighted pseudo-label loss (Stage C+)
- Domain adaptation losses (Stage D+)
- Curriculum scheduling for pseudo-label quality threshold
"""

import time
import random
import inspect
from contextlib import nullcontext
from typing import Dict, Any, Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.amp import autocast

from .trainer import Trainer, _CHECKPOINT_MAX_KEEP_UNSET
from .utils import clip_gradients


class SourceRetentionL2SP:
    """L2-SP regularizer against a detached warm-start parameter snapshot."""

    def __init__(self, model: nn.Module, config: Dict[str, Any]):
        self.enabled = bool(config.get("enabled", False))
        self.weight = float(config.get("weight", 0.0))
        self.include_prefixes = tuple(str(p) for p in config.get("include_prefixes", []) or [])
        self.exclude_prefixes = tuple(str(p) for p in config.get("exclude_prefixes", []) or [])
        self.normalize = bool(config.get("normalize", True))
        self._params: List[tuple[str, nn.Parameter]] = []
        self._reference: Dict[str, torch.Tensor] = {}

        if not self.active:
            self.parameter_names: tuple[str, ...] = ()
            return

        base_model = model.module if hasattr(model, "module") else model
        include_matches = {prefix: 0 for prefix in self.include_prefixes}
        for name, param in base_model.named_parameters():
            if not param.requires_grad:
                continue
            if self.include_prefixes:
                matched_include = False
                for prefix in self.include_prefixes:
                    if name.startswith(prefix):
                        include_matches[prefix] += 1
                        matched_include = True
                if not matched_include:
                    continue
            if self.exclude_prefixes and name.startswith(self.exclude_prefixes):
                continue
            self._params.append((name, param))
            self._reference[name] = param.detach().clone().float()

        unmatched = [prefix for prefix, count in include_matches.items() if count == 0]
        if unmatched:
            raise ValueError(
                "source_retention.include_prefixes did not match trainable parameters: "
                + ", ".join(unmatched)
            )
        if not self._params:
            raise ValueError("source_retention selected no trainable parameters")

        self.parameter_names = tuple(name for name, _ in self._params)

    @property
    def active(self) -> bool:
        return self.enabled and self.weight > 0.0

    def loss(self) -> torch.Tensor:
        total = None
        numel = 0
        for name, param in self._params:
            diff = param.float() - self._reference[name]
            value = diff.pow(2).sum()
            total = value if total is None else total + value
            numel += param.numel()
        if total is None:
            raise RuntimeError("source_retention has no tracked parameters")
        if self.normalize:
            total = total / max(1, numel)
        return total


class VCSUDATrainer(Trainer):
    """
    VC-SUDA trainer with dual forward-pass pipeline.

    Training loop:
    1. Teacher forward on target_weak (no_grad) -> pseudo-labels (Stage C+)
    2. Student forward on source with targets -> supervised loss
    3. Student forward on target_strong -> pseudo-label loss (Stage C+)
    4. Domain adaptation losses (Stage D+)
    5. Single backward + grad clip + optimizer step + EMA update
    """

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        lr_scheduler=None,
        train_loader=None,
        val_loader=None,
        val_dataset=None,
        config=None,
        device=torch.device("cuda"),
        output_dir="output",
        max_iter=100000,
        eval_period=5000,
        checkpoint_period=5000,
        checkpoint_max_keep=_CHECKPOINT_MAX_KEEP_UNSET,
        log_period=100,
        amp_enabled=True,
        clip_gradients=True,
        clip_value=1.0,
        resume=None,
        logger_config=None,
        # VC-SUDA specific
        ema_teacher=None,
        pseudo_label_scorer=None,
        curriculum_scheduler=None,
        vc_suda_config=None,
        domain_losses=None,
        smoke_test=False,
        use_uncertainty_weighting=False,
        uw_module=None,
    ):
        # VC-SUDA has extra resume state. Parent Trainer calls self.resume()
        # from its constructor, so defer resume until VC-specific attributes exist.
        pending_resume = resume
        super().__init__(
            model=model,
            criterion=None,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            train_loader=train_loader,
            val_loader=val_loader,
            val_dataset=val_dataset,
            config=config,
            device=device,
            output_dir=output_dir,
            max_iter=max_iter,
            eval_period=eval_period,
            checkpoint_period=checkpoint_period,
            checkpoint_max_keep=checkpoint_max_keep,
            log_period=log_period,
            amp_enabled=amp_enabled,
            clip_gradients=clip_gradients,
            clip_value=clip_value,
            resume=None,
            logger_config=logger_config,
        )

        # VC-SUDA components
        self.ema_teacher = ema_teacher
        self.pseudo_label_scorer = pseudo_label_scorer
        self.curriculum_scheduler = curriculum_scheduler
        self.vc_suda_config = vc_suda_config or {}
        self.domain_losses = domain_losses or {}
        self.smoke_test = smoke_test

        # VCSUDACriterion (wraps SetCriterion + pseudo-label loss)
        self.criterion = criterion.to(device) if isinstance(criterion, nn.Module) else criterion

        # Stage info
        self.stage = self.vc_suda_config.get("stage", "A")
        offline_cfg = self.vc_suda_config.get("offline_pseudo", {}) or {}
        self.offline_pseudo_enabled = bool(offline_cfg.get("enabled", False))
        self.use_ema = self.stage in ("C", "D", "E") and not self.offline_pseudo_enabled
        self.use_domain_losses = self.stage in ("D", "E")
        self.use_pseudo_labels = self.stage in ("C", "D", "E")
        self.target_labeled_weight = float(
            self.vc_suda_config.get("target_labeled_weight", 1.0)
        )

        # Epoch tracking (for curriculum)
        self.current_epoch = 0
        self._samples_this_epoch = 0
        try:
            self._iters_per_epoch = len(train_loader) if train_loader is not None else None
        except TypeError:
            self._iters_per_epoch = None
        self._samples_per_epoch = self._iters_per_epoch

        # Unpack domain loss modules
        self.prototype_loss = self._module_to_device(self.domain_losses.get("prototype"))
        self.boundary_loss = self._module_to_device(self.domain_losses.get("boundary"))
        self.modality_dropout_loss = self._module_to_device(self.domain_losses.get("modality_dropout"))

        # Domain loss weights
        da_cfg = self.vc_suda_config.get("domain_adaptation", {})
        self.prototype_weight = da_cfg.get("prototype_weight", 1.0)
        self.boundary_weight = da_cfg.get("boundary_weight", 0.5)
        self.modality_dropout_weight = da_cfg.get("modality_dropout_weight", 0.3)
        self.modality_dropout_prob = da_cfg.get("modality_dropout_prob", 0.3)

        if self.use_pseudo_labels:
            if self.offline_pseudo_enabled:
                if self.ema_teacher is not None:
                    raise ValueError("offline pseudo mode requires ema_teacher=None")
                if self.pseudo_label_scorer is not None:
                    raise ValueError("offline pseudo mode requires pseudo_label_scorer=None")
            else:
                if self.ema_teacher is None:
                    raise ValueError("VC-SUDA Stage C+ requires an EMA teacher")
                if self.pseudo_label_scorer is None:
                    raise ValueError("VC-SUDA Stage C+ requires a pseudo-label scorer")
        if self.use_domain_losses:
            missing_domain = []
            if self.prototype_weight > 0 and self.prototype_loss is None:
                missing_domain.append("prototype")
            if self.boundary_weight > 0 and self.boundary_loss is None:
                missing_domain.append("boundary")
            if self.modality_dropout_weight > 0 and self.modality_dropout_loss is None:
                missing_domain.append("modality_dropout")
            if missing_domain:
                raise ValueError(
                    "VC-SUDA Stage D+ has positive domain loss weights but missing modules: "
                    + ", ".join(missing_domain)
                )

        # Uncertainty weighting is learnable. It must be created before optimizer
        # construction, moved to the trainer device, and included in optimizer groups.
        self.use_uncertainty_weighting = use_uncertainty_weighting
        self.uw = None
        if self.use_uncertainty_weighting:
            if uw_module is None:
                raise ValueError(
                    "UncertaintyWeighting must be constructed by the entrypoint and added to the optimizer; "
                    "set use_uncertainty_weighting=False until that wiring exists."
                )
            self.uw = self._module_to_device(uw_module)
            self._assert_optimizer_owns_module(self.uw, "UncertaintyWeighting")
            self._console_log("[VCSUDA] UncertaintyWeighting enabled (3 tasks, optimizer-managed)")

        if self.use_ema and self.ema_teacher is not None:
            self._console_log(
                f"[VCSUDA] EMA teacher enabled, momentum={self.ema_teacher.momentum}, "
                f"warmup_steps={self.ema_teacher.warmup_steps}"
            )
        if self.use_domain_losses:
            self._console_log(
                f"[VCSUDA] Domain losses: proto={self.prototype_weight}, "
                f"boundary={self.boundary_weight}, moddrop={self.modality_dropout_weight}"
            )

        if pending_resume is not None:
            self.resume(pending_resume)

        self.source_retention = SourceRetentionL2SP(
            self.model,
            self.vc_suda_config.get("source_retention", {}),
        )

    def _module_to_device(self, module):
        if isinstance(module, nn.Module):
            return module.to(self.device)
        return module

    def _assert_optimizer_owns_module(self, module: nn.Module, name: str) -> None:
        module_param_ids = {id(p) for p in module.parameters() if p.requires_grad}
        if not module_param_ids:
            return
        optimizer_param_ids = {
            id(p)
            for group in self.optimizer.param_groups
            for p in group.get("params", [])
        }
        if not module_param_ids.issubset(optimizer_param_ids):
            raise ValueError(f"{name} parameters are not included in the optimizer")

    def _batch_tensor(self, batch: Dict[str, Any], key: str) -> Optional[torch.Tensor]:
        value = batch.get(key)
        if torch.is_tensor(value):
            return value.to(self.device)
        return value

    @staticmethod
    def _accepts_kwarg(callable_obj: Any, kwarg: str) -> bool:
        target = callable_obj.forward if isinstance(callable_obj, nn.Module) else callable_obj
        try:
            signature = inspect.signature(target)
        except (TypeError, ValueError):
            return False
        if kwarg in signature.parameters:
            return True
        return any(
            param.kind == inspect.Parameter.VAR_KEYWORD
            for param in signature.parameters.values()
        )

    def _call_with_depth_valid_masks(
        self,
        callable_obj: Any,
        *args: Any,
        depth_valid_masks: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> Any:
        if depth_valid_masks is not None and self._accepts_kwarg(callable_obj, "depth_valid_masks"):
            kwargs["depth_valid_masks"] = depth_valid_masks
        return callable_obj(*args, **kwargs)

    def _train_step(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """VC-SUDA training step with dual forward pass."""
        iter_start = time.perf_counter()

        # Determine epoch for curriculum from dataloader iterations.
        if self._iters_per_epoch:
            self.current_epoch = self.current_iter // self._iters_per_epoch
        self._samples_this_epoch += 1

        # Move source data to device. Key names come from SemiSupervisedDataset.collate_fn.
        source_images = batch["source_images"].to(self.device)
        source_depths = batch["source_depths"].to(self.device)
        source_depth_valid_masks = self._batch_tensor(batch, "source_depth_valid_masks")
        source_padding_masks = self._batch_tensor(batch, "source_padding_masks")
        source_noise_masks = self._batch_tensor(batch, "source_noise_masks")
        source_targets = batch.get("source_annotations", [])
        source_targets = self._prepare_targets(source_targets, batch)
        target_outputs = None

        amp_ctx = autocast("cuda") if self.amp_enabled else nullcontext()

        # ================================================================
        # Stage C+: Pseudo-label generation via teacher
        # ================================================================
        pseudo_targets = None
        pseudo_label_metrics = None
        offline_pseudo_annotations = batch.get("target_unlabeled_pseudo_annotations")
        if self.use_pseudo_labels and self.offline_pseudo_enabled:
            if offline_pseudo_annotations is None:
                raise ValueError(
                    "VC-SUDA offline pseudo mode requires offline pseudo annotations "
                    "under target_unlabeled_pseudo_annotations in every Stage C+ training batch."
                )
            pseudo_targets = self._prepare_offline_pseudo_targets(
                offline_pseudo_annotations
            )
            pseudo_label_metrics = self._compute_offline_pseudo_label_metrics(
                pseudo_targets
            )
        elif self.use_pseudo_labels and self.ema_teacher is not None:
            target_weak_images = batch.get("target_weak_images")
            target_weak_depths = batch.get("target_weak_depths")

            if target_weak_images is not None:
                target_weak_images = target_weak_images.to(self.device)
                target_weak_depths = target_weak_depths.to(self.device)
                target_weak_depth_valid_masks = self._batch_tensor(batch, "target_weak_depth_valid_masks")
                target_weak_padding_masks = self._batch_tensor(batch, "target_weak_padding_masks")
                target_weak_noise_masks = self._batch_tensor(batch, "target_weak_noise_masks")

                with torch.no_grad():
                    teacher_outputs = self._call_with_depth_valid_masks(
                        self.ema_teacher,
                        target_weak_images,
                        target_weak_depths,
                        padding_masks=target_weak_padding_masks,
                        depth_noise_masks=target_weak_noise_masks,
                        depth_valid_masks=target_weak_depth_valid_masks,
                    )

                # Score and filter pseudo-labels
                if self.pseudo_label_scorer is not None:
                    scored = self.pseudo_label_scorer.score(
                        teacher_outputs, target_weak_depths
                    )

                    # Get threshold from curriculum
                    if self.curriculum_scheduler is not None:
                        threshold = self.curriculum_scheduler.get_threshold(
                            self.current_epoch
                        )
                    else:
                        threshold = self.vc_suda_config.get(
                            "pseudo_label", {}
                        ).get("quality_threshold", 0.5)

                    filtered = self.pseudo_label_scorer.filter_by_threshold(
                        scored, threshold
                    )
                    pseudo_label_metrics = self._compute_pseudo_label_metrics(
                        scored, filtered, threshold
                    )

                    # Convert to pseudo_targets format for VCSUDACriterion
                    pseudo_targets = self._build_pseudo_targets(filtered)

        # ================================================================
        # Student supervised forward on source
        # ================================================================
        with amp_ctx:
            # Forward student on source (with targets -> computes loss internally)
            supervised_outputs = self._call_with_depth_valid_masks(
                self.model,
                source_images,
                source_depths,
                source_targets,
                padding_masks=source_padding_masks,
                depth_noise_masks=source_noise_masks,
                depth_valid_masks=source_depth_valid_masks,
            )
            # supervised_outputs is already a loss dict from model forward

        # Compute supervised loss
        if isinstance(supervised_outputs, dict) and "total_loss" in supervised_outputs:
            supervised_losses = supervised_outputs
        else:
            # If model didn't return loss dict, compute manually
            supervised_losses = self.criterion.supervised_loss(
                supervised_outputs, source_targets
            )

        total_loss = supervised_losses["total_loss"]
        supervised_losses["source_total_loss"] = total_loss.detach()
        if pseudo_label_metrics is not None:
            metric_device = total_loss.device if torch.is_tensor(total_loss) else self.device
            for key, value in pseudo_label_metrics.items():
                supervised_losses[key] = torch.tensor(
                    value, dtype=torch.float32, device=metric_device
                )

        # ================================================================
        # Stage B: Target labeled forward (small labeled subset)
        # ================================================================
        if batch.get("target_labeled_images") is not None:
            tl_images = batch["target_labeled_images"].to(self.device)
            tl_depths = batch["target_labeled_depths"].to(self.device)
            tl_depth_valid_masks = self._batch_tensor(batch, "target_labeled_depth_valid_masks")
            tl_targets = batch.get("target_labeled_annotations", [])
            tl_targets = self._prepare_targets(tl_targets, batch)
            tl_padding_masks = self._batch_tensor(batch, "target_labeled_padding_masks")
            tl_noise_masks = self._batch_tensor(batch, "target_labeled_noise_masks")
            with amp_ctx:
                tl_outputs = self._call_with_depth_valid_masks(
                    self.model,
                    tl_images,
                    tl_depths,
                    tl_targets,
                    padding_masks=tl_padding_masks,
                    depth_noise_masks=tl_noise_masks,
                    depth_valid_masks=tl_depth_valid_masks,
                )
            if isinstance(tl_outputs, dict) and "total_loss" in tl_outputs:
                tl_total_loss_raw = tl_outputs["total_loss"]
                tl_weight = tl_total_loss_raw.new_tensor(self.target_labeled_weight)
                tl_total_loss_weighted = tl_total_loss_raw * tl_weight
                total_loss = total_loss + tl_total_loss_weighted
                supervised_losses["tl_total_loss_raw"] = tl_total_loss_raw.detach()
                supervised_losses["tl_total_loss_weighted"] = tl_total_loss_weighted.detach()
                supervised_losses["tl_weight"] = tl_weight
                for k, v in tl_outputs.items():
                    if k != "total_loss":
                        supervised_losses[f"tl_{k}"] = v

        # ================================================================
        # Stage C+: Student pseudo-label forward on target_strong
        # ================================================================
        if self.use_pseudo_labels and pseudo_targets is not None:
            target_strong_images = batch.get("target_strong_images")
            target_strong_depths = batch.get("target_strong_depths")

            if target_strong_images is not None:
                target_strong_images = target_strong_images.to(self.device)
                target_strong_depths = target_strong_depths.to(self.device)
                target_strong_depth_valid_masks = self._batch_tensor(batch, "target_strong_depth_valid_masks")
                target_strong_padding_masks = self._batch_tensor(batch, "target_strong_padding_masks")
                target_strong_noise_masks = self._batch_tensor(batch, "target_strong_noise_masks")

                with amp_ctx:
                    # Raw decoder outputs are required because the model raises in training
                    # mode when targets are None unless return_features=True.
                    target_outputs = self._call_with_depth_valid_masks(
                        self.model,
                        target_strong_images,
                        target_strong_depths,
                        targets=None,
                        padding_masks=target_strong_padding_masks,
                        depth_noise_masks=target_strong_noise_masks,
                        depth_valid_masks=target_strong_depth_valid_masks,
                        return_features=True,
                    )

                # Pseudo-label loss
                if self.offline_pseudo_enabled:
                    pseudo_targets = self._align_offline_pseudo_targets_to_outputs(
                        pseudo_targets,
                        target_outputs,
                    )
                pseudo_losses = self.criterion.pseudo_label_loss(
                    target_outputs, pseudo_targets
                )

                # Unsupervised weight ramp-up
                unsup_weight = self._get_unsupervised_weight()
                pseudo_total = pseudo_losses.get(
                    "pseudo_total", torch.tensor(0.0, device=self.device)
                )
                unsup_weight_tensor = pseudo_total.new_tensor(unsup_weight)
                supervised_losses["unsupervised_weight"] = unsup_weight_tensor.detach()
                for k, v in pseudo_losses.items():
                    supervised_losses[k] = v
                total_loss = total_loss + unsup_weight_tensor * pseudo_total

        # ============================================================
        # Domain adaptation losses (independent of pseudo-labels)
        # ============================================================
        if self.use_domain_losses:
            _dl_target_images = batch.get("target_strong_images")
            _dl_target_depths = batch.get("target_strong_depths")
            if _dl_target_images is None:
                _dl_target_images = batch.get("target_unlabeled_images")
                _dl_target_depths = batch.get("target_unlabeled_depths")
            if _dl_target_images is not None:
                _dl_target_images = _dl_target_images.to(self.device)
                _dl_target_depths = _dl_target_depths.to(self.device)
                if batch.get("target_strong_images") is not None:
                    _dl_target_depth_valid_masks = self._batch_tensor(batch, "target_strong_depth_valid_masks")
                    _dl_target_padding_masks = self._batch_tensor(batch, "target_strong_padding_masks")
                    _dl_target_noise_masks = self._batch_tensor(batch, "target_strong_noise_masks")
                else:
                    _dl_target_depth_valid_masks = self._batch_tensor(batch, "target_unlabeled_depth_valid_masks")
                    _dl_target_padding_masks = self._batch_tensor(batch, "target_unlabeled_padding_masks")
                    _dl_target_noise_masks = self._batch_tensor(batch, "target_unlabeled_noise_masks")

                # Reuse target_outputs from pseudo-label block if available,
                # otherwise compute a fresh raw-output forward pass.
                if target_outputs is None:
                    with amp_ctx:
                        target_outputs = self._call_with_depth_valid_masks(
                            self.model,
                            _dl_target_images,
                            _dl_target_depths,
                            targets=None,
                            padding_masks=_dl_target_padding_masks,
                            depth_noise_masks=_dl_target_noise_masks,
                            depth_valid_masks=_dl_target_depth_valid_masks,
                            return_features=True,
                        )

                if self.use_domain_losses and target_outputs is not None:
                    # Boundary consistency loss
                    _boundary_loss_val = None
                    if (
                        self.boundary_loss is not None
                        and self.boundary_weight > 0
                    ):
                        target_pred_masks = (
                            target_outputs.get("pred_masks", torch.empty(0))
                            .sigmoid()
                        )
                        _boundary_loss_val = self.boundary_loss(
                            target_pred_masks, _dl_target_depths
                        )
                        supervised_losses["loss_boundary"] = _boundary_loss_val
                        if not self.use_uncertainty_weighting:
                            total_loss = total_loss + self.boundary_weight * _boundary_loss_val

                    # Prototype alignment loss
                    _proto_loss_val = None
                    if (
                        self.prototype_loss is not None
                        and self.prototype_weight > 0
                        and "features" in target_outputs
                    ):
                        # Source features from supervised forward
                        with torch.no_grad():
                            source_feat_outputs = self._call_with_depth_valid_masks(
                                self.model,
                                source_images,
                                source_depths,
                                targets=None,
                                padding_masks=source_padding_masks,
                                depth_noise_masks=source_noise_masks,
                                depth_valid_masks=source_depth_valid_masks,
                                return_features=True,
                            )

                        source_masks_sig = (
                            source_feat_outputs.get(
                                "pred_masks",
                                torch.zeros(
                                    1,
                                    1,
                                    1,
                                    1,
                                    device=self.device,
                                ),
                            )
                            .sigmoid()
                        )
                        target_masks_sig = target_outputs.get(
                            "pred_masks", torch.empty(0)
                        ).sigmoid()

                        _proto_loss_val = self.prototype_loss(
                            source_feat_outputs["features"],
                            target_outputs["features"],
                            source_masks_sig,
                            target_masks_sig,
                        )
                        supervised_losses["loss_prototype"] = _proto_loss_val
                        if not self.use_uncertainty_weighting:
                            total_loss = total_loss + self.prototype_weight * _proto_loss_val

                    # Modality dropout consistency loss
                    _moddrop_loss_val = None
                    if (
                        self.modality_dropout_loss is not None
                        and self.modality_dropout_weight > 0
                    ):
                        if random.random() < self.modality_dropout_prob:
                            # Forward with zeroed depth
                            zeroed_depth = torch.zeros_like(
                                _dl_target_depths
                            )
                            with amp_ctx:
                                dropped_outputs = self._call_with_depth_valid_masks(
                                    self.model,
                                    _dl_target_images,
                                    zeroed_depth,
                                    targets=None,
                                    padding_masks=_dl_target_padding_masks,
                                    depth_noise_masks=_dl_target_noise_masks,
                                    depth_valid_masks=_dl_target_depth_valid_masks,
                                    return_features=True,
                                )

                            # Full predictions (stop-gradient)
                            # target_outputs from above is the full RGBD forward
                            _moddrop_loss_val = self.modality_dropout_loss(
                                target_outputs, dropped_outputs
                            )
                            supervised_losses["loss_moddrop"] = _moddrop_loss_val
                            if not self.use_uncertainty_weighting:
                                total_loss = (
                                    total_loss
                                    + self.modality_dropout_weight * _moddrop_loss_val
                                )

                    # UW combination of domain losses
                    if self.use_uncertainty_weighting:
                        _uw_losses = [_proto_loss_val, _boundary_loss_val, _moddrop_loss_val]
                        if any(l is not None and isinstance(l, torch.Tensor) for l in _uw_losses):
                            uw_total, uw_weighted = self.uw(
                                _proto_loss_val, _boundary_loss_val, _moddrop_loss_val
                            )
                            total_loss = total_loss + uw_total
                            for uk, uv in uw_weighted.items():
                                supervised_losses[uk] = uv

        if self.source_retention.active:
            source_retention_l2sp = self.source_retention.loss()
            source_retention_weighted = source_retention_l2sp * total_loss.new_tensor(
                self.source_retention.weight
            )
            total_loss = total_loss + source_retention_weighted
            supervised_losses["source_retention_l2sp"] = source_retention_l2sp.detach()
            supervised_losses["source_retention_weighted"] = source_retention_weighted.detach()

        supervised_losses["total_loss"] = total_loss

        # ================================================================
        # Backward + optimizer step, preserving base Trainer accumulation.
        # ================================================================
        if self._accum_count == 0:
            self.optimizer.zero_grad()

        accum_loss = total_loss / self.grad_accum_steps
        is_last_accum = self._accum_count == self.grad_accum_steps - 1
        use_no_sync = (not is_last_accum) and self.distributed and hasattr(self.model, "no_sync")
        sync_ctx = self.model.no_sync() if use_no_sync else nullcontext()

        with sync_ctx:
            if self.amp_enabled:
                self.scaler.scale(accum_loss).backward()
            else:
                accum_loss.backward()

        self._accum_count += 1
        optimizer_stepped = False

        if self._accum_count >= self.grad_accum_steps:
            if self.clip_gradients:
                if self.amp_enabled:
                    self.scaler.unscale_(self.optimizer)
                clip_gradients(self.model, self.clip_value)

            if self.amp_enabled:
                prev_scale = self.scaler.get_scale()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                optimizer_stepped = self.scaler.get_scale() >= prev_scale
                if optimizer_stepped:
                    if hasattr(self.optimizer, "_opt_called"):
                        self.optimizer._opt_called = True
                    if hasattr(self.optimizer, "_step_count") and self.optimizer._step_count == 0:
                        self.optimizer._step_count = 1
            else:
                self.optimizer.step()
                optimizer_stepped = True

            self._accum_count = 0

            if self.lr_scheduler is not None and optimizer_stepped:
                self.lr_scheduler.step()

            # EMA teacher update
            if self.use_ema and self.ema_teacher is not None and optimizer_stepped:
                self.ema_teacher.update_ema(self.model, self.current_iter)

        iter_time_sec = time.perf_counter() - iter_start
        self._iter_time_window_sec.append(iter_time_sec)

        # Logging
        if self.current_iter % self.log_period == 0:
            self._log_training(supervised_losses)

        self.current_iter += 1

        # Smoke test: exit after 2 iterations
        if self.smoke_test and self.current_iter >= 2:
            self._console_log(
                f"[VCSUDA] Smoke test complete after {self.current_iter} iters"
            )
            self.save_checkpoint(is_best=False)
            raise StopIteration("Smoke test complete")

        return supervised_losses

    def _align_offline_pseudo_targets_to_outputs(
        self,
        pseudo_targets: List[Dict[str, Any]],
        target_outputs: Dict[str, torch.Tensor],
    ) -> List[Dict[str, Any]]:
        """Match fixed-bank image-space masks to the pseudo-loss output grid."""
        pred_masks = target_outputs.get("pred_masks")
        if pred_masks is None:
            raise ValueError("target_outputs missing pred_masks for offline pseudo loss")
        output_size = tuple(int(v) for v in pred_masks.shape[-2:])
        aligned = []
        for index, target in enumerate(pseudo_targets):
            masks = target.get("masks")
            if masks is None:
                aligned.append(target)
                continue
            if masks.ndim != 3:
                raise ValueError(
                    f"offline pseudo annotation {index} masks must have shape (N,H,W), "
                    f"got {tuple(masks.shape)}"
                )
            if tuple(int(v) for v in masks.shape[-2:]) == output_size:
                aligned.append(target)
                continue
            resized_target = dict(target)
            if int(masks.shape[0]) == 0:
                resized_target["masks"] = masks.new_zeros((0, *output_size))
            else:
                resized = F.interpolate(
                    masks.float().unsqueeze(1),
                    size=output_size,
                    mode="nearest",
                ).squeeze(1)
                if masks.dtype == torch.bool:
                    resized = resized > 0.5
                else:
                    resized = resized.to(dtype=masks.dtype)
                resized_target["masks"] = resized
            aligned.append(resized_target)
        return aligned


    def _build_pseudo_targets(
        self, scored_results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Convert scored results to pseudo_targets format for VCSUDACriterion."""
        pseudo_targets = []
        for result in scored_results:
            pseudo_targets.append(
                {
                    "labels": result["labels"],
                    "masks": result["masks"],
                    "quality_scores": result["scores"],
                }
            )
        return pseudo_targets

    def _prepare_offline_pseudo_targets(
        self, annotations: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Move fixed offline pseudo annotations to the trainer device."""
        pseudo_targets = []
        for index, annotation in enumerate(annotations):
            missing = [
                key
                for key in ("labels", "masks", "quality_scores")
                if key not in annotation
            ]
            if missing:
                raise ValueError(
                    f"offline pseudo annotation {index} missing required fields {missing}"
                )
            target: Dict[str, Any] = {}
            for key, value in annotation.items():
                if torch.is_tensor(value):
                    target[key] = value.to(self.device)
                elif key in {"labels", "masks", "boxes", "quality_scores", "fill_ratios"}:
                    target[key] = torch.as_tensor(value, device=self.device)
                else:
                    target[key] = value
            labels = target["labels"]
            masks = target["masks"]
            quality_scores = target["quality_scores"]
            count = int(labels.numel())
            if masks.shape[0] != count or quality_scores.numel() != count:
                raise ValueError(
                    "offline pseudo labels, masks, and quality_scores must have matching lengths "
                    f"for annotation {index}: labels={count}, masks={masks.shape[0]}, "
                    f"quality_scores={quality_scores.numel()}"
                )
            if "fill_ratios" in target and target["fill_ratios"].numel() != count:
                raise ValueError(
                    "offline pseudo labels and fill_ratios must have matching lengths "
                    f"for annotation {index}: labels={count}, fill_ratios={target['fill_ratios'].numel()}"
                )
            pseudo_targets.append(target)
        return pseudo_targets

    @staticmethod
    def _compute_offline_pseudo_label_metrics(
        pseudo_targets: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        count = 0
        empty_images = 0
        quality_sum = 0.0
        fill_sum = 0.0
        fill_count = 0
        pre_filter_count = 0
        dropped_by_filter = 0
        dropped_by_max_instances = 0
        for target in pseudo_targets:
            labels = target["labels"]
            target_count = int(labels.numel())
            count += target_count
            if target_count == 0:
                empty_images += 1
            quality_scores = target["quality_scores"].detach().float()
            quality_sum += float(quality_scores.sum().item())
            fill_ratios = target.get("fill_ratios")
            if fill_ratios is not None:
                fill_values = fill_ratios.detach().float()
                fill_sum += float(fill_values.sum().item())
                fill_count += int(fill_values.numel())
            pre_filter_count += int(target.get("offline_pseudo_pre_filter_count", target_count))
            dropped_by_filter += int(target.get("offline_pseudo_dropped_by_filter", 0))
            dropped_by_max_instances += int(target.get("offline_pseudo_dropped_by_max_instances", 0))

        weighted_total_ratio = quality_sum / float(count) if count else 0.0
        fill_ratio_mean = fill_sum / float(fill_count) if fill_count else 0.0
        return {
            "pseudo_offline_mode": 1.0,
            "pseudo_offline_count": float(count),
            "pseudo_offline_empty_images": float(empty_images),
            "pseudo_offline_pre_filter_count": float(pre_filter_count),
            "pseudo_offline_dropped_by_filter": float(dropped_by_filter),
            "pseudo_offline_dropped_by_max_instances": float(dropped_by_max_instances),
            "pseudo_offline_quality_sum": float(quality_sum),
            "pseudo_offline_weighted_total_ratio": float(weighted_total_ratio),
            "pseudo_offline_fill_ratio_mean": float(fill_ratio_mean),
            "pseudo_kept_count": float(count),
            "pseudo_empty_images": float(empty_images),
        }

    @staticmethod
    def _pseudo_label_count(result: Dict[str, Any]) -> int:
        scores = result.get("scores")
        if scores is None:
            return 0
        if torch.is_tensor(scores):
            return int(scores.numel())
        return int(len(scores))

    @staticmethod
    def _compute_pseudo_label_metrics(
        scored_results: List[Dict[str, Any]],
        filtered_results: List[Dict[str, Any]],
        threshold: float,
    ) -> Dict[str, float]:
        """Summarize pseudo-label filtering for structured train metrics."""
        candidate_count = sum(
            VCSUDATrainer._pseudo_label_count(result) for result in scored_results
        )
        kept_count = sum(
            VCSUDATrainer._pseudo_label_count(result) for result in filtered_results
        )
        empty_images = sum(
            1
            for result in filtered_results
            if VCSUDATrainer._pseudo_label_count(result) == 0
        )
        keep_rate = float(kept_count) / float(candidate_count) if candidate_count else 0.0

        return {
            "pseudo_kept_count": float(kept_count),
            "pseudo_empty_images": float(empty_images),
            "pseudo_threshold": float(threshold),
            "pseudo_keep_rate": float(keep_rate),
        }

    def _get_unsupervised_weight(self) -> float:
        """Get current unsupervised loss weight with warmup ramp."""
        max_weight = float(self.vc_suda_config.get("unsupervised_weight", 1.0))
        warmup_iters = int(self.vc_suda_config.get("unsupervised_warmup_iters", 0))
        if warmup_iters > 0:
            progress = min(1.0, float(self.current_iter + 1) / float(warmup_iters))
            return float(max_weight * (progress ** 2))
        if self.curriculum_scheduler is not None:
            warmup_epochs = self.vc_suda_config.get(
                "unsupervised_warmup_epochs", 10
            )
            if self._iters_per_epoch:
                progress_epoch = (self.current_iter + 1) / float(self._iters_per_epoch)
                if progress_epoch >= warmup_epochs:
                    return max_weight
                progress = progress_epoch / max(float(warmup_epochs), 1.0)
                return max_weight * (progress ** 2)
            return self.curriculum_scheduler.get_unsupervised_weight(
                self.current_epoch + 1, max_weight=max_weight, warmup_epochs=warmup_epochs
            )
        return max_weight

    def save_checkpoint(self, is_best: bool = False) -> None:
        """Save checkpoint with VC-SUDA components."""
        checkpoint = {
            "iter": self.current_iter,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_metric": self.best_metric,
            "config": self.config,
            "current_epoch": self.current_epoch,
        }

        if self.lr_scheduler is not None:
            checkpoint["lr_scheduler_state_dict"] = self.lr_scheduler.state_dict()

        if self.scaler is not None:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()

        # VC-SUDA specific state
        if self.ema_teacher is not None:
            checkpoint["ema_teacher_state_dict"] = self.ema_teacher.state_dict()

        if self.curriculum_scheduler is not None:
            checkpoint["curriculum_state_dict"] = self.curriculum_scheduler.state_dict()

        if hasattr(self, "uw") and self.uw is not None:
            checkpoint["uw_state_dict"] = self.uw.state_dict()

        filename = self.output_dir / f"checkpoint_iter_{self.current_iter:07d}.pth"
        from .utils import save_checkpoint as _save
        _save(checkpoint, filename, is_best=is_best)
        self._cleanup_old_checkpoints(max_keep=self.checkpoint_max_keep)

    def resume(self, checkpoint_path: str) -> None:
        """Resume training with VC-SUDA components."""
        from .utils import load_checkpoint

        print(f"[VCSUDATrainer] Resuming from {checkpoint_path}...")
        checkpoint = load_checkpoint(checkpoint_path, self.model, self.optimizer)

        self.start_iter = checkpoint.get("iter", 0)
        self.current_iter = self.start_iter
        self.best_metric = checkpoint.get("best_metric", float("-inf"))
        self.current_epoch = checkpoint.get("current_epoch", 0)

        if self.lr_scheduler is not None and "lr_scheduler_state_dict" in checkpoint:
            self.lr_scheduler.load_state_dict(
                checkpoint["lr_scheduler_state_dict"]
            )

        if self.scaler is not None and "scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])

        if self.use_ema and self.ema_teacher is not None:
            if "ema_teacher_state_dict" not in checkpoint:
                raise ValueError(
                    "VC-SUDA Stage C+ runtime.resume checkpoint is missing "
                    "ema_teacher_state_dict; use model.finetune_weights for "
                    "model-only warm-start instead."
                )
            self.ema_teacher.load_state_dict(
                checkpoint["ema_teacher_state_dict"]
            )
            print("[VCSUDATrainer] Restored EMA teacher state")

        if self.curriculum_scheduler is not None and "curriculum_state_dict" in checkpoint:
            self.curriculum_scheduler.load_state_dict(
                checkpoint["curriculum_state_dict"]
            )

        if hasattr(self, "uw") and self.uw is not None and "uw_state_dict" in checkpoint:
            self.uw.load_state_dict(checkpoint["uw_state_dict"])
            print("[VCSUDATrainer] Restored UncertaintyWeighting state")

        print(
            f"[VCSUDATrainer] Resumed from iter {self.start_iter}, "
            f"epoch {self.current_epoch}, best_metric {self.best_metric:.4f}"
        )


class VCSUDADDPTrainer(VCSUDATrainer):
    """Distributed VC-SUDA trainer."""

    def __init__(self, *args, **kwargs):
        find_unused_parameters = bool(
            kwargs.pop("find_unused_parameters", False)
        )
        super().__init__(*args, **kwargs)

        self.distributed = True
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()
        self.local_rank = (
            torch.cuda.current_device() if torch.cuda.is_available() else self.rank
        )

        # Class-weight buffers are constants; per-forward broadcasts mutate their
        # autograd version across Stage B source + target_labeled forwards.
        self.model = torch.nn.parallel.DistributedDataParallel(
            self.model,
            device_ids=[self.local_rank],
            find_unused_parameters=find_unused_parameters,
            broadcast_buffers=False,
        )

        print(
            f"[VCSUDADDPTrainer] rank {self.rank}/{self.world_size}, "
            f"stage={self.stage}"
        )

    def _log_training(self, losses: Dict[str, torch.Tensor]) -> None:
        """Only log on rank 0."""
        if self.rank == 0:
            super()._log_training(losses)

    def save_checkpoint(self, is_best: bool = False) -> None:
        """Only save on rank 0."""
        if self.rank == 0:
            # Unwrap DDP for state dict
            model_state = self.model.module.state_dict()
            checkpoint = {
                "iter": self.current_iter,
                "model_state_dict": model_state,
                "optimizer_state_dict": self.optimizer.state_dict(),
                "best_metric": self.best_metric,
                "config": self.config,
                "current_epoch": self.current_epoch,
            }

            if self.lr_scheduler is not None:
                checkpoint["lr_scheduler_state_dict"] = self.lr_scheduler.state_dict()

            if self.scaler is not None:
                checkpoint["scaler_state_dict"] = self.scaler.state_dict()

            if self.ema_teacher is not None:
                checkpoint["ema_teacher_state_dict"] = self.ema_teacher.state_dict()

            if self.curriculum_scheduler is not None:
                checkpoint["curriculum_state_dict"] = self.curriculum_scheduler.state_dict()

            if hasattr(self, "uw") and self.uw is not None:
                checkpoint["uw_state_dict"] = self.uw.state_dict()

            filename = self.output_dir / f"checkpoint_iter_{self.current_iter:07d}.pth"
            from .utils import save_checkpoint as _save
            _save(checkpoint, filename, is_best=is_best)
            self._cleanup_old_checkpoints(max_keep=self.checkpoint_max_keep)

    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """Distributed evaluation."""
        self.model.eval()
        from .eval_runtime import run_inference_evaluation

        category_ids = (
            list(getattr(self.val_dataset, "category_ids", [])) or None
        )
        result = run_inference_evaluation(
            self.model,
            self.val_loader,
            coco_gt=getattr(self.val_dataset, "coco", None),
            device=self.device,
            output_dir=self.output_dir,
            amp_enabled=self.amp_enabled,
            category_ids=category_ids,
            iou_types=getattr(self, "eval_iou_types", ["bbox", "segm"]),
            max_images=getattr(self, "eval_max_images", None),
            inference_topk=getattr(self, "eval_inference_topk", 100),
            max_dets=getattr(self, "eval_max_dets", 100),
        )

        if self.rank == 0:
            self._finalize_eval_result(result)

        if dist.is_available() and dist.is_initialized():
            dist.barrier()

        self.model.train()
        return result.log_dict if self.rank == 0 else {}
