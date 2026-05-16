# -*- coding: utf-8 -*-
"""
VC-SUDA Configuration Schema

Visibility-Constrained Semi-supervised Unsupervised Domain Adaptation
configuration for sim2real transfer.
"""

from typing import List, Literal, Optional
from pydantic import BaseModel, ConfigDict, Field, model_validator


class EMATeacherConfig(BaseModel):
    """EMA teacher configuration."""
    enabled: bool = Field(default=False, description="Enable EMA teacher-student training")
    ema_momentum: float = Field(default=0.999, description="EMA decay rate (higher = slower update)")
    warmup_steps: int = Field(default=500, description="Steps to ramp up EMA momentum from 0.99 to target")
    model_config = ConfigDict(extra="forbid")


class PseudoLabelConfig(BaseModel):
    """Pseudo-label generation and filtering configuration."""
    quality_threshold: float = Field(default=0.5, description="Minimum quality score for pseudo-labels")
    use_curriculum: bool = Field(default=True, description="Enable curriculum learning for threshold decay")
    max_instances: int = Field(default=100, description="Max pseudo-label instances per image")
    model_config = ConfigDict(extra="forbid")


class DomainAdaptationConfig(BaseModel):
    """Domain adaptation loss configuration."""
    prototype_weight: float = Field(default=1.0, description="Weight for prototype alignment loss")
    boundary_weight: float = Field(default=0.5, description="Weight for boundary consistency loss")
    modality_dropout_weight: float = Field(default=0.3, description="Weight for modality dropout consistency loss")
    num_prototypes: int = Field(default=64, description="Number of prototype vectors per class")
    boundary_confidence_threshold: float = Field(
        default=0.5, description="Depth gradient threshold for boundary gating"
    )
    modality_dropout_prob: float = Field(default=0.3, description="Probability of dropping depth modality")
    use_uncertainty_weighting: bool = Field(default=False, description="Use learnable UW instead of fixed weights")
    model_config = ConfigDict(extra="forbid")


class CurriculumConfig(BaseModel):
    """Curriculum learning for pseudo-label quality threshold."""
    start_threshold: float = Field(default=0.7, description="Initial quality threshold (conservative)")
    end_threshold: float = Field(default=0.3, description="Final quality threshold (permissive)")
    warmup_epochs: int = Field(default=15, description="Epochs to decay threshold over")
    model_config = ConfigDict(extra="forbid")


class PseudoExteriorRingLossConfig(BaseModel):
    """Matched pseudo-positive exterior ring loss configuration."""
    enabled: bool = Field(
        default=False,
        strict=True,
        description="Enable exterior ring probability penalty for matched pseudo positives",
    )
    weight: float = Field(
        default=0.0,
        ge=0.0,
        strict=True,
        description="Weight for matched pseudo-positive exterior ring probability penalty",
    )
    radius: int = Field(
        default=2,
        ge=0,
        strict=True,
        description="Dilation radius used to form the exterior ring around pseudo masks",
    )
    model_config = ConfigDict(extra="forbid")


class TargetUnlabeledSamplingConfig(BaseModel):
    """Prediction-only target-unlabeled sampling configuration."""
    enabled: bool = Field(
        default=False,
        strict=True,
        description="Enable target_unlabeled repeat sampling from prediction-only stats.",
    )
    stats_path: Optional[str] = Field(
        default=None,
        description="Path to target_unlabeled sampling stats JSON built from predictions only.",
    )
    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def validate_enabled_stats_path(self):
        if self.enabled and not self.stats_path:
            raise ValueError("target_unlabeled_sampling.stats_path is required when enabled.")
        return self


class SourceDatasetConfig(BaseModel):
    """One labeled source dataset used by VC-SUDA source mixing."""
    name: str = Field(description="Stable source dataset name for metadata.")
    root: str = Field(description="Dataset root for this source dataset.")
    ann: str = Field(description="Annotation file for this source dataset.")
    split: str = Field(default="train", description="Dataset split for this source dataset.")
    weight: int = Field(default=1, ge=1, description="Deterministic weighted round-robin weight.")
    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def validate_required_strings(self):
        for field_name in ("name", "root", "ann", "split"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"source_datasets.{field_name} must not be empty.")
        return self


class VCSUDAConfig(BaseModel):
    """VC-SUDA top-level configuration."""
    enabled: bool = Field(default=False, description="Enable VC-SUDA training entrypoint behavior")
    stage: Literal["A", "B", "C", "D", "E"] = Field(default="A", description="Training stage: A, B, C, D, or E")
    ema_teacher: EMATeacherConfig = Field(default_factory=EMATeacherConfig)
    pseudo_label: PseudoLabelConfig = Field(default_factory=PseudoLabelConfig)
    domain_adaptation: DomainAdaptationConfig = Field(default_factory=DomainAdaptationConfig)
    curriculum: CurriculumConfig = Field(default_factory=CurriculumConfig)
    unsupervised_weight: float = Field(default=1.0, description="Weight for unsupervised (pseudo-label) loss")
    pseudo_unmatched_negative_enabled: bool = Field(
        default=False,
        strict=True,
        description="Enable high-score unmatched-query background CE in target-unlabeled pseudo branch",
    )
    pseudo_unmatched_negative_weight: float = Field(
        default=0.05,
        ge=0.0,
        strict=True,
        description="Weight for pseudo unmatched high-score background CE",
    )
    pseudo_unmatched_negative_score_thresh: float = Field(
        default=0.9,
        ge=0.0,
        le=1.0,
        strict=True,
        description="Foreground confidence threshold for pseudo unmatched background CE",
    )
    pseudo_exterior_ring_loss: PseudoExteriorRingLossConfig = Field(
        default_factory=PseudoExteriorRingLossConfig,
        description="Matched pseudo-positive exterior ring loss settings",
    )
    target_unlabeled_sampling: TargetUnlabeledSamplingConfig = Field(
        default_factory=TargetUnlabeledSamplingConfig,
        description="Prediction-only repeat sampling for target_unlabeled images.",
    )
    target_labeled_weight: float = Field(
        default=1.0,
        ge=0,
        description="Weight for target labeled supervised loss",
    )
    unsupervised_warmup_epochs: int = Field(
        default=10, description="Epochs to ramp up unsupervised weight from 0 to 1"
    )
    # Data split paths (relative to dataset_root)
    source_root: Optional[str] = Field(
        default=None,
        description="Optional source dataset root override; target splits still use data.dataset_root",
    )
    source_ann: str = Field(
        default="annotations/instances_source.json",
        description="Source (synthetic) annotation file",
    )
    source_split: str = Field(default="train", description="Legacy single-source split")
    source_datasets: Optional[List[SourceDatasetConfig]] = Field(
        default=None,
        description="Optional labeled source datasets for deterministic weighted mixing.",
    )
    target_labeled_split: str = Field(default="train", description="Target labeled split")
    target_unlabeled_split: str = Field(default="train", description="Target unlabeled split")
    target_labeled_ann: Optional[str] = Field(
        default=None, description="Target labeled annotation file (Stage B+)"
    )
    target_unlabeled_ann: Optional[str] = Field(
        default=None, description="Target unlabeled annotation file (Stage C+)"
    )

    @model_validator(mode="after")
    def validate_stage_data_requirements(self):
        """Fail fast when a VC-SUDA stage is missing required data splits."""
        if self.source_datasets is not None:
            if len(self.source_datasets) == 0:
                raise ValueError("source_datasets must not be empty when set.")
            names = [source.name for source in self.source_datasets]
            duplicate_names = sorted({name for name in names if names.count(name) > 1})
            if duplicate_names:
                raise ValueError(f"duplicate source_datasets name: {duplicate_names[0]}")
        if self.stage in {"B", "C", "D", "E"} and not self.target_labeled_ann:
            raise ValueError(
                f"VC-SUDA stage {self.stage} requires target_labeled_ann."
            )
        if self.stage in {"C", "D", "E"} and not self.target_unlabeled_ann:
            raise ValueError(
                f"VC-SUDA stage {self.stage} requires target_unlabeled_ann."
            )
        return self

    model_config = ConfigDict(extra="forbid")
