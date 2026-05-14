# -*- coding: utf-8 -*-
"""
VC-SUDA Configuration Schema

Visibility-Constrained Semi-supervised Unsupervised Domain Adaptation
configuration for sim2real transfer.
"""

from typing import Literal, Optional
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


class VCSUDAConfig(BaseModel):
    """VC-SUDA top-level configuration."""
    enabled: bool = Field(default=False, description="Enable VC-SUDA training entrypoint behavior")
    stage: Literal["A", "B", "C", "D", "E"] = Field(default="A", description="Training stage: A, B, C, D, or E")
    ema_teacher: EMATeacherConfig = Field(default_factory=EMATeacherConfig)
    pseudo_label: PseudoLabelConfig = Field(default_factory=PseudoLabelConfig)
    domain_adaptation: DomainAdaptationConfig = Field(default_factory=DomainAdaptationConfig)
    curriculum: CurriculumConfig = Field(default_factory=CurriculumConfig)
    unsupervised_weight: float = Field(default=1.0, description="Weight for unsupervised (pseudo-label) loss")
    target_labeled_weight: float = Field(
        default=1.0,
        ge=0,
        description="Weight for target labeled supervised loss",
    )
    unsupervised_warmup_epochs: int = Field(
        default=10, description="Epochs to ramp up unsupervised weight from 0 to 1"
    )
    # Data split paths (relative to dataset_root)
    source_ann: str = Field(
        default="annotations/instances_source.json",
        description="Source (synthetic) annotation file",
    )
    target_labeled_ann: Optional[str] = Field(
        default=None, description="Target labeled annotation file (Stage B+)"
    )
    target_unlabeled_ann: Optional[str] = Field(
        default=None, description="Target unlabeled annotation file (Stage C+)"
    )

    @model_validator(mode="after")
    def validate_stage_data_requirements(self):
        """Fail fast when a VC-SUDA stage is missing required data splits."""
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
