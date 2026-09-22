# -*- coding: utf-8 -*-
"""Curriculum Scheduler for VC-SUDA pseudo-label quality threshold.

Linearly decays the quality threshold from start_threshold to end_threshold
over warmup_epochs. This gradually allows lower-quality pseudo-labels
into training as the model improves.
"""

from typing import Optional


class CurriculumScheduler:
    """
    Curriculum scheduler for pseudo-label quality threshold.
    
    Starts conservative (high threshold) and gradually lowers to accept
    more pseudo-labels as the student model improves.
    
    Example:
        scheduler = CurriculumScheduler(
            start_threshold=0.7,
            end_threshold=0.3,
            warmup_epochs=15,
        )
        
        for epoch in range(30):
            threshold = scheduler.get_threshold(epoch)
            # threshold goes from 0.7 → 0.3 over first 15 epochs
            # then stays at 0.3
    """
    
    def __init__(
        self,
        start_threshold: float = 0.7,
        end_threshold: float = 0.3,
        warmup_epochs: int = 15,
        total_epochs: Optional[int] = None,
    ):
        """
        Args:
            start_threshold: Initial quality threshold (conservative, keeps only best)
            end_threshold: Final quality threshold (permissive, keeps more)
            warmup_epochs: Epochs over which to decay threshold
            total_epochs: Total training epochs (for logging). Not used in computation.
        """
        assert 0 < end_threshold <= start_threshold <= 1.0, \
            f"Need 0 < end ({end_threshold}) <= start ({start_threshold}) <= 1.0"
        assert warmup_epochs > 0, f"warmup_epochs must be > 0, got {warmup_epochs}"
        
        self.start_threshold = start_threshold
        self.end_threshold = end_threshold
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
    
    def get_threshold(self, epoch: int) -> float:
        """
        Get quality threshold for a given epoch.
        
        Linear decay from start_threshold to end_threshold over warmup_epochs.
        After warmup, returns end_threshold.
        
        Args:
            epoch: Current epoch (0-indexed)
            
        Returns:
            Quality threshold in [end_threshold, start_threshold]
        """
        if epoch >= self.warmup_epochs:
            return self.end_threshold
        
        # Linear interpolation
        progress = epoch / self.warmup_epochs
        threshold = self.start_threshold + (
            self.end_threshold - self.start_threshold
        ) * progress
        
        return threshold
    
    def get_unsupervised_weight(
        self,
        epoch: int,
        max_weight: float = 1.0,
        warmup_epochs: int = 10,
    ) -> float:
        """
        Get unsupervised loss weight with quadratic ramp-up.
        
        Starts at 0 and ramps to max_weight over warmup_epochs.
        This prevents noisy pseudo-labels from dominating early training.
        
        Args:
            epoch: Current epoch (0-indexed)
            max_weight: Maximum weight after warmup
            warmup_epochs: Epochs to ramp up over
            
        Returns:
            Loss weight in [0, max_weight]
        """
        if epoch >= warmup_epochs:
            return max_weight
        
        # Quadratic ramp: starts slow, accelerates
        progress = epoch / warmup_epochs
        return max_weight * (progress ** 2)
    
    def state_dict(self) -> dict:
        """Return scheduler state for checkpointing."""
        return {
            "start_threshold": self.start_threshold,
            "end_threshold": self.end_threshold,
            "warmup_epochs": self.warmup_epochs,
            "total_epochs": self.total_epochs,
        }
    
    def load_state_dict(self, state: dict) -> None:
        """Load scheduler state from checkpoint."""
        self.start_threshold = state.get("start_threshold", self.start_threshold)
        self.end_threshold = state.get("end_threshold", self.end_threshold)
        self.warmup_epochs = state.get("warmup_epochs", self.warmup_epochs)
        self.total_epochs = state.get("total_epochs", self.total_epochs)
    
    def __repr__(self) -> str:
        return (
            f"CurriculumScheduler("
            f"start={self.start_threshold}, "
            f"end={self.end_threshold}, "
            f"warmup={self.warmup_epochs} epochs)"
        )
