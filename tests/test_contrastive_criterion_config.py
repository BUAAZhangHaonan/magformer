from __future__ import annotations

from types import SimpleNamespace

import torch

from magformer.models.common.criterion import SetCriterion
from magformer.models.magformer.arch import MagFormerArch


class _FixedMatcher:
    def __call__(self, outputs, targets):
        del outputs, targets
        return [(torch.tensor([0, 1]), torch.tensor([0, 1]))]


class _ConstantContrastiveLoss(torch.nn.Module):
    def __init__(self, value: float):
        super().__init__()
        self.value = float(value)
        self.calls = 0

    def forward(self, query_embeddings, targets, indices):
        del targets, indices
        self.calls += 1
        return query_embeddings.new_tensor(self.value)


def _criterion(*, contrastive_enabled: bool, contrastive_weight: float = 0.5) -> SetCriterion:
    criterion = SetCriterion(
        num_classes=1,
        matcher=_FixedMatcher(),
        weight_dict={"loss_ce": 1.0, "loss_mask": 0.0, "loss_dice": 0.0},
        eos_coef=0.1,
        losses=("labels",),
        contrastive_enabled=contrastive_enabled,
        contrastive_weight=contrastive_weight,
        contrastive_temperature=0.11,
    )
    criterion.contrastive_loss_fn = _ConstantContrastiveLoss(4.0)
    return criterion


def _batch():
    outputs = {
        "pred_logits": torch.tensor([[[4.0, -4.0], [3.0, -3.0], [-3.0, 3.0]]]),
        "pred_masks": torch.zeros(1, 3, 4, 4),
        "query_embeddings": torch.randn(1, 3, 8),
    }
    targets = [{"labels": torch.tensor([0, 0]), "masks": torch.ones(2, 4, 4)}]
    return outputs, targets


def test_set_criterion_omits_contrastive_loss_when_disabled():
    criterion = _criterion(contrastive_enabled=False)
    outputs, targets = _batch()

    losses = criterion(outputs, targets)

    assert "contrastive" not in losses
    assert criterion.contrastive_loss_fn.calls == 0


def test_set_criterion_includes_contrastive_loss_when_enabled():
    criterion = _criterion(contrastive_enabled=True, contrastive_weight=0.25)
    outputs, targets = _batch()

    losses = criterion(outputs, targets)

    assert losses["contrastive"] == torch.tensor(1.0)
    assert losses["total_loss"] == losses["loss_ce"] + losses["contrastive"]
    assert criterion.contrastive_loss_fn.calls == 1


def test_runtime_contrastive_config_wires_to_synced_criterion():
    model = object.__new__(MagFormerArch)
    torch.nn.Module.__init__(model)
    model.num_classes = 1
    model_cfg = SimpleNamespace(
        mask_former=SimpleNamespace(
            class_weight=1.0,
            mask_weight=5.0,
            dice_weight=1.0,
            no_object_weight=0.1,
            train_num_points=8,
            oversample_ratio=3.0,
            importance_sample_ratio=0.75,
            balanced_ce=False,
            balanced_ce_min_fg_ratio=0.01,
            deep_supervision=False,
            dec_layers=1,
        )
    )
    runtime_cfg = SimpleNamespace(
        contrastive_enabled=False,
        contrastive_weight=0.25,
        contrastive_temperature=0.11,
    )

    model._sync_criterion_from_config(model_cfg, runtime_cfg=runtime_cfg)

    assert model.criterion.contrastive_enabled is False
    assert model.criterion.contrastive_weight == 0.25
    assert model.criterion.contrastive_loss_fn.temperature == 0.11
