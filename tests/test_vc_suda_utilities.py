import pytest
import torch
import torch.nn as nn

from magformer.models.common.ema_teacher import EMATeacherWrapper
from magformer.models.common.pseudo_label_scorer import PseudoLabelScorer
from magformer.models.magformer.domain_losses import (
    BoundaryConsistencyLoss,
    PrototypeAlignmentLoss,
    UncertaintyWeighting,
)
from magformer.models.magformer.vc_suda_criterion import VCSUDACriterion


class _BufferModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([1.0]))
        self.register_buffer("running", torch.tensor([2.0]))
        self.register_buffer("seen", torch.tensor([3], dtype=torch.long))

    def forward_inference_decoder_outputs(self, **kwargs):
        return {}


def test_ema_updates_float_buffers_and_copies_integer_buffers():
    student = _BufferModel()
    teacher = EMATeacherWrapper(student, momentum=0.5, warmup_steps=0)

    student.running.fill_(10.0)
    student.seen.fill_(7)

    teacher.update_ema(student, step=0)

    assert torch.allclose(teacher.teacher.running, torch.tensor([6.0]))
    assert teacher.teacher.seen.dtype == torch.long
    assert teacher.teacher.seen.item() == 7


def test_prototype_alignment_loss_is_differentiable_from_current_inputs():
    loss_fn = PrototypeAlignmentLoss(num_classes=1, feature_dim=2, ema_rate=0.9)
    source_features = torch.tensor(
        [[[[2.0, 1.0], [2.0, 1.0]], [[0.0, 0.0], [0.0, 0.0]]]],
        requires_grad=True,
    )
    target_features = torch.tensor(
        [[[[0.0, 0.0], [0.0, 0.0]], [[1.0, 2.0], [1.0, 2.0]]]],
        requires_grad=True,
    )
    source_masks = torch.tensor([[[[0.9, 0.8], [0.1, 0.2]]]], requires_grad=True)
    target_masks = torch.tensor([[[[0.2, 0.1], [0.8, 0.9]]]], requires_grad=True)

    loss = loss_fn(source_features, target_features, source_masks, target_masks)
    assert loss.requires_grad

    loss.backward()

    assert source_features.grad.abs().sum() > 0
    assert target_features.grad.abs().sum() > 0
    assert source_masks.grad.abs().sum() > 0
    assert target_masks.grad.abs().sum() > 0


def test_boundary_consistency_loss_backpropagates_to_soft_masks():
    loss_fn = BoundaryConsistencyLoss(boundary_confidence_threshold=0.2)
    pred_masks = torch.tensor(
        [
            [
                [
                    [0.1, 0.2, 0.8, 0.9],
                    [0.1, 0.2, 0.8, 0.9],
                    [0.1, 0.2, 0.8, 0.9],
                    [0.1, 0.2, 0.8, 0.9],
                ]
            ]
        ],
        requires_grad=True,
    )
    depth = torch.tensor(
        [[[[0.0, 0.0, 5.0, 5.0], [0.0, 0.0, 5.0, 5.0], [0.0, 0.0, 5.0, 5.0], [0.0, 0.0, 5.0, 5.0]]]]
    )

    loss = loss_fn(pred_masks, depth)
    assert loss.requires_grad

    loss.backward()

    assert pred_masks.grad.abs().sum() > 0


def test_uncertainty_weighting_returns_tensor_log_values():
    weighting = UncertaintyWeighting(num_tasks=2)

    total, metrics = weighting(torch.tensor(2.0), torch.tensor(4.0))

    assert isinstance(total, torch.Tensor)
    assert total.requires_grad
    assert metrics
    assert all(isinstance(value, torch.Tensor) for value in metrics.values())
    assert all(isinstance(value.item(), float) for value in metrics.values())


def test_pseudo_label_scorer_preserves_high_confidence_small_objects():
    scorer = PseudoLabelScorer(max_instances=10)
    logits = torch.tensor([[[8.0, -8.0]]])
    masks = torch.full((1, 1, 16, 16), -8.0)
    masks[:, :, 7:9, 7:9] = 8.0
    depth = torch.zeros(1, 1, 16, 16)
    depth[:, :, 7:9, 7:9] = 10.0

    result = scorer.score({"pred_logits": logits, "pred_masks": masks}, depth)[0]

    assert result["scores"].numel() == 1
    assert result["scores"].item() > 0.5


def _target_mask():
    mask = torch.zeros(1, 4, 4)
    mask[:, 1:3, 1:3] = 1.0
    return mask


def _student_outputs(unmatched_foreground_logit: float, matched_foreground_logit: float = 8.0):
    logits = torch.tensor(
        [
            [
                [matched_foreground_logit, -matched_foreground_logit],
                [unmatched_foreground_logit, -unmatched_foreground_logit],
                [unmatched_foreground_logit, -unmatched_foreground_logit],
            ]
        ]
    )
    masks = torch.full((1, 3, 4, 4), -8.0)
    masks[:, 0, 1:3, 1:3] = 8.0
    return {"pred_logits": logits, "pred_masks": masks}


def test_empty_pseudo_image_does_not_create_all_background_ce():
    criterion = VCSUDACriterion(supervised_criterion=None)
    outputs = _student_outputs(8.0)
    outputs["pred_logits"].requires_grad_()
    outputs["pred_masks"].requires_grad_()
    pseudo_targets = [
        {
            "labels": torch.empty(0, dtype=torch.long),
            "masks": torch.empty(0, 4, 4),
            "quality_scores": torch.empty(0),
        }
    ]

    losses = criterion.pseudo_label_loss(outputs, pseudo_targets)

    assert losses["pseudo_loss_ce"].item() == 0.0
    assert losses["pseudo_loss_mask"].item() == 0.0
    assert losses["pseudo_loss_dice"].item() == 0.0
    assert losses["pseudo_total"].item() == 0.0
    losses["pseudo_total"].backward()
    assert outputs["pred_logits"].grad is not None


def test_pseudo_loss_ignores_unmatched_foreground_queries_for_ce():
    criterion = VCSUDACriterion(supervised_criterion=None)
    pseudo_targets = [
        {
            "labels": torch.tensor([0]),
            "masks": _target_mask(),
            "quality_scores": torch.tensor([1.0]),
        }
    ]

    high_unmatched = criterion.pseudo_label_loss(_student_outputs(8.0), pseudo_targets)
    low_unmatched = criterion.pseudo_label_loss(_student_outputs(-8.0), pseudo_targets)

    assert high_unmatched["pseudo_loss_ce"].item() == pytest.approx(
        low_unmatched["pseudo_loss_ce"].item(), abs=1e-6
    )


def test_pseudo_loss_ce_increases_when_matched_query_predicts_wrong_class():
    criterion = VCSUDACriterion(supervised_criterion=None)
    pseudo_targets = [
        {
            "labels": torch.tensor([0]),
            "masks": _target_mask(),
            "quality_scores": torch.tensor([1.0]),
        }
    ]

    good = criterion.pseudo_label_loss(
        _student_outputs(-8.0, matched_foreground_logit=8.0), pseudo_targets
    )
    wrong = criterion.pseudo_label_loss(
        _student_outputs(-8.0, matched_foreground_logit=-8.0), pseudo_targets
    )

    assert wrong["pseudo_loss_ce"] > good["pseudo_loss_ce"] + 10.0


def test_ignored_unmatched_queries_do_not_poison_pseudo_loss_with_nan():
    criterion = VCSUDACriterion(supervised_criterion=None)
    outputs = _student_outputs(-8.0, matched_foreground_logit=8.0)
    outputs["pred_logits"][:, 1:] = float("nan")
    outputs["pred_masks"][:, 1:] = float("nan")
    pseudo_targets = [
        {
            "labels": torch.tensor([0]),
            "masks": _target_mask(),
            "quality_scores": torch.tensor([1.0]),
        }
    ]

    losses = criterion.pseudo_label_loss(outputs, pseudo_targets)

    assert torch.isfinite(losses["pseudo_loss_ce"])
    assert torch.isfinite(losses["pseudo_loss_mask"])
    assert torch.isfinite(losses["pseudo_loss_dice"])
    assert torch.isfinite(losses["pseudo_total"])


def test_pseudo_quality_scores_lower_matched_loss_contribution():
    criterion = VCSUDACriterion(
        supervised_criterion=None,
        pseudo_weight_ce=1.0,
        pseudo_weight_mask=1.0,
        pseudo_weight_dice=1.0,
    )
    high_quality = [
        {
            "labels": torch.tensor([0]),
            "masks": _target_mask(),
            "quality_scores": torch.tensor([1.0]),
        }
    ]
    low_quality = [
        {
            "labels": torch.tensor([0]),
            "masks": _target_mask(),
            "quality_scores": torch.tensor([0.25]),
        }
    ]
    outputs = _student_outputs(-8.0, matched_foreground_logit=-2.0)

    high = criterion.pseudo_label_loss(outputs, high_quality)
    low = criterion.pseudo_label_loss(outputs, low_quality)

    assert low["pseudo_loss_ce"] < high["pseudo_loss_ce"]
    assert low["pseudo_loss_mask"] < high["pseudo_loss_mask"]
    assert low["pseudo_loss_dice"] < high["pseudo_loss_dice"]
    assert low["pseudo_total"] < high["pseudo_total"]


def test_pseudo_mask_loss_uses_soft_teacher_masks_without_binarizing():
    criterion = VCSUDACriterion(
        supervised_criterion=None,
        pseudo_weight_ce=0.0,
        pseudo_weight_mask=1.0,
        pseudo_weight_dice=0.0,
    )
    logits = torch.tensor([[[8.0, -8.0]]])
    masks = torch.full((1, 1, 4, 4), 2.0)
    soft_teacher_mask = torch.full((1, 4, 4), 0.25)
    pseudo_targets = [
        {
            "labels": torch.tensor([0]),
            "masks": soft_teacher_mask,
            "quality_scores": torch.tensor([1.0]),
        }
    ]

    losses = criterion.pseudo_label_loss(
        {"pred_logits": logits, "pred_masks": masks}, pseudo_targets
    )

    expected_soft = torch.nn.functional.binary_cross_entropy_with_logits(
        masks[0, 0].flatten(), soft_teacher_mask[0].flatten(), reduction="mean"
    )
    expected_hard = torch.nn.functional.binary_cross_entropy_with_logits(
        masks[0, 0].flatten(), torch.zeros(16), reduction="mean"
    )
    assert losses["pseudo_loss_mask"].item() == pytest.approx(expected_soft.item())
    assert losses["pseudo_loss_mask"].item() != pytest.approx(expected_hard.item())


def test_pseudo_loss_supports_aux_outputs():
    criterion = VCSUDACriterion(supervised_criterion=None)
    pseudo_targets = [
        {
            "labels": torch.tensor([0]),
            "masks": _target_mask(),
            "quality_scores": torch.tensor([1.0]),
        }
    ]
    outputs = _student_outputs(8.0)
    outputs["aux_outputs"] = [_student_outputs(8.0)]

    losses = criterion.pseudo_label_loss(outputs, pseudo_targets)

    assert "pseudo_loss_ce_0" in losses
    assert "pseudo_loss_mask_0" in losses
    assert "pseudo_loss_dice_0" in losses
    assert losses["pseudo_total"] >= losses["pseudo_loss_ce_0"]


def test_pseudo_loss_default_does_not_return_diagnostics_key():
    criterion = VCSUDACriterion(supervised_criterion=None)
    pseudo_targets = [
        {
            "labels": torch.tensor([0]),
            "masks": _target_mask(),
            "quality_scores": torch.tensor([1.0]),
        }
    ]

    losses = criterion.pseudo_label_loss(_student_outputs(8.0), pseudo_targets)

    assert "pseudo_diagnostics" not in losses
    assert "pseudo_unmatched_negative_loss" not in losses
    assert "pseudo_unmatched_high_score_count" not in losses
    assert "pseudo_exterior_ring_loss" not in losses


def test_pseudo_exterior_ring_default_off_does_not_change_loss():
    pseudo_targets = [
        {
            "labels": torch.tensor([0]),
            "masks": _target_mask(),
            "quality_scores": torch.tensor([1.0]),
        }
    ]
    outputs = _student_outputs(8.0)
    baseline = VCSUDACriterion(supervised_criterion=None).pseudo_label_loss(
        outputs, pseudo_targets
    )
    disabled = VCSUDACriterion(
        supervised_criterion=None,
        pseudo_exterior_ring_enabled=False,
        pseudo_exterior_ring_weight=1.0,
        pseudo_exterior_ring_radius=1,
    ).pseudo_label_loss(outputs, pseudo_targets)

    assert "pseudo_exterior_ring_loss" not in disabled
    assert disabled["pseudo_total"].item() == pytest.approx(baseline["pseudo_total"].item())


def test_pseudo_exterior_ring_enabled_penalizes_matched_ring_probability():
    criterion = VCSUDACriterion(
        supervised_criterion=None,
        pseudo_weight_ce=0.0,
        pseudo_weight_mask=0.0,
        pseudo_weight_dice=0.0,
        pseudo_exterior_ring_enabled=True,
        pseudo_exterior_ring_weight=1.0,
        pseudo_exterior_ring_radius=1,
    )
    logits = torch.tensor([[[8.0, -8.0]]])
    masks = torch.full((1, 1, 5, 5), -8.0)
    masks[:, 0, 1:4, 1:4] = 8.0
    pseudo_targets = [
        {
            "labels": torch.tensor([0]),
            "masks": torch.zeros(1, 5, 5),
            "quality_scores": torch.tensor([1.0]),
        }
    ]
    pseudo_targets[0]["masks"][:, 2, 2] = 1.0

    losses = criterion.pseudo_label_loss({"pred_logits": logits, "pred_masks": masks}, pseudo_targets)

    assert losses["pseudo_exterior_ring_loss"].item() > 0.99
    assert losses["pseudo_total"].item() == pytest.approx(
        losses["pseudo_exterior_ring_loss"].item()
    )
    assert losses["pseudo_exterior_ring_matched_count"].item() == pytest.approx(1.0)
    assert losses["pseudo_exterior_ring_valid_count"].item() == pytest.approx(1.0)
    assert losses["pseudo_exterior_ring_pixel_count"].item() == pytest.approx(8.0)
    assert losses["pseudo_exterior_ring_prob"].item() > 0.99


def test_pseudo_exterior_ring_loss_only_uses_exterior_ring_pixels():
    criterion = VCSUDACriterion(
        supervised_criterion=None,
        pseudo_weight_ce=0.0,
        pseudo_weight_mask=0.0,
        pseudo_weight_dice=0.0,
        pseudo_exterior_ring_enabled=True,
        pseudo_exterior_ring_weight=1.0,
        pseudo_exterior_ring_radius=1,
    )
    logits = torch.tensor([[[8.0, -8.0]]])
    pseudo_targets = [
        {
            "labels": torch.tensor([0]),
            "masks": torch.zeros(1, 5, 5),
            "quality_scores": torch.tensor([1.0]),
        }
    ]
    pseudo_targets[0]["masks"][:, 2, 2] = 1.0
    low_interior = torch.full((1, 1, 5, 5), -8.0)
    high_interior = low_interior.clone()
    high_interior[:, 0, 2, 2] = 8.0

    low_loss = criterion.pseudo_label_loss(
        {"pred_logits": logits, "pred_masks": low_interior}, pseudo_targets
    )
    high_loss = criterion.pseudo_label_loss(
        {"pred_logits": logits, "pred_masks": high_interior}, pseudo_targets
    )

    assert high_loss["pseudo_exterior_ring_loss"].item() == pytest.approx(
        low_loss["pseudo_exterior_ring_loss"].item(), abs=1e-6
    )


def test_pseudo_exterior_ring_zero_ring_is_skipped_and_counted():
    criterion = VCSUDACriterion(
        supervised_criterion=None,
        pseudo_weight_ce=0.0,
        pseudo_weight_mask=0.0,
        pseudo_weight_dice=0.0,
        pseudo_exterior_ring_enabled=True,
        pseudo_exterior_ring_weight=1.0,
        pseudo_exterior_ring_radius=0,
    )
    pseudo_targets = [
        {
            "labels": torch.tensor([0]),
            "masks": _target_mask(),
            "quality_scores": torch.tensor([1.0]),
        }
    ]

    losses = criterion.pseudo_label_loss(_student_outputs(8.0), pseudo_targets)

    assert losses["pseudo_exterior_ring_loss"].item() == pytest.approx(0.0)
    assert losses["pseudo_exterior_ring_matched_count"].item() == pytest.approx(1.0)
    assert losses["pseudo_exterior_ring_valid_count"].item() == pytest.approx(0.0)
    assert losses["pseudo_exterior_ring_zero_ring_count"].item() == pytest.approx(1.0)
    assert losses["pseudo_exterior_ring_pixel_count"].item() == pytest.approx(0.0)
    assert losses["pseudo_exterior_ring_prob"].item() == pytest.approx(0.0)


def test_pseudo_unmatched_negative_enabled_penalizes_only_high_score_unmatched_query():
    criterion = VCSUDACriterion(
        supervised_criterion=None,
        pseudo_weight_ce=0.0,
        pseudo_weight_mask=0.0,
        pseudo_weight_dice=0.0,
        pseudo_unmatched_negative_enabled=True,
        pseudo_unmatched_negative_weight=1.0,
        pseudo_unmatched_negative_score_thresh=0.9,
    )
    outputs = _student_outputs(-8.0)
    outputs["pred_logits"] = torch.tensor(
        [
            [
                [8.0, -8.0],
                [6.0, -6.0],
                [1.0, -1.0],
            ]
        ]
    )
    pseudo_targets = [
        {
            "labels": torch.tensor([0]),
            "masks": _target_mask(),
            "quality_scores": torch.tensor([1.0]),
        }
    ]

    losses = criterion.pseudo_label_loss(outputs, pseudo_targets)

    expected = torch.nn.functional.cross_entropy(
        outputs["pred_logits"][0, 1].float().unsqueeze(0),
        torch.tensor([1]),
    )
    assert losses["pseudo_unmatched_high_score_count"].item() == pytest.approx(1.0)
    assert losses["pseudo_unmatched_negative_loss"].item() == pytest.approx(expected.item())
    assert losses["pseudo_total"].item() == pytest.approx(expected.item())


def test_pseudo_unmatched_negative_does_not_penalize_matched_high_score_query():
    criterion = VCSUDACriterion(
        supervised_criterion=None,
        pseudo_weight_ce=0.0,
        pseudo_weight_mask=0.0,
        pseudo_weight_dice=0.0,
        pseudo_unmatched_negative_enabled=True,
        pseudo_unmatched_negative_weight=1.0,
        pseudo_unmatched_negative_score_thresh=0.9,
    )
    outputs = _student_outputs(-8.0, matched_foreground_logit=8.0)
    pseudo_targets = [
        {
            "labels": torch.tensor([0]),
            "masks": _target_mask(),
            "quality_scores": torch.tensor([1.0]),
        }
    ]

    losses = criterion.pseudo_label_loss(outputs, pseudo_targets)

    assert losses["pseudo_unmatched_high_score_count"].item() == pytest.approx(0.0)
    assert losses["pseudo_unmatched_negative_loss"].item() == pytest.approx(0.0)
    assert losses["pseudo_total"].item() == pytest.approx(0.0)


def test_pseudo_diagnostics_counts_unmatched_high_score_queries_and_iou():
    criterion = VCSUDACriterion(supervised_criterion=None)
    outputs = _student_outputs(-8.0)
    outputs["pred_logits"] = torch.tensor(
        [
            [
                [8.0, -8.0],
                [6.0, -6.0],
                [-6.0, 6.0],
            ]
        ]
    )
    outputs["pred_masks"][:, 1] = torch.full((4, 4), -8.0)
    outputs["pred_masks"][:, 1, 1:3, 2:4] = 8.0
    pseudo_targets = [
        {
            "labels": torch.tensor([0]),
            "masks": _target_mask(),
            "quality_scores": torch.tensor([0.95]),
        }
    ]

    diagnostics = criterion.pseudo_label_diagnostics(
        outputs,
        pseudo_targets,
        high_score_thresholds=(0.7, 0.9),
    )

    assert diagnostics["image_count"] == 1
    assert diagnostics["query_count"] == 3
    assert diagnostics["kept_pseudo_count"] == 1
    assert diagnostics["matched_query_count"] == 1
    assert diagnostics["unmatched_query_count"] == 2
    assert diagnostics["unmatched_high_score_counts"]["0.700"] == 1
    assert diagnostics["unmatched_high_score_counts"]["0.900"] == 1
    assert diagnostics["unmatched_high_score_score_distribution"]["0.700"]["count"] == 1
    assert diagnostics["unmatched_high_score_score_distribution"]["0.700"]["min"] > 0.99
    assert diagnostics["unmatched_high_score_max_iou_distribution"]["0.700"]["count"] == 1
    assert diagnostics["unmatched_high_score_max_iou_distribution"]["0.700"]["max"] > 0.3


def test_exterior_ring_mask_uses_dilation_without_interior_pixels():
    mask = torch.zeros(1, 5, 5)
    mask[:, 2, 2] = 1.0

    ring, interior, far = VCSUDACriterion._exterior_ring_masks(mask, radius=1)

    assert interior.sum().item() == 1
    assert ring.sum().item() == 8
    assert ring[0, 2, 2].item() is False
    assert far.sum().item() == 16


def test_pseudo_diagnostics_records_matched_exterior_ring_stats():
    criterion = VCSUDACriterion(supervised_criterion=None)
    logits = torch.tensor([[[8.0, -8.0]]])
    masks = torch.full((1, 1, 5, 5), -8.0)
    masks[:, 0, 1:4, 1:4] = 8.0
    pseudo_targets = [
        {
            "labels": torch.tensor([0]),
            "masks": torch.zeros(1, 5, 5),
            "quality_scores": torch.tensor([1.0]),
        }
    ]
    pseudo_targets[0]["masks"][:, 2, 2] = 1.0

    diagnostics = criterion.pseudo_label_diagnostics(
        {"pred_logits": logits, "pred_masks": masks},
        pseudo_targets,
        exterior_ring_radius=1,
    )
    ring = diagnostics["matched_exterior_ring"]

    assert ring["radius"] == 1
    assert ring["matched_count"] == 1
    assert ring["gt_density_bucket_supported"] is False
    assert ring["exterior_ring_prob_distribution"]["count"] == 1
    assert ring["exterior_ring_prob_distribution"]["mean"] > 0.99
    assert ring["interior_prob_distribution"]["mean"] > 0.99
    assert ring["background_far_prob_distribution"]["count"] == 1
    assert ring["pred_target_area_ratio_distribution"]["mean"] == pytest.approx(9.0)
    assert ring["ring_pixel_count_distribution"]["mean"] == pytest.approx(8.0)
