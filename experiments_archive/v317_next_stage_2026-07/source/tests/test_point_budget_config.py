from __future__ import annotations

import warnings

import pytest
from pydantic import ValidationError

from magformer.config.schema import MaskFormerConfig, ModalityFusionConfig


def test_legacy_train_num_points_populates_both_point_budgets() -> None:
    with pytest.warns(FutureWarning, match="train_num_points is deprecated"):
        config = MaskFormerConfig(train_num_points=37)

    assert config.train_num_points == 37
    assert config.matcher_num_points == 37
    assert config.loss_num_points == 37


def test_explicit_point_budgets_override_legacy_independently() -> None:
    with pytest.warns(FutureWarning, match="train_num_points is deprecated"):
        config = MaskFormerConfig(
            train_num_points=37,
            matcher_num_points=41,
            loss_num_points=43,
        )

    assert config.train_num_points == 37
    assert config.matcher_num_points == 41
    assert config.loss_num_points == 43


def test_split_point_budgets_do_not_emit_legacy_warning() -> None:
    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        config = MaskFormerConfig(matcher_num_points=41, loss_num_points=43)

    assert config.matcher_num_points == 41
    assert config.loss_num_points == 43
    assert not any("train_num_points is deprecated" in str(item.message) for item in records)


@pytest.mark.parametrize("field", ["matcher_num_points", "loss_num_points"])
@pytest.mark.parametrize("value", [0, -1])
def test_split_point_budgets_reject_non_positive_values(field: str, value: int) -> None:
    with pytest.raises(ValidationError):
        MaskFormerConfig(**{field: value})


def test_dccg_confidence_options_are_typed_schema_fields() -> None:
    config = ModalityFusionConfig(
        dccg_conf_hidden=32,
        dccg_use_confidence=False,
    )

    assert config.dccg_conf_hidden == 32
    assert config.dccg_use_confidence is False
    assert config.model_dump()["dccg_use_confidence"] is False
