from __future__ import annotations

from pathlib import Path

import pytest
import yaml


def test_msmformer_canonical_recipe_alignment() -> None:
    from baselines.run_msmformer_ecc import build_msmformer_recipe

    recipe = build_msmformer_recipe(register="0831")

    assert recipe.use_depth is True
    assert recipe.use_other_backbone is False
    assert recipe.num_classes == 1
    assert recipe.convs_dim == 64
    assert recipe.mask_dim == 256
    assert recipe.pixel_decoder_name == "SimpleBasePixelDecoder"
    assert recipe.transformer_in_feature == "multi_scale_pixel_decoder"
    assert recipe.transformer_decoder_name == "PretrainedMeanShiftTransformerDecoder"
    assert recipe.use_meanshift_cross_attention is True
    assert recipe.use_meanshift_self_attention is True
    assert recipe.disable_attention_mask is False
    assert recipe.decoder_block_norm is True
    assert recipe.class_weight == pytest.approx(2.0)
    assert recipe.mask_weight == pytest.approx(5.0)
    assert recipe.dice_weight == pytest.approx(5.0)
    assert recipe.dropout == pytest.approx(0.0)
    assert recipe.dec_layers == 7
    assert recipe.object_mask_threshold == pytest.approx(0.8)
    assert recipe.overlap_threshold == pytest.approx(0.8)


def test_msmformer_tracks_config_matches_canonical_recipe() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    cfg_path = repo_root / "configs" / "baselines" / "msmformer_0831_1k_tracks.yaml"
    payload = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

    model = payload["MODEL"]
    head = model["SEM_SEG_HEAD"]
    mask_former = model["MASK_FORMER"]
    solver = payload["SOLVER"]

    assert head["NUM_CLASSES"] == 1
    assert head["CONVS_DIM"] == 64
    assert head["MASK_DIM"] == 256
    assert head["PIXEL_DECODER_NAME"] == "SimpleBasePixelDecoder"
    assert mask_former["TRANSFORMER_IN_FEATURE"] == "multi_scale_pixel_decoder"
    assert mask_former["TRANSFORMER_DECODER_NAME"] == "PretrainedMeanShiftTransformerDecoder"
    assert mask_former["USE_MEANSHIFT_CROSS_ATTENTION"] is True
    assert mask_former["USE_MEANSHIFT_SELF_ATTENTION"] is True
    assert mask_former["DISABLE_MEANSHIFT_ATTENTION_MASK"] is False
    assert mask_former["DECODER_BLOCK_NORM"] is True
    assert mask_former["CLASS_WEIGHT"] == pytest.approx(2.0)
    assert mask_former["MASK_WEIGHT"] == pytest.approx(5.0)
    assert mask_former["DICE_WEIGHT"] == pytest.approx(5.0)
    assert mask_former["DROPOUT"] == pytest.approx(0.0)
    assert mask_former["DEC_LAYERS"] == 7
    assert mask_former["TEST"]["OBJECT_MASK_THRESHOLD"] == pytest.approx(0.8)
    assert mask_former["TEST"]["OVERLAP_THRESHOLD"] == pytest.approx(0.8)
    assert solver["IMS_PER_BATCH"] == 4
    assert solver["BASE_LR"] == pytest.approx(1.0e-4)
    assert solver["WEIGHT_DECAY"] == pytest.approx(0.05)


def test_msmformer_prefers_cached_official_pretrained_weights(tmp_path: Path) -> None:
    from baselines.run_msmformer_ecc import _default_msmformer_pretrained_path, _with_default_model_weights

    local_weights = _default_msmformer_pretrained_path(repo_root=tmp_path)
    local_weights.parent.mkdir(parents=True, exist_ok=True)
    local_weights.write_bytes(b"test-official-msmformer-weights")

    resolved = _with_default_model_weights([], explicit_pretrained=None, repo_root=tmp_path)
    assert resolved == ["MODEL.WEIGHTS", str(local_weights.resolve())]

    custom_weights = tmp_path / "custom_model.pth"
    custom_weights.write_bytes(b"test-custom-msmformer-weights")
    preserved = _with_default_model_weights(
        ["MODEL.WEIGHTS", str(custom_weights)],
        explicit_pretrained=None,
        repo_root=tmp_path,
    )
    assert preserved == ["MODEL.WEIGHTS", str(custom_weights)]

    disabled = _with_default_model_weights([], explicit_pretrained="none", repo_root=tmp_path)
    assert disabled == []
