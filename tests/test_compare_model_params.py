from scripts.compare_model_params import classify_parameter_group


def test_magformer_decoder_parameters_are_not_other():
    # Regression: MagFormer decoder.* used to be miscounted as "other".
    assert classify_parameter_group("decoder.transformer.encoder.layers.0.linear1.weight") == "pixel_decoder"
    assert classify_parameter_group("decoder.decoder.layers.0.self_attn.in_proj_weight") == "transformer_decoder"


def test_mask2former_predictor_is_transformer_decoder_group():
    assert classify_parameter_group("sem_seg_head.predictor.class_embed.weight") == "transformer_decoder"


def test_mgm_and_backbones_are_grouped_correctly():
    assert classify_parameter_group("rgb_backbone.layers.0.weight") == "rgb_backbone"
    assert classify_parameter_group("depth_backbone.stages.0.weight") == "depth_backbone"
    assert classify_parameter_group("fusion.conf_pred.head.0.weight") == "fusion/mgm"
