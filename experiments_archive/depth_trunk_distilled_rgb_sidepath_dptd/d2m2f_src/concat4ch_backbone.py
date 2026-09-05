"""Concat-4ch Swin backbone: replicate the historical anchor recipe.

D2SwinTransformer with patch_embed in_channels=4. The 4th-column patch-embed
weights come from a patched checkpoint (build_init_weights.py zero-pads the
3ch COCO weights); everything else is unchanged.
"""
import torch.nn as nn

from detectron2.modeling import BACKBONE_REGISTRY

from mask2former.modeling.backbone.swin import D2SwinTransformer, PatchEmbed


@BACKBONE_REGISTRY.register()
class D2SwinTransformer4CH(D2SwinTransformer):
    def __init__(self, cfg, input_shape):
        super().__init__(cfg, input_shape)
        # rebuild patch embed with 4 input channels
        old = self.patch_embed
        self.patch_embed = PatchEmbed(
            patch_size=cfg.MODEL.SWIN.PATCH_SIZE,
            in_chans=4,
            embed_dim=cfg.MODEL.SWIN.EMBED_DIM,
            norm_layer=nn.LayerNorm if cfg.MODEL.SWIN.PATCH_NORM else None,
        )
        del old
