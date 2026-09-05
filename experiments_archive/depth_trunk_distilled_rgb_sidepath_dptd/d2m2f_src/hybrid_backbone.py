"""Hybrid fusion Swin backbone: concat-4ch stem + stage-wise FiLM re-injection.

Rationale: concat4ch injects depth only once (patch embed); its gradient path
is open from step 0, which is why it learns fast. SGF injects depth at every
stage but through zero-init gates only, which opens too slowly at screening
budgets. The hybrid keeps both: the 4ch patch embed (identical to concat at
init, loads the same P4CH checkpoint) AND the SGF depth tower + per-stage
zero-init FiLM. At step 0 the model is EXACTLY concat4ch (FiLM contribution
is 0); training can open the re-injection paths where depth helps.

Parameter budget over concat4ch: tower ~0.10M + FiLM convs ~0.26M = ~0.36M.
"""
import torch.nn as nn
import torch.nn.functional as F

from detectron2.modeling import BACKBONE_REGISTRY

from mask2former.modeling.backbone.swin import D2SwinTransformer, PatchEmbed
from sgf_backbone import DepthTower, FiLM


@BACKBONE_REGISTRY.register()
class D2SwinTransformerHybrid(D2SwinTransformer):
    def __init__(self, cfg, input_shape):
        super().__init__(cfg, input_shape)
        old = self.patch_embed
        self.patch_embed = PatchEmbed(
            patch_size=cfg.MODEL.SWIN.PATCH_SIZE,
            in_chans=4,
            embed_dim=cfg.MODEL.SWIN.EMBED_DIM,
            norm_layer=nn.LayerNorm if cfg.MODEL.SWIN.PATCH_NORM else None,
        )
        del old
        self.depth_tower = DepthTower()
        dims = [self.num_features[i] for i in range(4)]
        self.films = nn.ModuleList([FiLM(c, d) for c, d in zip(DepthTower.CHANS, dims)])
        self._debug_film = False

    def forward(self, x):
        assert x.dim() == 4 and x.shape[1] == 4, f"Hybrid expects (N,4,H,W), got {x.shape}"
        depth = x[:, 3:4]  # already z-scored by pixel_mean/std at the meta-arch level
        feats = self.depth_tower(depth)

        # full 4ch input through the patch embed, exactly like concat4ch
        t = self.patch_embed(x)
        Wh, Ww = t.size(2), t.size(3)
        t = t.flatten(2).transpose(1, 2)
        t = self.pos_drop(t)

        outs = {}
        for i in range(self.num_layers):
            layer = self.layers[i]
            x_out, H, W, t, Wh, Ww = layer(t, Wh, Ww)
            if i in self.out_indices:
                norm_layer = getattr(self, f"norm{i}")
                x_out = norm_layer(x_out)
                out = x_out.view(-1, H, W, self.num_features[i]).permute(0, 3, 1, 2).contiguous()

                d = feats[i]
                if d.shape[-2:] != out.shape[-2:]:
                    d = F.interpolate(d, size=out.shape[-2:], mode="bilinear", align_corners=False)
                scale, shift = self.films[i](d)
                if self._debug_film:
                    print(f"[HYBRID-FILM] stage{i}: scale absmax={scale.abs().max().item():.6f} "
                          f"shift absmax={shift.abs().max().item():.6f}", flush=True)
                    self._debug_film = False
                out = out * (1 + scale) + shift
                outs["res{}".format(i + 2)] = out

        outputs = {k: v for k, v in outs.items() if k in self._out_features}
        return outputs
