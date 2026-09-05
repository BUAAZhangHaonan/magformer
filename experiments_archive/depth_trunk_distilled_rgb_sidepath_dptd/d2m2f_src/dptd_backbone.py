"""DPTD: Depth-primary trunk + distilled RGB side-path (mirror of hybrid_ptd).

Round-5 pivot: depth_only beats every fusion (69.63 vs concat 65.30 @10K) on
this dataset — depth is the dominant modality. DPTD mirrors the winning
hybrid_ptd recipe with roles swapped: the Swin trunk is fed the z-scored depth
channel replicated 3x (loads the 3ch COCO stem), and a distilled RGB tower
(mask-proxy pretrained) injects RGB texture per stage via zero-init FiLM.
Step 0 is EXACTLY depth_only; RGB can only be learned on top.
"""
import torch.nn as nn
import torch.nn.functional as F

from detectron2.modeling import BACKBONE_REGISTRY

from mask2former.modeling.backbone.swin import D2SwinTransformer
from sgf_backbone import DepthTower, FiLM


@BACKBONE_REGISTRY.register()
class D2SwinTransformerDPTD(D2SwinTransformer):
    def __init__(self, cfg, input_shape):
        super().__init__(cfg, input_shape)
        self.rgb_tower = DepthTower(in_chans=3)
        dims = [self.num_features[i] for i in range(4)]
        self.films = nn.ModuleList([FiLM(c, d) for c, d in zip(DepthTower.CHANS, dims)])
        self._debug_film = False

    def forward(self, x):
        assert x.dim() == 4 and x.shape[1] == 4, f"DPTD expects (N,4,H,W), got {x.shape}"
        feats = self.rgb_tower(x[:, :3])  # z-scored RGB

        t = self.patch_embed(x[:, 3:4].repeat(1, 3, 1, 1))  # depth-primary stem
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
                    print(f"[DPTD-FILM] stage{i}: scale absmax={scale.abs().max().item():.6f} "
                          f"shift absmax={shift.abs().max().item():.6f}", flush=True)
                    self._debug_film = False
                out = out * (1 + scale) + shift
                outs["res{}".format(i + 2)] = out

        outputs = {k: v for k, v in outs.items() if k in self._out_features}
        return outputs
