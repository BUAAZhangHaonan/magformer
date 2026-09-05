"""SGF (Structured Gated Fusion) Swin backbone.

Input is still the 4ch RGBD tensor from RGBDConcatMapper. RGB goes through the
frozen-shape 3ch patch embed (86143f loads untouched); depth (channel 3, which
is metric_depth*255) is z-scored and passed through a light conv tower whose
strides (4/8/16/32) align with the Swin stages. Each stage output feature
(res2..res5, AFTER the per-stage norm, i.e. on the final backbone output
tensor of that stage) is modulated FiLM-style:

    out = out * (1 + scale(depth_feat)) + shift(depth_feat)

with scale/shift from a single zero-initialised 1x1 conv per stage (output
2*dim channels, split in half). At step 0 the FiLM contribution is exactly 0,
so the model starts identical to the COCO-pretrained Swin.

Injection point (recorded): on x_out after norm{i}, at the same point where
the base forward produces the outs["res{i+2}"] feature map.

Parameter budget: tower ~0.10M + FiLM convs ~0.26M = ~0.36M total (<0.5M).
The per-stage "1x1 proj to token dim" from the original sketch was folded
into the FiLM conv (tower_feat -> 2*dim) because proj+FiLM would exceed the
0.5M cap; behaviour is equivalent.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from detectron2.modeling import BACKBONE_REGISTRY

from mask2former.modeling.backbone.swin import D2SwinTransformer

DEPTH_MEAN = 125.628  # mapper: metric depth * 255
DEPTH_STD = 15.157


class DepthTower(nn.Module):
    """1ch z-scored depth (or 3ch z-scored RGB) -> 4 pyramid levels, strides 4-32."""

    CHANS = (16, 32, 64, 128)

    def __init__(self, in_chans=1):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_chans, 16, 3, stride=4, padding=1),
            nn.GroupNorm(4, 16),
            nn.ReLU(inplace=True),
        )
        self.c1 = nn.Sequential(nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.GroupNorm(8, 32), nn.ReLU(inplace=True))
        self.c2 = nn.Sequential(nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.GroupNorm(16, 64), nn.ReLU(inplace=True))
        self.c3 = nn.Sequential(nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.GroupNorm(32, 128), nn.ReLU(inplace=True))

    def forward(self, depth):
        f0 = self.stem(depth)   # 1/4
        f1 = self.c1(f0)        # 1/8
        f2 = self.c2(f1)        # 1/16
        f3 = self.c3(f2)        # 1/32
        return [f0, f1, f2, f3]


class FiLM(nn.Module):
    """Zero-init 1x1 conv: depth_feat -> (scale, shift), each dim channels."""

    def __init__(self, in_ch, dim):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, 2 * dim, 1)
        nn.init.zeros_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)
        self.dim = dim

    def forward(self, feat):
        film = self.conv(feat)
        scale, shift = film.chunk(2, dim=1)
        return scale, shift


@BACKBONE_REGISTRY.register()
class D2SwinTransformerSGF(D2SwinTransformer):
    def __init__(self, cfg, input_shape):
        super().__init__(cfg, input_shape)
        self.depth_tower = DepthTower()
        dims = [self.num_features[i] for i in range(4)]
        self.films = nn.ModuleList([FiLM(c, d) for c, d in zip(DepthTower.CHANS, dims)])
        # FiLM diagnostics (set True to print abs-max once)
        self._debug_film = False

    def forward(self, x):
        assert x.dim() == 4 and x.shape[1] == 4, f"SGF expects (N,4,H,W), got {x.shape}"
        rgb, depth = x[:, :3], x[:, 3:4]
        # input depth already z-scored by pixel_mean/std at the meta-arch level
        feats = self.depth_tower(depth)

        # replicate M2F SwinTransformer.forward with FiLM on stage outputs
        t = self.patch_embed(rgb)
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
                    print(f"[SGF-FILM] stage{i}: scale absmax={scale.abs().max().item():.6f} "
                          f"shift absmax={shift.abs().max().item():.6f}", flush=True)
                    self._debug_film = False
                out = out * (1 + scale) + shift
                outs["res{}".format(i + 2)] = out

        outputs = {k: v for k, v in outs.items() if k in self._out_features}
        return outputs

    def depth_tower_grad_norm(self):
        total = 0.0
        for p in self.depth_tower.parameters():
            if p.grad is not None:
                total += p.grad.norm().item() ** 2
        return total ** 0.5
