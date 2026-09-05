"""CDTI depth tower + zero-init FiLM (ported from E24 d2m2f sgf_backbone).

The tower sees the 4th channel of the normalized 4ch input, i.e.
(depth*255 - 125.628) / 15.157 — identical to what the D2-side distilled
tower saw during pretraining. FiLM outputs start exactly zero, so a CDTI
model at step 0 is bit-identical to the plain 4ch concat model.
"""
import torch.nn as nn
import torch.nn.functional as F


class DepthTower(nn.Module):
    """1ch z-scored depth -> 4 pyramid levels, strides 4/8/16/32. ~0.10M params."""

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
        f0 = self.stem(depth)
        f1 = self.c1(f0)
        f2 = self.c2(f1)
        f3 = self.c3(f2)
        return [f0, f1, f2, f3]


class FiLM(nn.Module):
    """Zero-init 1x1 conv: tower_feat -> (scale, shift), each dim channels."""

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


def apply_film_injection(features, tower, films, stages, depth_normed):
    """Modulate res2..res5 features with zero-init FiLM from tower features.

    features: dict with keys res2..res5 (modified in place and returned)
    depth_normed: the z-scored 4th channel of the normalized 4ch input
    stages: iterable of stage indices 0..3 to inject (others untouched)
    """
    tfeats = tower(depth_normed)
    for i in stages:
        key = "res{}".format(i + 2)
        d = tfeats[i]
        f = features[key]
        if d.shape[-2:] != f.shape[-2:]:
            d = F.interpolate(d, size=f.shape[-2:], mode="bilinear", align_corners=False)
        scale, shift = films[i](d)
        features[key] = f * (1.0 + scale) + shift
    return features
