"""MGC (Modality-Gated Concat) Swin backbone.

Round-3 lesson: every fusion that learns a from-scratch depth feature extractor
(tower + FiLM) loses to plain concat at screening budgets, even though the
injections eventually contribute (zero-film eval drops hybrid 62.8 -> 58.9).
The only path that never loses is concat's own: the depth column of the patch
embed, one gradient step from the loss.

MGC keeps concat EXACTLY and adds input-adaptive per-location gating of the
depth channel before the patch embed:

    depth' = depth * (1 + tanh(G(depth)))     G zero-init  ->  depth' = depth

so step 0 is bit-identical to concat4ch, the gradient path into the patch-embed
depth column is unchanged, and training can learn to suppress depth where it
hurts (noise, occlusion boundaries) and amplify it where it helps. Gate G is a
1->8->1 zero-init 1x1 conv stack (~25 params).
"""
import torch
import torch.nn as nn

from detectron2.modeling import BACKBONE_REGISTRY

from mask2former.modeling.backbone.swin import D2SwinTransformer, PatchEmbed


class DepthGate(nn.Module):
    def __init__(self, hidden=8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, hidden, 1), nn.GroupNorm(hidden, hidden), nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 1, 1),
        )
        # zero-init ONLY the last layer: first layer must produce live features or
        # both weight gradients stay 0 (dead gate, learned only a global bias)
        nn.init.zeros_(self.net[3].weight); nn.init.zeros_(self.net[3].bias)

    def forward(self, depth):
        return depth * (1 + torch.tanh(self.net(depth)))


@BACKBONE_REGISTRY.register()
class D2SwinTransformerMGC(D2SwinTransformer):
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
        self.depth_gate = DepthGate()

    def forward(self, x):
        assert x.dim() == 4 and x.shape[1] == 4, f"MGC expects (N,4,H,W), got {x.shape}"
        x = torch.cat([x[:, :3], self.depth_gate(x[:, 3:4])], dim=1)
        return super().forward(x)
