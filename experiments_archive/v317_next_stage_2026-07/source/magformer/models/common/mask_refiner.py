import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskRefiner(nn.Module):
    """Image-conditioned mask boundary refiner.
    
    Takes a coarse (bilinear-upsampled) mask and the original image,
    learns a residual correction using image edge information.
    Last conv initialized to zero for identity start (stable training).
    
    ~10K parameters: 4*32*3*3 + 32*32*3*3 + 32*1*1*1 + biases = 1152 + 9216 + 32 + biases ≈ 10.4K
    """
    
    def __init__(self, hidden_dim=32):
        super().__init__()
        self.refine = nn.Sequential(
            nn.Conv2d(4, hidden_dim, 3, padding=1),  # 3 (RGB) + 1 (coarse mask)
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, 1, 1),
        )
        # Zero init for identity start
        nn.init.zeros_(self.refine[-1].weight)
        nn.init.zeros_(self.refine[-1].bias)
    
    def forward(self, coarse_masks, images):
        """
        Args:
            coarse_masks: (N, 1, H, W) bilinear-upsampled masks (logits)
            images: (N, 3, H_img, W_img) normalized images (may differ from mask size)
        Returns:
            refined_masks: (N, 1, H, W) boundary-refined masks (logits)
        """
        # Downsample images to match mask resolution if needed
        if images.shape[-2:] != coarse_masks.shape[-2:]:
            images = F.interpolate(images, size=coarse_masks.shape[-2:], mode='bilinear', align_corners=False)
        x = torch.cat([coarse_masks, images], dim=1)
        residual = self.refine(x)
        return coarse_masks + residual


class LearnedUpsampler(nn.Module):
    """Learned mask feature upsampler: stride-4 -> stride-2.
    
    Bilinear baseline + learned residual via PixelShuffle.
    Zero-init conv so starting point = bilinear upsampling (identity).
    
    This means initial mask predictions are identical to the baseline, since:
        einsum(mask_embed, bilinear(mask_features)) == bilinear(einsum(mask_embed, mask_features))
    
    ~590K parameters: 256*1024*3*3 + bias = 2,359,808 + bias
    """
    
    def __init__(self, channels=256):
        super().__init__()
        self.residual = nn.Sequential(
            nn.Conv2d(channels, channels * 4, 3, padding=1),
            nn.GELU(),
            nn.PixelShuffle(2),
        )
        # Zero-init so starting point = bilinear upsampling (identity)
        nn.init.zeros_(self.residual[0].weight)
        nn.init.zeros_(self.residual[0].bias)
    
    def forward(self, x):
        baseline = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        return baseline + self.residual(x)
