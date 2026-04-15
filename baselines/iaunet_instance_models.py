from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


def _dice_loss(
    pred_masks: torch.Tensor,
    target_masks: torch.Tensor,
    *,
    eps: float = 1.0,
) -> torch.Tensor:
    pred = pred_masks.sigmoid().flatten(1)
    target = target_masks.flatten(1)
    numerator = 2.0 * (pred * target).sum(dim=1) + float(eps)
    denominator = pred.sum(dim=1) + target.sum(dim=1) + float(eps)
    return 1.0 - (numerator / denominator)


def _batch_pairwise_dice_cost(
    pred_masks: torch.Tensor,
    target_masks: torch.Tensor,
    *,
    eps: float = 1.0,
) -> torch.Tensor:
    pred = pred_masks.flatten(1)
    target = target_masks.flatten(1)
    numerator = 2.0 * torch.einsum("qc,mc->qm", pred, target) + float(eps)
    denominator = pred.sum(dim=1, keepdim=True) + target.sum(dim=1).unsqueeze(0) + float(eps)
    return 1.0 - (numerator / denominator)


def _batch_sigmoid_bce_cost(
    pred_logits: torch.Tensor,
    target_masks: torch.Tensor,
) -> torch.Tensor:
    pred = pred_logits.flatten(1)
    target = target_masks.flatten(1)
    pred_prob = pred.sigmoid()
    pred_prob = pred_prob[:, None, :].expand(-1, target.shape[0], -1)
    target = target[None, :, :].expand(pred.shape[0], -1, -1)
    return F.binary_cross_entropy(pred_prob, target, reduction="none").mean(dim=-1)


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv = ConvBlock(in_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv = ConvBlock(out_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = self.proj(x)
        return self.conv(torch.cat([x, skip], dim=1))


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, num_layers: int = 3) -> None:
        super().__init__()
        dims = [in_dim] + [hidden_dim] * max(0, int(num_layers) - 1) + [out_dim]
        layers: List[nn.Module] = []
        for idx in range(len(dims) - 1):
            layers.append(nn.Linear(dims[idx], dims[idx + 1]))
            if idx < len(dims) - 2:
                layers.append(nn.ReLU(inplace=True))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LightweightPixelDecoder(nn.Module):
    def __init__(
        self,
        in_channels_list: Sequence[int],
        hidden_dim: int,
        mask_dim: int,
        pooled_size: int = 8,
    ) -> None:
        super().__init__()
        self.pooled_size = int(pooled_size)
        self.input_projs = nn.ModuleList(
            [nn.Conv2d(in_channels, hidden_dim, kernel_size=1) for in_channels in in_channels_list]
        )
        self.output_convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(hidden_dim),
                    nn.ReLU(inplace=True),
                )
                for _ in in_channels_list
            ]
        )
        self.mask_proj = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, mask_dim, kernel_size=1),
        )

    def forward(self, features: Sequence[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        fused: torch.Tensor | None = None
        processed: List[torch.Tensor] = []
        for feat, proj, out_conv in zip(reversed(features), reversed(self.input_projs), reversed(self.output_convs)):
            x = proj(feat)
            if fused is not None:
                x = x + F.interpolate(fused, size=x.shape[-2:], mode="bilinear", align_corners=False)
            fused = out_conv(x)
            processed.append(fused)
        processed = list(reversed(processed))

        mask_features = self.mask_proj(processed[0])
        memory_tokens = [
            F.adaptive_avg_pool2d(feat, output_size=(self.pooled_size, self.pooled_size))
            .flatten(2)
            .permute(2, 0, 1)
            for feat in processed
        ]
        memory = torch.cat(memory_tokens, dim=0)
        return mask_features, memory


class IAUNetQueryDecoder(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_queries: int,
        num_layers: int,
        num_heads: int,
    ) -> None:
        super().__init__()
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.query_pos = nn.Embedding(num_queries, hidden_dim)
        self.layers = nn.ModuleList(
            [
                nn.TransformerDecoderLayer(
                    d_model=hidden_dim,
                    nhead=num_heads,
                    dim_feedforward=hidden_dim * 4,
                    dropout=0.0,
                    batch_first=False,
                    activation="relu",
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, memory: torch.Tensor) -> List[torch.Tensor]:
        batch_size = int(memory.shape[1])
        query = self.query_embed.weight.unsqueeze(1).repeat(1, batch_size, 1)
        query_pos = self.query_pos.weight.unsqueeze(1).repeat(1, batch_size, 1)
        hidden_states: List[torch.Tensor] = []
        tgt = query
        for layer in self.layers:
            tgt = layer(tgt=tgt + query_pos, memory=memory)
            hidden_states.append(self.norm(tgt).permute(1, 0, 2))
        return hidden_states


class IAUNetInstanceModel(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int = 3,
        base_channels: int = 32,
        hidden_dim: int = 128,
        num_queries: int = 64,
        num_decoder_layers: int = 4,
        num_heads: int = 8,
        mask_dim: int | None = None,
    ) -> None:
        super().__init__()
        mask_dim = int(mask_dim or hidden_dim)
        c1 = int(base_channels)
        c2 = c1 * 2
        c3 = c2 * 2
        c4 = c3 * 2
        c5 = c4 * 2

        self.stem = ConvBlock(in_channels, c1)
        self.down1 = DownBlock(c1, c2)
        self.down2 = DownBlock(c2, c3)
        self.down3 = DownBlock(c3, c4)
        self.down4 = DownBlock(c4, c5)

        self.up3 = UpBlock(c5, c4, c4)
        self.up2 = UpBlock(c4, c3, c3)
        self.up1 = UpBlock(c3, c2, c2)

        self.pixel_decoder = LightweightPixelDecoder(
            in_channels_list=[c2, c3, c4, c5],
            hidden_dim=hidden_dim,
            mask_dim=mask_dim,
        )
        self.query_decoder = IAUNetQueryDecoder(
            hidden_dim=hidden_dim,
            num_queries=num_queries,
            num_layers=num_decoder_layers,
            num_heads=num_heads,
        )
        self.class_head = nn.Linear(hidden_dim, 2)
        self.mask_embed = MLP(hidden_dim, hidden_dim, mask_dim)

    def _decode_queries(
        self,
        hidden_states: Sequence[torch.Tensor],
        mask_features: torch.Tensor,
    ) -> Tuple[Dict[str, torch.Tensor], List[Dict[str, torch.Tensor]]]:
        predictions: List[Dict[str, torch.Tensor]] = []
        for hidden in hidden_states:
            class_logits = self.class_head(hidden)
            mask_embed = self.mask_embed(hidden)
            mask_logits = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)
            predictions.append({"pred_logits": class_logits, "pred_masks": mask_logits})
        final = predictions[-1]
        return final, predictions[:-1]

    def forward(self, images: torch.Tensor) -> Dict[str, Any]:
        x1 = self.stem(images)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        y4 = self.up3(x5, x4)
        y3 = self.up2(y4, x3)
        y2 = self.up1(y3, x2)

        mask_features, memory = self.pixel_decoder([y2, y3, y4, x5])
        hidden_states = self.query_decoder(memory)
        final, aux_outputs = self._decode_queries(hidden_states, mask_features)
        if aux_outputs:
            final["aux_outputs"] = aux_outputs
        return final


def _resize_target_masks(
    targets: Sequence[Dict[str, torch.Tensor]],
    mask_size: Tuple[int, int],
    device: torch.device,
) -> List[torch.Tensor]:
    resized: List[torch.Tensor] = []
    for target in targets:
        masks = target["masks"].to(device=device, dtype=torch.float32)
        if masks.numel() == 0:
            resized.append(masks.reshape(0, mask_size[0], mask_size[1]))
            continue
        if masks.shape[-2:] != mask_size:
            masks = F.interpolate(masks[:, None, ...], size=mask_size, mode="nearest").squeeze(1)
        resized.append(masks)
    return resized


class IAUNetHungarianMatcher:
    def __init__(
        self,
        *,
        cost_class: float = 2.0,
        cost_mask: float = 5.0,
        cost_dice: float = 5.0,
    ) -> None:
        self.cost_class = float(cost_class)
        self.cost_mask = float(cost_mask)
        self.cost_dice = float(cost_dice)

    @torch.no_grad()
    def __call__(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Sequence[Dict[str, torch.Tensor]],
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        pred_logits = outputs["pred_logits"]
        pred_masks = outputs["pred_masks"]
        device = pred_logits.device
        target_masks = _resize_target_masks(targets, pred_masks.shape[-2:], device)
        out_prob = pred_logits.softmax(dim=-1)

        indices: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for batch_idx, masks in enumerate(target_masks):
            if masks.numel() == 0:
                empty = torch.zeros((0,), dtype=torch.int64, device=device)
                indices.append((empty, empty))
                continue

            object_cost = -out_prob[batch_idx, :, 1].unsqueeze(1).repeat(1, masks.shape[0])
            mask_cost = _batch_sigmoid_bce_cost(pred_masks[batch_idx], masks)
            dice_cost = _batch_pairwise_dice_cost(pred_masks[batch_idx].sigmoid(), masks)
            cost_matrix = (
                self.cost_class * object_cost
                + self.cost_mask * mask_cost
                + self.cost_dice * dice_cost
            ).detach().cpu()
            src_idx, tgt_idx = linear_sum_assignment(cost_matrix)
            indices.append(
                (
                    torch.as_tensor(src_idx, dtype=torch.int64, device=device),
                    torch.as_tensor(tgt_idx, dtype=torch.int64, device=device),
                )
            )
        return indices


class IAUNetCriterion(nn.Module):
    def __init__(
        self,
        *,
        matcher: IAUNetHungarianMatcher,
        eos_coef: float = 0.1,
    ) -> None:
        super().__init__()
        self.matcher = matcher
        self.register_buffer("class_weights", torch.tensor([float(eos_coef), 1.0], dtype=torch.float32))

    def _compute_losses(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Sequence[Dict[str, torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        pred_logits = outputs["pred_logits"]
        pred_masks = outputs["pred_masks"]
        device = pred_logits.device
        indices = self.matcher(outputs, targets)
        target_masks = _resize_target_masks(targets, pred_masks.shape[-2:], device)

        target_classes = torch.zeros(pred_logits.shape[:2], dtype=torch.int64, device=device)
        matched_pred_masks: List[torch.Tensor] = []
        matched_target_masks: List[torch.Tensor] = []

        for batch_idx, (src_idx, tgt_idx) in enumerate(indices):
            if src_idx.numel() == 0:
                continue
            target_classes[batch_idx, src_idx] = 1
            matched_pred_masks.append(pred_masks[batch_idx, src_idx])
            matched_target_masks.append(target_masks[batch_idx][tgt_idx])

        loss_ce = F.cross_entropy(pred_logits.transpose(1, 2), target_classes, weight=self.class_weights)
        if matched_pred_masks:
            src_masks = torch.cat(matched_pred_masks, dim=0)
            tgt_masks = torch.cat(matched_target_masks, dim=0)
            loss_mask = F.binary_cross_entropy_with_logits(src_masks, tgt_masks)
            loss_dice = _dice_loss(src_masks, tgt_masks).mean()
        else:
            zero = pred_logits.sum() * 0.0
            loss_mask = zero
            loss_dice = zero
        return {
            "loss_ce": loss_ce,
            "loss_mask": loss_mask,
            "loss_dice": loss_dice,
        }

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Sequence[Dict[str, torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        losses = self._compute_losses(outputs, targets)
        aux_outputs = outputs.get("aux_outputs", [])
        for layer_idx, aux_output in enumerate(aux_outputs):
            aux_losses = self._compute_losses(aux_output, targets)
            for key, value in aux_losses.items():
                losses[f"{key}_aux{layer_idx}"] = value
        return losses


@torch.no_grad()
def iaunet_inference(
    outputs: Dict[str, torch.Tensor],
    *,
    original_sizes: Sequence[Tuple[int, int]],
    score_threshold: float = 0.4,
    mask_threshold: float = 0.5,
    min_area: int = 20,
    max_instances: int | None = None,
) -> List[Dict[str, torch.Tensor]]:
    pred_logits = outputs["pred_logits"]
    pred_masks = outputs["pred_masks"]
    scores = pred_logits.softmax(dim=-1)[..., 1]

    results: List[Dict[str, torch.Tensor]] = []
    for batch_idx, orig_size in enumerate(original_sizes):
        sample_scores = scores[batch_idx]
        sample_masks = pred_masks[batch_idx]
        keep = sample_scores >= float(score_threshold)
        if keep.any():
            sample_scores = sample_scores[keep]
            sample_masks = sample_masks[keep]
        else:
            empty_masks = torch.zeros((0, int(orig_size[0]), int(orig_size[1])), dtype=torch.uint8)
            results.append(
                {
                    "scores": torch.zeros((0,), dtype=torch.float32),
                    "category_ids": torch.zeros((0,), dtype=torch.int64),
                    "masks": empty_masks,
                }
            )
            continue

        order = torch.argsort(sample_scores, descending=True)
        if max_instances is not None and order.numel() > int(max_instances):
            order = order[: int(max_instances)]
        sample_scores = sample_scores[order]
        sample_masks = sample_masks[order]

        sample_masks = F.interpolate(
            sample_masks[:, None, ...],
            size=(int(orig_size[0]), int(orig_size[1])),
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)
        binary_masks = (sample_masks.sigmoid() >= float(mask_threshold)).to(torch.uint8)
        flat_area = binary_masks.flatten(1).sum(dim=1)
        keep_area = flat_area >= int(min_area)

        if keep_area.any():
            sample_scores = sample_scores[keep_area].detach().cpu()
            binary_masks = binary_masks[keep_area].detach().cpu()
            category_ids = torch.zeros((binary_masks.shape[0],), dtype=torch.int64)
        else:
            sample_scores = torch.zeros((0,), dtype=torch.float32)
            binary_masks = torch.zeros((0, int(orig_size[0]), int(orig_size[1])), dtype=torch.uint8)
            category_ids = torch.zeros((0,), dtype=torch.int64)

        results.append(
            {
                "scores": sample_scores,
                "category_ids": category_ids,
                "masks": binary_masks,
            }
        )
    return results


def count_trainable_parameters(model: nn.Module) -> int:
    return int(sum(param.numel() for param in model.parameters() if param.requires_grad))
