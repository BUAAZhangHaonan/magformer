"""
Copy-Paste augmentation for MagFormer instance segmentation.

Implements the "Simple Copy-Paste Is a Strong Data Augmentation Method
for Instance Segmentation" (Ghiasi et al., CVPR 2021) approach adapted
for MagFormer's data pipeline.

How it works:
1. Sample a "source" image and extract its instances (masks + crop regions)
2. Paste selected instances onto the "target" image at random locations
3. Update the target's annotations to include pasted instances
4. Handle overlaps by paste-order (later instances occlude earlier ones)

Expected improvement: +1-2 AP, especially for small/rare objects.

Usage in dataset __getitem__:
    from magformer.data.copy_paste_aug import CopyPasteAugmentation

    copy_paste = CopyPasteAugmentation(
        dataset=dataset,        # reference to the dataset for sampling source images
        prob=0.5,               # probability of applying copy-paste per image
        max_paste_instances=5,  # max instances to paste per image
        min_instance_area=16,   # skip very tiny instances
        max_instance_area=0.3,  # skip instances > 30% of image area
    )

    # In __getitem__:
    if self.training:
        image, annotations = copy_paste(image, annotations, idx)
"""

import random
import math
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)


class CopyPasteAugmentation:
    """Copy-Paste augmentation for instance segmentation.

    Copies object instances from one image and pastes them onto another,
    with random positioning and scale jitter. Handles mask overlaps via
    paste-order (later pasted objects occlude earlier ones).

    Args:
        dataset: The dataset object (must support __getitem__ to sample source images).
        prob: Probability of applying copy-paste to each image.
        max_paste_instances: Maximum number of instances to paste.
        min_instance_area: Minimum instance area (pixels) to consider for copying.
        max_instance_area_ratio: Maximum instance area as fraction of total image area.
        scale_jitter: Tuple of (min, max) scale factors for pasted instances.
        iou_threshold: IoU threshold - skip paste if overlap with existing instance > this.
    """

    def __init__(
        self,
        dataset=None,
        prob: float = 0.5,
        max_paste_instances: int = 5,
        min_instance_area: int = 16,
        max_instance_area_ratio: float = 0.3,
        scale_jitter: Tuple[float, float] = (0.8, 1.2),
        iou_threshold: float = 0.7,
        training: bool = True,
    ):
        self.dataset = dataset
        self.prob = prob
        self.max_paste_instances = max_paste_instances
        self.min_instance_area = min_instance_area
        self.max_instance_area_ratio = max_instance_area_ratio
        self.scale_jitter = scale_jitter
        self.iou_threshold = iou_threshold
        self.training = training

    def _get_instance_crop(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        bbox: List[int],
    ) -> Tuple[np.ndarray, np.ndarray, int, int]:
        """Extract instance crop from image and mask.

        Returns:
            cropped_image: [H_crop, W_crop, 3] uint8
            cropped_mask: [H_crop, W_crop] bool
            cy, cx: center of the crop in original image coords
        """
        x, y, w, h = bbox
        # Add small padding for safety
        pad = 2
        y1 = max(0, y - pad)
        y2 = min(image.shape[0], y + h + pad)
        x1 = max(0, x - pad)
        x2 = min(image.shape[1], x + w + pad)

        cropped_image = image[y1:y2, x1:x2].copy()
        cropped_mask = mask[y1:y2, x1:x2].copy()

        cy = (y1 + y2) // 2
        cx = (x1 + x2) // 2

        return cropped_image, cropped_mask, cy, cx

    def _paste_instance(
        self,
        target_image: np.ndarray,
        target_mask_canvas: np.ndarray,
        target_instance_id: int,
        source_crop_img: np.ndarray,
        source_crop_mask: np.ndarray,
        paste_y: int,
        paste_x: int,
        scale: float = 1.0,
    ) -> Tuple[np.ndarray, np.ndarray, bool]:
        """Paste a single instance onto the target image.

        Args:
            target_image: [H, W, 3] uint8 - modified in place
            target_mask_canvas: [H, W] int - instance ID canvas, modified in place
            target_instance_id: ID for this instance on the canvas
            source_crop_img: [H_crop, W_crop, 3] uint8
            source_crop_mask: [H_crop, W_crop] bool
            paste_y, paste_x: top-left corner of paste location
            scale: scale factor for the crop

        Returns:
            Modified target_image, target_mask_canvas, and whether paste succeeded
        """
        h_img, w_img = target_image.shape[:2]

        # Scale the crop if needed
        if scale != 1.0:
            new_h = int(source_crop_img.shape[0] * scale)
            new_w = int(source_crop_img.shape[1] * scale)
            if new_h < 2 or new_w < 2:
                return target_image, target_mask_canvas, False

            import cv2
            source_crop_img = cv2.resize(source_crop_img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            source_crop_mask = cv2.resize(
                source_crop_mask.astype(np.uint8), (new_w, new_h), interpolation=cv2.INTER_NEAREST
            ).astype(bool)

        crop_h, crop_w = source_crop_img.shape[:2]

        # Clip to image boundaries
        y1 = max(0, paste_y)
        x1 = max(0, paste_x)
        y2 = min(h_img, paste_y + crop_h)
        x2 = min(w_img, paste_x + crop_w)

        if y1 >= y2 or x1 >= x2:
            return target_image, target_mask_canvas, False

        # Source crop offsets (in case paste location is partially outside)
        sy1 = y1 - paste_y
        sx1 = x1 - paste_x
        sy2 = sy1 + (y2 - y1)
        sx2 = sx1 + (x2 - x1)

        # Get the paste region
        paste_mask = source_crop_mask[sy1:sy2, sx1:sx2]

        # Check IoU with existing instances - skip if too much overlap
        existing_ids = target_mask_canvas[y1:y2, x1:x2]
        overlap_area = np.sum(paste_mask & (existing_ids > 0))
        paste_area = np.sum(paste_mask)
        if paste_area > 0 and overlap_area / paste_area > self.iou_threshold:
            return target_image, target_mask_canvas, False

        # Paste: use mask to blend
        mask_3ch = np.stack([paste_mask] * 3, axis=-1)
        target_image[y1:y2, x1:x2] = np.where(
            mask_3ch,
            source_crop_img[sy1:sy2, sx1:sx2],
            target_image[y1:y2, x1:x2],
        )

        # Update mask canvas (later pastes overwrite earlier ones)
        target_mask_canvas[y1:y2, x1:x2] = np.where(
            paste_mask,
            target_instance_id,
            target_mask_canvas[y1:y2, x1:x2],
        )

        return target_image, target_mask_canvas, True

    def __call__(
        self,
        image: np.ndarray,
        masks: List[np.ndarray],
        bboxes: List[List[int]],
        categories: List[int],
        current_idx: int,
    ) -> Tuple[np.ndarray, List[np.ndarray], List[List[int]], List[int]]:
        """Apply copy-paste augmentation.

        Args:
            image: [H, W, 3] uint8 RGB image
            masks: List of [H, W] bool masks
            bboxes: List of [x, y, w, h] bounding boxes
            categories: List of category IDs
            current_idx: Current image index (to avoid sampling same image)

        Returns:
            Modified (image, masks, bboxes, categories)
        """
        if not self.training or random.random() > self.prob:
            return image, masks, bboxes, categories

        if self.dataset is None or len(masks) == 0:
            return image, masks, bboxes, categories

        h_img, w_img = image.shape[:2]
        total_area = h_img * w_img
        img_area_ratio = self.max_instance_area_ratio

        # Build instance mask canvas from current annotations
        mask_canvas = np.zeros((h_img, w_img), dtype=np.int32)
        for i, mask in enumerate(masks):
            mask_canvas[mask] = i + 1  # 1-indexed instance IDs

        # Sample source image(s)
        n_source_attempts = 3
        source_candidates = []

        for _ in range(n_source_attempts):
            src_idx = random.randint(0, len(self.dataset) - 1)
            if src_idx == current_idx:
                continue

            try:
                src_data = self.dataset[src_idx]
                if "instances" in src_data:
                    src_masks = src_data["instances"].masks  # depends on format
                    src_bboxes = src_data["instances"].bboxes
                    src_cats = src_data["instances"].categories
                    src_image = src_data.get("image_src", image)  # raw image if available
                    source_candidates.append((src_image, src_masks, src_bboxes, src_cats))
            except Exception:
                continue

        if not source_candidates:
            return image, masks, bboxes, categories

        # Collect pastable instances from source images
        pastable = []
        for src_image, src_masks, src_bboxes, src_cats in source_candidates:
            for j, (mask, bbox, cat) in enumerate(zip(src_masks, src_bboxes, src_cats)):
                area = mask.sum()
                if area < self.min_instance_area:
                    continue
                if area > total_area * img_area_ratio:
                    continue
                pastable.append((src_image, mask, bbox, cat))

        if not pastable:
            return image, masks, bboxes, categories

        # Select instances to paste
        n_paste = min(random.randint(1, self.max_paste_instances), len(pastable))
        selected = random.sample(pastable, n_paste)

        new_masks = list(masks)
        new_bboxes = list(bboxes)
        new_categories = list(categories)

        for src_image, src_mask, src_bbox, src_cat in selected:
            # Get the instance crop
            crop_img, crop_mask, _, _ = self._get_instance_crop(src_image, src_mask, src_bbox)

            # Scale jitter
            scale = random.uniform(*self.scale_jitter)

            # Random paste location
            crop_h, crop_w = crop_img.shape[:2]
            paste_y = random.randint(-crop_h // 2, h_img - crop_h // 2)
            paste_x = random.randint(-crop_w // 2, w_img - crop_w // 2)

            instance_id = len(new_masks) + 1
            image, mask_canvas, success = self._paste_instance(
                image, mask_canvas, instance_id,
                crop_img, crop_mask, paste_y, paste_x, scale
            )

            if success:
                # Extract the pasted mask from canvas
                pasted_mask = (mask_canvas == instance_id)

                if pasted_mask.sum() < self.min_instance_area:
                    # Too small after clipping, undo
                    mask_canvas[mask_canvas == instance_id] = 0
                    continue

                # Compute bbox from pasted mask
                ys, xs = np.where(pasted_mask)
                new_bbox = [int(xs.min()), int(ys.min()),
                           int(xs.max() - xs.min() + 1),
                           int(ys.max() - ys.min() + 1)]

                new_masks.append(pasted_mask)
                new_bboxes.append(new_bbox)
                new_categories.append(src_cat)

        return image, new_masks, new_bboxes, new_categories


class CopyPasteAugmentationTensor:
    """Tensor-based copy-paste augmentation for use in training transforms.

    Works with torch tensors instead of numpy arrays. Simpler interface
    designed to integrate into MagFormer's training data pipeline.

    Usage in transforms:
        cp_aug = CopyPasteAugmentationTensor(prob=0.5)

        # After loading image + annotations as tensors:
        image, masks, boxes = cp_aug(image, masks, boxes, dataset, img_idx)
    """

    def __init__(
        self,
        prob: float = 0.5,
        max_paste_instances: int = 3,
        min_instance_area: int = 16,
        scale_jitter: Tuple[float, float] = (0.8, 1.2),
    ):
        self.prob = prob
        self.max_paste_instances = max_paste_instances
        self.min_instance_area = min_instance_area
        self.scale_jitter = scale_jitter

    def __call__(
        self,
        image: torch.Tensor,
        masks: torch.Tensor,
        boxes: torch.Tensor,
        dataset=None,
        current_idx: int = -1,
        training: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply copy-paste augmentation on tensors.

        Args:
            image: [3, H, W] float tensor
            masks: [N, H, W] bool/float tensor
            boxes: [N, 4] int tensor (x1, y1, x2, y2)
            dataset: dataset for sampling source images
            current_idx: current image index
            training: whether in training mode

        Returns:
            Modified (image, masks, boxes)
        """
        if not training or random.random() > self.prob:
            return image, masks, boxes

        if dataset is None or masks.shape[0] == 0:
            return image, masks, boxes

        H, W = image.shape[1], image.shape[2]
        device = image.device

        # Sample source image
        for _ in range(3):
            src_idx = random.randint(0, len(dataset) - 1)
            if src_idx == current_idx:
                continue

            try:
                src_data = dataset[src_idx]
                src_image = src_data["image"].to(device)  # [3, H, W]
                src_masks = src_data.get("masks", torch.zeros(0, H, W, device=device))
                src_boxes = src_data.get("boxes", torch.zeros(0, 4, device=device))

                if src_masks.shape[0] == 0:
                    continue

                break
            except Exception:
                continue
        else:
            return image, masks, boxes

        # Select instances to paste
        n_paste = min(random.randint(1, self.max_paste_instances), src_masks.shape[0])
        paste_indices = random.sample(range(src_masks.shape[0]), n_paste)

        new_masks_list = [masks]
        new_boxes_list = [boxes]

        for idx in paste_indices:
            src_mask = src_masks[idx]  # [H, W]
            src_box = src_boxes[idx]   # [4]

            mask_area = src_mask.sum().item()
            if mask_area < self.min_instance_area:
                continue

            # Get crop region
            x1, y1, x2, y2 = src_box.int().tolist()
            # Add padding
            pad = 2
            cy1, cy2 = max(0, y1 - pad), min(H, y2 + pad)
            cx1, cx2 = max(0, x1 - pad), min(W, x2 + pad)

            crop_img = src_image[:, cy1:cy2, cx1:cx2].clone()  # [3, h, w]
            crop_mask = src_mask[cy1:cy2, cx1:cx2].clone()      # [h, w]
            crop_h, crop_w = crop_mask.shape

            if crop_h < 2 or crop_w < 2:
                continue

            # Random paste location
            paste_y = random.randint(0, max(0, H - crop_h))
            paste_x = random.randint(0, max(0, W - crop_w))

            # Clip to image bounds
            py1 = max(0, paste_y)
            px1 = max(0, paste_x)
            py2 = min(H, paste_y + crop_h)
            px2 = min(W, paste_x + crop_w)

            # Source offsets
            sy1 = py1 - paste_y
            sx1 = px1 - paste_x
            sy2 = sy1 + (py2 - py1)
            sx2 = sx1 + (px2 - px1)

            paste_region_mask = crop_mask[sy1:sy2, sx1:sx2]  # [h', w']
            if paste_region_mask.sum().item() < self.min_instance_area:
                continue

            # Paste onto image using mask blending
            mask_3d = paste_region_mask.unsqueeze(0).expand(3, -1, -1)  # [3, h', w']
            image[:, py1:py2, px1:px2] = torch.where(
                mask_3d.bool(),
                crop_img[:, sy1:sy2, sx1:sx2],
                image[:, py1:py2, px1:px2],
            )

            # Create full-size mask for pasted instance
            new_mask = torch.zeros(H, W, dtype=masks.dtype, device=device)
            new_mask[py1:py2, px1:px2] = paste_region_mask

            # Compute bbox
            ys, xs = torch.where(new_mask > 0)
            if len(ys) == 0:
                continue
            new_box = torch.tensor(
                [xs.min().item(), ys.min().item(), xs.max().item(), ys.max().item()],
                dtype=boxes.dtype, device=device
            ).unsqueeze(0)

            new_masks_list.append(new_mask.unsqueeze(0))
            new_boxes_list.append(new_box)

        # Concatenate results
        if len(new_masks_list) > 1:
            masks = torch.cat(new_masks_list, dim=0)
            boxes = torch.cat(new_boxes_list, dim=0)

        return image, masks, boxes
