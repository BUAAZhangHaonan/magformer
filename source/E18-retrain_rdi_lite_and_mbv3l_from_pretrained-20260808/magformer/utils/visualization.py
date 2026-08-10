from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import cv2
import numpy as np


_PALETTE: List[Tuple[int, int, int]] = [
    (57, 197, 187),
    (253, 121, 168),
    (116, 185, 255),
    (85, 239, 196),
    (255, 234, 167),
    (162, 155, 254),
    (232, 67, 147),
    (0, 184, 148),
]


def _to_numpy_array(value: Any) -> np.ndarray:
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        return value.detach().cpu().numpy()
    if hasattr(value, "cpu") and hasattr(value, "numpy"):
        return value.cpu().numpy()
    return np.asarray(value)


def _as_numpy_mask(mask: Any) -> np.ndarray:
    arr = _to_numpy_array(mask)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D mask, got shape {arr.shape}")
    if arr.dtype == np.bool_:
        return arr.astype(np.uint8)
    if np.issubdtype(arr.dtype, np.floating):
        return (arr >= 0.5).astype(np.uint8)
    return (arr > 0).astype(np.uint8)


def _color_for_index(index: int) -> Tuple[int, int, int]:
    return _PALETTE[index % len(_PALETTE)]


def draw_yolov8_mask(
    image: np.ndarray,
    mask: Any,
    *,
    color: Tuple[int, int, int],
    alpha: float = 0.3,
) -> np.ndarray:
    out = image.copy()
    mask_u8 = _as_numpy_mask(mask).astype(bool)
    if not np.any(mask_u8):
        return out
    color_arr = np.asarray(color, dtype=np.float32)
    blended = np.round((1.0 - float(alpha)) *
                       out[mask_u8] + float(alpha) * color_arr).astype(np.uint8)
    out[mask_u8] = blended
    return out


def draw_yolov8_contour(
    image: np.ndarray,
    mask: Any,
    *,
    color: Tuple[int, int, int],
    thickness: int = 1,
) -> np.ndarray:
    out = image.copy()
    mask_u8 = (_as_numpy_mask(mask) * 255).astype(np.uint8)
    contours, _ = cv2.findContours(
        mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cv2.drawContours(out, contours, -1, color, int(thickness))
    return out


def prediction_to_lists(prediction: Dict[str, Any]) -> Tuple[List[np.ndarray], List[float], List[int]]:
    masks_raw = prediction.get("masks", [])
    masks_arr = _to_numpy_array(masks_raw)
    if isinstance(masks_arr, np.ndarray) and masks_arr.ndim == 2:
        masks = [_as_numpy_mask(masks_arr)]
    else:
        masks = [_as_numpy_mask(mask) for mask in list(masks_raw)]

    scores = [float(x) for x in list(_to_numpy_array(prediction.get("scores", [])))]
    labels_raw = (
        prediction.get("labels")
        or prediction.get("category_ids")
        or prediction.get("class_ids")
        or []
    )
    labels = [int(x) for x in list(_to_numpy_array(labels_raw))]

    while len(scores) < len(masks):
        scores.append(0.0)
    while len(labels) < len(masks):
        labels.append(0)

    return masks, scores[: len(masks)], labels[: len(masks)]


def _draw_label(
    image: np.ndarray,
    *,
    text: str,
    mask: np.ndarray,
    color: Tuple[int, int, int],
) -> np.ndarray:
    out = image.copy()
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return out
    x, y = int(xs.min()), int(ys.min())
    cv2.putText(
        out,
        text,
        (x, max(y - 4, 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.35,
        color,
        1,
        cv2.LINE_AA,
    )
    return out


def visualize_predictions(
    *,
    image: np.ndarray,
    masks: Sequence[Any],
    scores: Sequence[float],
    labels: Sequence[int],
    class_names: Sequence[str] | None = None,
    score_threshold: float = 0.5,
    alpha: float = 0.3,
    show_labels: bool = False,
    show_contours: bool = True,
    contour_thickness: int = 1,
    show_masks: bool = True,
    output_path: str | None = None,
) -> np.ndarray:
    out = image.copy()
    class_names = list(class_names or [])
    for idx, (mask, score, label) in enumerate(zip(masks, scores, labels)):
        if float(score) < float(score_threshold):
            continue
        mask_u8 = _as_numpy_mask(mask)
        color = _color_for_index(idx)
        if show_masks:
            out = draw_yolov8_mask(out, mask_u8, color=color, alpha=alpha)
        if show_contours:
            out = draw_yolov8_contour(
                out, mask_u8, color=color, thickness=contour_thickness)
        if show_labels:
            label_name = class_names[label] if 0 <= int(
                label) < len(class_names) else str(int(label))
            out = _draw_label(
                out, text=f"{label_name}:{float(score):.2f}", mask=mask_u8, color=color)

    if output_path:
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
    return out


def _render_gt_panel(
    image: np.ndarray,
    masks: Iterable[Any],
    *,
    alpha: float,
) -> np.ndarray:
    out = image.copy()
    for idx, mask in enumerate(masks):
        color = _color_for_index(idx)
        out = draw_yolov8_mask(out, mask, color=color, alpha=alpha)
        out = draw_yolov8_contour(out, mask, color=color, thickness=1)
    return out


def render_gt_prediction_comparison(
    *,
    image: np.ndarray,
    gt_masks: Sequence[Any],
    prediction: Dict[str, Any],
    score_threshold: float = 0.5,
    alpha: float = 0.3,
) -> np.ndarray:
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError(
            f"image must be an HWC uint8 RGB array, got shape={image.shape}, dtype={image.dtype}"
        )
    for key in ("masks", "scores", "category_ids"):
        if key not in prediction:
            raise KeyError(f"prediction is missing required key: {key}")

    masks = list(prediction["masks"])
    scores = [float(value) for value in _to_numpy_array(prediction["scores"])]
    labels = [int(value) for value in _to_numpy_array(prediction["category_ids"])]
    if not (len(masks) == len(scores) == len(labels)):
        raise ValueError(
            "prediction masks, scores, and category_ids must have equal lengths"
        )

    gt_panel = _render_gt_panel(image, gt_masks, alpha=alpha)
    prediction_panel = visualize_predictions(
        image=image,
        masks=masks,
        scores=scores,
        labels=labels,
        score_threshold=score_threshold,
        alpha=alpha,
        show_labels=False,
        show_contours=True,
        contour_thickness=1,
        show_masks=True,
    )

    comparison = cv2.hconcat([gt_panel, prediction_panel])
    title_band = np.full((24, comparison.shape[1], 3), 255, dtype=np.uint8)
    cv2.putText(
        title_band,
        "GT",
        (8, 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (20, 20, 20),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        title_band,
        "Prediction",
        (image.shape[1] + 8, 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (20, 20, 20),
        1,
        cv2.LINE_AA,
    )
    return cv2.vconcat([title_band, comparison])


def save_evaluation_comparisons(
    *,
    output_dir: str | Path,
    step: int,
    batch: Dict[str, Any],
    outputs: Dict[str, Any],
    max_samples: int = 4,
    score_threshold: float = 0.5,
    alpha: float = 0.3,
) -> List[Path]:
    if step < 0:
        raise ValueError(f"step must be non-negative, got {step}")
    if max_samples <= 0:
        raise ValueError(f"max_samples must be positive, got {max_samples}")
    for key in ("images", "targets", "image_ids"):
        if key not in batch:
            raise KeyError(f"evaluation batch is missing required key: {key}")
    if "predictions" not in outputs:
        raise KeyError("evaluation outputs are missing required key: predictions")

    images = _to_numpy_array(batch["images"])
    if images.ndim != 4 or images.shape[1] != 3:
        raise ValueError(f"images must have shape [B, 3, H, W], got {images.shape}")
    if np.issubdtype(images.dtype, np.floating):
        if not np.isfinite(images).all() or images.min() < 0.0 or images.max() > 255.0:
            raise ValueError("floating evaluation images must be finite and in [0, 255]")
        images = np.rint(images).astype(np.uint8)
    elif images.dtype != np.uint8:
        raise TypeError(f"evaluation images must be float or uint8, got {images.dtype}")
    images = images.transpose(0, 2, 3, 1)

    targets = list(batch["targets"])
    predictions = list(outputs["predictions"])
    image_ids = [int(value) for value in _to_numpy_array(batch["image_ids"])]
    sample_count = len(images)
    if sample_count == 0:
        raise ValueError("evaluation visualization requires at least one sample")
    if not (sample_count == len(targets) == len(predictions) == len(image_ids)):
        raise ValueError(
            "images, targets, predictions, and image_ids must have equal lengths"
        )

    step_dir = Path(output_dir) / "visualizations" / f"step_{step:07d}"
    step_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: List[Path] = []
    for sample_index in range(min(sample_count, max_samples)):
        target = targets[sample_index]
        if "masks" not in target:
            raise KeyError("evaluation target is missing required key: masks")
        comparison = render_gt_prediction_comparison(
            image=images[sample_index],
            gt_masks=target["masks"],
            prediction=predictions[sample_index],
            score_threshold=score_threshold,
            alpha=alpha,
        )
        save_path = step_dir / f"sample_{sample_index:02d}_image_{image_ids[sample_index]}.png"
        written = cv2.imwrite(str(save_path), cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR))
        if not written:
            raise RuntimeError(f"Failed to write evaluation visualization: {save_path}")
        saved_paths.append(save_path)
    return saved_paths

def render_triptych_comparison(
    *,
    image: np.ndarray,
    gt_masks: Sequence[Any],
    magformer_prediction: Dict[str, Any],
    mask2former_prediction: Dict[str, Any],
    score_threshold: float = 0.5,
    alpha: float = 0.3,
    show_labels: bool = False,
    add_titles: bool = False,
) -> np.ndarray:
    gt_panel = _render_gt_panel(image, gt_masks, alpha=alpha)
    mag_masks, mag_scores, mag_labels = prediction_to_lists(
        magformer_prediction)
    mag_panel = visualize_predictions(
        image=image,
        masks=mag_masks,
        scores=mag_scores,
        labels=mag_labels,
        score_threshold=score_threshold,
        alpha=alpha,
        show_labels=show_labels,
    )
    mask_masks, mask_scores, mask_labels = prediction_to_lists(
        mask2former_prediction)
    mask_panel = visualize_predictions(
        image=image,
        masks=mask_masks,
        scores=mask_scores,
        labels=mask_labels,
        score_threshold=score_threshold,
        alpha=alpha,
        show_labels=show_labels,
    )

    stacked = cv2.hconcat([gt_panel, mag_panel, mask_panel])
    if not add_titles:
        return stacked

    title_band = np.full((24, stacked.shape[1], 3), 255, dtype=np.uint8)
    titles = ["GT", "MagFormer", "Mask2Former"]
    width = image.shape[1]
    for idx, title in enumerate(titles):
        x = idx * width + 8
        cv2.putText(title_band, title, (x, 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 20), 1, cv2.LINE_AA)
    return cv2.vconcat([title_band, stacked])
