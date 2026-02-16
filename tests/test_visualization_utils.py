import numpy as np

from magformer.utils.visualization import (
    draw_yolov8_mask,
    draw_yolov8_contour,
    visualize_predictions as visualize_yolov8_predictions,
)


def test_draw_yolov8_mask_only_changes_mask_region():
    image = np.array(
        [
            [[10, 20, 30], [40, 50, 60]],
            [[70, 80, 90], [100, 110, 120]],
        ],
        dtype=np.uint8,
    )
    mask = np.array([[1, 0], [0, 1]], dtype=np.uint8)
    color = (200, 100, 50)
    alpha = 0.3

    result = draw_yolov8_mask(image, mask, color=color, alpha=alpha)

    # Non-mask pixels must stay exactly unchanged.
    assert np.array_equal(result[0, 1], image[0, 1])
    assert np.array_equal(result[1, 0], image[1, 0])

    expected_00 = np.round((1.0 - alpha) * image[0, 0] + alpha * np.array(color)).astype(np.uint8)
    expected_11 = np.round((1.0 - alpha) * image[1, 1] + alpha * np.array(color)).astype(np.uint8)
    assert np.array_equal(result[0, 0], expected_00)
    assert np.array_equal(result[1, 1], expected_11)


def test_draw_yolov8_mask_accepts_bool_prob_and_u8_masks():
    image = np.full((2, 2, 3), 100, dtype=np.uint8)
    color = (250, 0, 0)
    alpha = 0.4

    mask_bool = np.array([[True, False], [False, True]], dtype=bool)
    mask_prob = np.array([[0.9, 0.2], [0.3, 0.8]], dtype=np.float32)
    mask_u8 = np.array([[255, 0], [0, 255]], dtype=np.uint8)

    out_bool = draw_yolov8_mask(image, mask_bool, color=color, alpha=alpha)
    out_prob = draw_yolov8_mask(image, mask_prob, color=color, alpha=alpha)
    out_u8 = draw_yolov8_mask(image, mask_u8, color=color, alpha=alpha)

    assert np.array_equal(out_bool, out_prob)
    assert np.array_equal(out_bool, out_u8)


def test_visualize_predictions_assigns_distinct_colors_per_instance():
    image = np.zeros((6, 8, 3), dtype=np.uint8)
    mask_a = np.zeros((6, 8), dtype=np.uint8)
    mask_b = np.zeros((6, 8), dtype=np.uint8)
    mask_a[1:3, 1:3] = 1
    mask_b[3:5, 5:7] = 1

    out = visualize_yolov8_predictions(
        image=image,
        masks=[mask_a, mask_b],
        scores=[0.9, 0.8],
        labels=[0, 0],  # same category, still needs distinct instance colors
        score_threshold=0.5,
        alpha=1.0,
        show_labels=False,
        show_contours=False,
        show_masks=True,
    )

    color_a = out[1, 1]
    color_b = out[3, 5]
    assert not np.array_equal(color_a, color_b)


def test_draw_yolov8_contour_thicker_line_changes_more_pixels():
    image = np.zeros((12, 12, 3), dtype=np.uint8)
    mask = np.zeros((12, 12), dtype=np.uint8)
    mask[3:9, 3:9] = 1

    out_thin = draw_yolov8_contour(image, mask, color=(255, 0, 0), thickness=1)
    out_thick = draw_yolov8_contour(image, mask, color=(255, 0, 0), thickness=2)

    changed_thin = (out_thin != image).any(axis=2).sum()
    changed_thick = (out_thick != image).any(axis=2).sum()
    assert changed_thick > changed_thin
