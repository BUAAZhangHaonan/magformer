"""4-channel (RGB-D concat) LSJ DatasetMapper for detectron2 / Mask2Former.

Geometry (scale 0.1-2.0 + hflip + fixed-size 1024 crop) mirrors detectron2's
Mask2Former LSJ augmentation; the depth npy is warped with the same transform
(bilinear) and appended as a 4th channel: metric depth * 255.

Known deviation from the official M2F LSJ mapper (intentional, all E24 arms
share it): FixedSizeCrop pads with 0 instead of image_net_std, and padded
depth reads as 0 m (z-scored far outlier). Do NOT change without re-running:
every E24 number was produced with this behavior.
"""
import copy
import os

import numpy as np
import torch

from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T

DEPTH_MEAN = 0.492  # meters
DEPTH_STD = 0.059

RGB_PIXEL_MEAN = [123.675, 116.280, 103.530]
RGB_PIXEL_STD = [58.395, 57.120, 57.375]
# 4th channel is depth*255, so mean/std in the same 0-255 scale
RGBD_PIXEL_MEAN = RGB_PIXEL_MEAN + [DEPTH_MEAN * 255.0]
RGBD_PIXEL_STD = RGB_PIXEL_STD + [DEPTH_STD * 255.0]


def build_lsj_augs():
    return T.AugmentationList(
        [
            T.ResizeScale(0.1, 2.0, target_height=1024, target_width=1024),
            T.RandomFlip(horizontal=True),
            T.FixedSizeCrop((1024, 1024), pad_value=0),
        ]
    )


class RGBDConcatMapper:
    """Produces records with 'image' = (4, H, W) float32 on a 0-255 scale."""

    def __init__(self, cfg, is_train=True):
        self.is_train = is_train
        # test-time: deterministic resize only (LSJ random flip/crop is train-only)
        self.aug = build_lsj_augs() if is_train else T.ResizeShortestEdge(int(os.environ.get("RGBD_TEST_SHORT", 800)), int(os.environ.get("RGBD_TEST_MAX", 1333)))
        self.img_format = cfg.INPUT.FORMAT
        self.instance_mask_format = cfg.INPUT.MASK_FORMAT

    def __call__(self, dataset_dict):
        dataset_dict = copy.deepcopy(dataset_dict)
        image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        depth = np.load(dataset_dict["depth_file_name"]).astype(np.float32)  # H, W, meters

        aug_input = T.AugInput(image)
        transforms = self.aug(aug_input)
        image = aug_input.image
        # same geometric transform on depth, bilinear interpolation
        from detectron2.data.transforms.transform import ResizeTransform

        tlist = transforms.transforms if hasattr(transforms, "transforms") else [transforms]
        for t in tlist:
            if isinstance(t, ResizeTransform):
                # bilinear resize for smooth metric depth
                import PIL.Image as PILImage

                depth = t.apply_image(depth, interp=PILImage.BILINEAR)
            else:
                depth = t.apply_image(depth)

        image_shape = image.shape[:2]  # h, w
        # 4th channel: metric depth scaled to 0-255 (matches RGBD_PIXEL_MEAN/STD)
        img4 = np.concatenate([image, depth[:, :, None] * 255.0], axis=2)  # H, W, 4
        # depth-corruption hooks:
        #  eval:  RGBD_CORRUPT=<mode> fixed deterministic mode (mode: neutral|black|noise<k>)
        #  train: RGBD_CORRUPT_TRAIN=<p> stochastic, each sample corrupted with prob p,
        #         mode drawn uniformly from {neutral, black, noise1, noise2}
        _mode = "" if self.is_train else os.environ.get("RGBD_CORRUPT", "")
        if not _mode and self.is_train and os.environ.get("RGBD_CORRUPT_TRAIN", ""):
            if np.random.rand() < float(os.environ["RGBD_CORRUPT_TRAIN"]):
                _mode = str(np.random.choice(["neutral", "black", "noise1", "noise2"]))
        if _mode:
            rng = np.random.RandomState(1000 + int(dataset_dict.get("image_id", 0)) % 1000)
            if _mode == "neutral":       # z-scored -> 0 (sensor missing)
                img4[:, :, 3] = 125.628
            elif _mode == "black":        # z-scored -> -8.3 (dead channel)
                img4[:, :, 3] = 0.0
            elif _mode.startswith("noise"):  # gaussian noise, k * depth-std
                k = float(_mode[5:])
                img4[:, :, 3] = img4[:, :, 3] + rng.normal(0.0, k * 15.157, img4.shape[:2])
        dataset_dict["image"] = torch.as_tensor(
            np.ascontiguousarray(img4.transpose(2, 0, 1)).astype(np.float32)
        )

        if not self.is_train:
            dataset_dict.pop("annotations", None)
            return dataset_dict

        annos = [
            utils.transform_instance_annotations(obj, transforms, image_shape)
            for obj in dataset_dict.pop("annotations")
            if obj.get("iscrowd", 0) == 0
        ]
        # single-class dataset: COCO category id 1 -> contiguous class 0
        # (class 1 in the 2-way head is the no-object class; passing raw ids
        # would train every instance AS no-object)
        for obj in annos:
            if obj["category_id"] != 0:
                obj["category_id"] = 0
        instances = utils.annotations_to_instances(annos, image_shape)
        instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
        instances = utils.filter_empty_instances(instances)
        # M2F expects gt_masks as a plain bit-mask tensor
        from mask2former.data.dataset_mappers.coco_instance_new_baseline_dataset_mapper import convert_coco_poly_to_mask

        h, w = instances.image_size
        if hasattr(instances, "gt_masks"):
            gt_masks = instances.gt_masks
            gt_masks = convert_coco_poly_to_mask(gt_masks.polygons, h, w)
            instances.gt_masks = gt_masks
        dataset_dict["instances"] = instances
        return dataset_dict
