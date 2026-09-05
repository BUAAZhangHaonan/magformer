"""Register the 20260318_1K_32254 electronic-components dataset (RGB-D) for detectron2.

Each record carries `depth_file_name` pointing to a float32 metric-depth npy.
"""
import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_coco_json

DATASET_ROOT = "/home/hdd1/wanghaoran/magformer/magformer_datasets/20260318_1K_32254"

_SPLITS = {
    "ecc20260318_1k_rgbd_train": "train",
    "ecc20260318_1k_rgbd_val": "val",
}


def _register():
    for name, split in _SPLITS.items():
        image_dir = os.path.join(DATASET_ROOT, "images", split)
        json_file = os.path.join(
            DATASET_ROOT, "annotations", f"instances_{split}.validated.json"
        )
        depth_dir = os.path.join(DATASET_ROOT, "depth", "depth_npy", split)
        MetadataCatalog.get(name).set(
            thing_classes=["component"],
            json_file=json_file,
            image_root=image_dir,
            depth_root=depth_dir,
            evaluator_type="coco",
        )

        def _load(json_file=json_file, image_dir=image_dir, depth_dir=depth_dir, name=name):
            records = load_coco_json(json_file, image_root=image_dir, dataset_name=name)
            for r in records:
                stem = os.path.splitext(os.path.basename(r["file_name"]))[0]
                r["depth_file_name"] = os.path.join(depth_dir, stem + ".npy")
            return records

        DatasetCatalog.register(name, _load)


_register()


# --- E24 val subset (seed-42 sample of 1000 val images) ---
def _register_subset():
    name = "ecc20260318_1k_rgbd_val_subset1000"
    image_dir = os.path.join(DATASET_ROOT, "images", "val")
    json_file = os.path.join(DATASET_ROOT, "annotations", "instances_val_subset1000.json")
    depth_dir = os.path.join(DATASET_ROOT, "depth", "depth_npy", "val")
    MetadataCatalog.get(name).set(
        thing_classes=["component"],
        json_file=json_file,
        image_root=image_dir,
        depth_root=depth_dir,
        evaluator_type="coco",
    )

    def _load(json_file=json_file, image_dir=image_dir, depth_dir=depth_dir, name=name):
        records = load_coco_json(json_file, image_root=image_dir, dataset_name=name)
        for r in records:
            stem = os.path.splitext(os.path.basename(r["file_name"]))[0]
            r["depth_file_name"] = os.path.join(depth_dir, stem + ".npy")
        return records

    DatasetCatalog.register(name, _load)


_register_subset()
