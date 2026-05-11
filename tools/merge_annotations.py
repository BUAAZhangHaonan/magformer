#!/usr/bin/env python3
"""Merge multiple COCO annotation JSONs into a single annotation file."""

import json
import os


def merge_annotations():
    base_dir = "/home/hdd3/zhanghaonan/magformer/magformer_datasets/20260318_1K_1566/annotations"
    splits = ["train", "val", "test"]
    input_files = [os.path.join(base_dir, "instances_{}.json".format(s)) for s in splits]
    output_file = os.path.join(base_dir, "instances_all.json")

    merged_images = []
    merged_annotations = []
    current_image_id = 1
    current_ann_id = 1
    image_id_map = {}

    for split_name, input_file in zip(splits, input_files):
        print("Reading {}: {}".format(split_name, input_file))
        with open(input_file, "r") as f:
            data = json.load(f)

        images = data["images"]
        annotations = data["annotations"]
        print("  Images: {}, Annotations: {}".format(len(images), len(annotations)))

        for img in images:
            old_id = img["id"]
            image_id_map[old_id] = current_image_id
            new_img = dict(img)
            new_img["id"] = current_image_id
            merged_images.append(new_img)
            current_image_id += 1

        for ann in annotations:
            new_ann = dict(ann)
            new_ann["id"] = current_ann_id
            new_ann["image_id"] = image_id_map[ann["image_id"]]
            merged_annotations.append(new_ann)
            current_ann_id += 1

        image_id_map = {}

    merged = {
        "images": merged_images,
        "annotations": merged_annotations,
        "categories": [{"id": 1, "name": "component", "supercategory": "component"}],
    }

    print("\nWriting merged file: {}".format(output_file))
    with open(output_file, "w") as f:
        json.dump(merged, f)

    # Verification
    print("\n=== Verification ===")
    print("Total images: {}".format(len(merged_images)))
    print("Total annotations: {}".format(len(merged_annotations)))
    print("Categories: {}".format(merged["categories"]))

    print("\nSample images (first 3):")
    for img in merged_images[:3]:
        print("  {}".format(img))

    print("\nSample annotations (first 3):")
    for ann in merged_annotations[:3]:
        print("  id={}, image_id={}, bbox={}, area={}".format(
            ann["id"], ann["image_id"], ann["bbox"], ann["area"]))

    # Verify ID ranges
    img_ids = [img["id"] for img in merged_images]
    ann_ids = [ann["id"] for ann in merged_annotations]
    print("\nImage ID range: {} - {}".format(min(img_ids), max(img_ids)))
    print("Annotation ID range: {} - {}".format(min(ann_ids), max(ann_ids)))
    print("Image IDs unique: {}".format(len(img_ids) == len(set(img_ids))))
    print("Annotation IDs unique: {}".format(len(ann_ids) == len(set(ann_ids))))


if __name__ == "__main__":
    merge_annotations()
