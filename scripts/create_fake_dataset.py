#!/usr/bin/env python3
"""
创建假数据集用于 MAGFormer 烟雾测试。

生成 1024x1024 的 fake RGB + depth 图像和 COCO 格式标注。

用法:
    python scripts/create_fake_dataset.py --output _fake_dataset --num-images 4
"""

import argparse
import json
import os
import numpy as np
from pathlib import Path
from PIL import Image


def create_fake_dataset(output_dir: str, num_images: int = 4, img_size: int = 1024):
    root = Path(output_dir)

    # 创建目录结构
    for split in ("train", "val"):
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "depth" / "depth_npy" / split).mkdir(parents=True, exist_ok=True)
    (root / "annotations").mkdir(parents=True, exist_ok=True)

    for split in ("train", "val"):
        images_info = []
        annotations = []
        ann_id = 1

        n = num_images if split == "train" else max(1, num_images // 2)

        for i in range(n):
            img_id = i + 1
            fname = f"{img_id:06d}.png"

            # --- RGB 图像 (随机噪声 + 彩色方块) ---
            img = np.random.randint(0, 50, (img_size, img_size, 3), dtype=np.uint8)

            # --- 深度图 (随机 float32) ---
            depth = np.random.uniform(0.2, 0.8, (img_size, img_size)).astype(np.float32)

            # 在图像上放置 2~4 个随机矩形 "物体"
            num_objs = np.random.randint(2, 5)
            for j in range(num_objs):
                # 随机位置和大小
                obj_w = np.random.randint(80, 300)
                obj_h = np.random.randint(80, 300)
                x1 = np.random.randint(0, img_size - obj_w)
                y1 = np.random.randint(0, img_size - obj_h)
                x2 = x1 + obj_w
                y2 = y1 + obj_h

                # 画到 RGB 上
                color = np.random.randint(100, 255, 3)
                img[y1:y2, x1:x2] = color

                # 画到 depth 上
                depth[y1:y2, x1:x2] = np.random.uniform(0.05, 0.3)

                # COCO polygon 分割 (就用矩形的4个角)
                seg = [float(x1), float(y1),
                       float(x2), float(y1),
                       float(x2), float(y2),
                       float(x1), float(y2)]

                area = float(obj_w * obj_h)
                bbox = [float(x1), float(y1), float(obj_w), float(obj_h)]

                annotations.append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": 1,
                    "segmentation": [seg],
                    "bbox": bbox,
                    "area": area,
                    "iscrowd": 0,
                })
                ann_id += 1

            # 保存 RGB
            Image.fromarray(img).save(str(root / "images" / split / fname))

            # 保存 Depth
            np.save(str(root / "depth" / "depth_npy" / split / f"{img_id:06d}.npy"), depth)

            images_info.append({
                "id": img_id,
                "file_name": fname,
                "height": img_size,
                "width": img_size,
            })

        # 保存 COCO JSON
        coco_json = {
            "images": images_info,
            "annotations": annotations,
            "categories": [{"id": 1, "name": "component"}],
        }

        ann_path = root / "annotations" / f"instances_{split}.json"
        with open(ann_path, "w") as f:
            json.dump(coco_json, f, indent=2)

        print(f"[{split}] {n} images, {len(annotations)} annotations -> {ann_path}")

    print(f"\nFake dataset created at: {root.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="_fake_dataset", help="Output directory")
    parser.add_argument("--num-images", type=int, default=4, help="Number of training images")
    parser.add_argument("--img-size", type=int, default=1024, help="Image size")
    args = parser.parse_args()
    create_fake_dataset(args.output, args.num_images, args.img_size)
