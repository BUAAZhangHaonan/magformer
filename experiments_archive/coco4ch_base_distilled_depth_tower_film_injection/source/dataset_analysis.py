from pycocotools.coco import COCO
import json, os
import numpy as np
from collections import Counter

# Analyze the main dataset (1566 images)
print("=" * 60)
print("ANALYSIS OF 20260318_1K_1566 (current training dataset)")
print("=" * 60)
train_coco = COCO("magformer_datasets/20260318_1K_1566/annotations/instances_train.json")
print("Train images:", len(train_coco.getImgIds()))
print("Train annotations:", len(train_coco.getAnnIds()))
print("Train categories:", train_coco.loadCats(train_coco.getCatIds()))

val_coco = COCO("magformer_datasets/20260318_1K_1566/annotations/instances_val.json")
print("Val images:", len(val_coco.getImgIds()))
print("Val annotations:", len(val_coco.getAnnIds()))

# Image sizes
sizes = {}
for img_id in train_coco.getImgIds():
    img = train_coco.loadImgs(img_id)[0]
    key = (img["height"], img["width"])
    sizes[key] = sizes.get(key, 0) + 1
print("Train image sizes:", sizes)

# Object size distribution
areas = []
for ann_id in train_coco.getAnnIds():
    ann = train_coco.loadAnns(ann_id)[0]
    areas.append(ann["area"])
areas = np.array(areas)
print("Object area stats: min=%.0f, median=%.0f, mean=%.0f, max=%.0f" % (areas.min(), np.median(areas), areas.mean(), areas.max()))
print("Small (<32x32=1024): %d (%.1f%%)" % ((areas < 1024).sum(), (areas < 1024).mean()*100))
print("Medium (1024-9216): %d (%.1f%%)" % (((areas >= 1024) & (areas < 9216)).sum(), ((areas >= 1024) & (areas < 9216)).mean()*100))
print("Large (>9216): %d (%.1f%%)" % ((areas >= 9216).sum(), (areas >= 9216).mean()*100))

# Category distribution
cat_counts = Counter()
for ann_id in train_coco.getAnnIds():
    ann = train_coco.loadAnns(ann_id)[0]
    cat_counts[ann["category_id"]] += 1
cats = {c["id"]: c["name"] for c in train_coco.loadCats(train_coco.getCatIds())}
print("Category distribution (train):")
for cid, count in sorted(cat_counts.items(), key=lambda x: -x[1]):
    print("  %s (id=%s): %d" % (cats.get(cid, cid), cid, count))

# Objects per image
objs_per_img = []
for img_id in train_coco.getImgIds():
    ann_ids = train_coco.getAnnIds(imgIds=img_id)
    objs_per_img.append(len(ann_ids))
objs_per_img = np.array(objs_per_img)
print("Objects per image: min=%d, median=%.0f, mean=%.1f, max=%d" % (objs_per_img.min(), np.median(objs_per_img), objs_per_img.mean(), objs_per_img.max()))

# Bbox size distribution
bboxes = []
for ann_id in train_coco.getAnnIds():
    ann = train_coco.loadAnns(ann_id)[0]
    bboxes.append(ann["bbox"])
bboxes = np.array(bboxes)
widths = bboxes[:, 2]
heights = bboxes[:, 3]
print("Bbox width stats: min=%.1f, median=%.1f, mean=%.1f, max=%.1f" % (widths.min(), np.median(widths), widths.mean(), widths.max()))
print("Bbox height stats: min=%.1f, median=%.1f, mean=%.1f, max=%.1f" % (heights.min(), np.median(heights), heights.mean(), heights.max()))

# Now analyze the large dataset (32254 images)
print("\n" + "=" * 60)
print("ANALYSIS OF 20260318_1K_32254 (large dataset)")
print("=" * 60)
try:
    big_coco = COCO("magformer_datasets/20260318_1K_32254/annotations/instances_train.json")
    print("Train images:", len(big_coco.getImgIds()))
    print("Train annotations:", len(big_coco.getAnnIds()))
    print("Train categories:", big_coco.loadCats(big_coco.getCatIds()))
    
    big_val = COCO("magformer_datasets/20260318_1K_32254/annotations/instances_val.json")
    print("Val images:", len(big_val.getImgIds()))
    print("Val annotations:", len(big_val.getAnnIds()))
    
    big_test = COCO("magformer_datasets/20260318_1K_32254/annotations/instances_test.json")
    print("Test images:", len(big_test.getImgIds()))
    print("Test annotations:", len(big_test.getAnnIds()))
    
    # Image sizes
    big_sizes = {}
    for img_id in big_coco.getImgIds()[:100]:
        img = big_coco.loadImgs(img_id)[0]
        key = (img["height"], img["width"])
        big_sizes[key] = big_sizes.get(key, 0) + 1
    print("Sample image sizes (first 100):", big_sizes)
    
    # Category distribution
    big_cat_counts = Counter()
    for ann_id in big_coco.getAnnIds():
        ann = big_coco.loadAnns(ann_id)[0]
        big_cat_counts[ann["category_id"]] += 1
    big_cats = {c["id"]: c["name"] for c in big_coco.loadCats(big_coco.getCatIds())}
    print("Category distribution (big train):")
    for cid, count in sorted(big_cat_counts.items(), key=lambda x: -x[1]):
        print("  %s (id=%s): %d" % (big_cats.get(cid, cid), cid, count))
    
    # Object size distribution
    big_areas = []
    for ann_id in big_coco.getAnnIds():
        ann = big_coco.loadAnns(ann_id)[0]
        big_areas.append(ann["area"])
    big_areas = np.array(big_areas)
    print("Object area stats: min=%.0f, median=%.0f, mean=%.0f, max=%.0f" % (big_areas.min(), np.median(big_areas), big_areas.mean(), big_areas.max()))
    print("Small (<32x32=1024): %d (%.1f%%)" % ((big_areas < 1024).sum(), (big_areas < 1024).mean()*100))
    print("Medium (1024-9216): %d (%.1f%%)" % (((big_areas >= 1024) & (big_areas < 9216)).sum(), ((big_areas >= 1024) & (big_areas < 9216)).mean()*100))
    print("Large (>9216): %d (%.1f%%)" % ((big_areas >= 9216).sum(), (big_areas >= 9216).mean()*100))
    
    # Objects per image
    big_objs = []
    for img_id in big_coco.getImgIds():
        ann_ids = big_coco.getAnnIds(imgIds=img_id)
        big_objs.append(len(ann_ids))
    big_objs = np.array(big_objs)
    print("Objects per image: min=%d, median=%.0f, mean=%.1f, max=%d" % (big_objs.min(), np.median(big_objs), big_objs.mean(), big_objs.max()))
    
    # Sample image filenames
    sample_imgs = [big_coco.loadImgs(i)[0] for i in big_coco.getImgIds()[:3]]
    for img in sample_imgs:
        print("  Sample: file_name=%s, height=%s, width=%s" % (img["file_name"], img["height"], img["width"]))
except Exception as e:
    print("Error loading big dataset:", e)

# Also check 0831_1K
print("\n" + "=" * 60)
print("ANALYSIS OF 0831_1K")
print("=" * 60)
try:
    old_coco = COCO("magformer_datasets/0831_1K/annotations/instances_train.json")
    print("Train images:", len(old_coco.getImgIds()))
    print("Train annotations:", len(old_coco.getAnnIds()))
    print("Train categories:", old_coco.loadCats(old_coco.getCatIds()))
except Exception as e:
    print("Error:", e)
