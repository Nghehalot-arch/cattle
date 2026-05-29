import json
from pathlib import Path

src = Path("datasets/keypoints/coco_format/annotations/train.json")
dst = Path("datasets/keypoints/coco_format/annotations/train_small.json")

with open(src, "r", encoding="utf-8") as f:
    data = json.load(f)

# Keep only first 20 images
small_images = data["images"][:20]
small_ids = {img["id"] for img in small_images}
small_annotations = [ann for ann in data["annotations"] if ann["image_id"] in small_ids]

data["images"] = small_images
data["annotations"] = small_annotations

with open(dst, "w", encoding="utf-8") as f:
    json.dump(data, f)

print("Created:", dst)
print("Images:", len(small_images))
print("Annotations:", len(small_annotations))