import json
from pathlib import Path

files = [
    Path("datasets/keypoints/coco_format/annotations/train.json"),
    Path("datasets/keypoints/coco_format/annotations/test.json"),
]

for file in files:
    with open(file, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "categories" in data:
        for cat in data["categories"]:
            cat["name"] = "cattle"

    with open(file, "w", encoding="utf-8") as f:
        json.dump(data, f)

    print("Fixed:", file)

print("Done. Category name is now cattle.")