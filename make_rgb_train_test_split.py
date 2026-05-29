import json
import random
import re
import shutil
from copy import deepcopy
from pathlib import Path

ANNOTATION_JSON = Path("data/annotations/rgb_keypoints.json")
SOURCE_ROOT = Path("datasets/keypoints")
OUTPUT_ROOT = Path("datasets/keypoints/coco_format")
ANN_DIR = OUTPUT_ROOT / "annotations"

SPLIT_RATIOS = {
    "train": 0.70,
    "val": 0.15,
    "test": 0.10,
    "demo": 0.05,
}
SEED = 42
CLEAN_OUTPUT_DIRS = True


def safe_output_name(file_name):
    """Keep source folder identity so repeated frame names do not collide."""
    normalized = file_name.replace("\\", "/").strip("/")
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", normalized)
    return safe


def split_images(images):
    shuffled = list(images)
    random.Random(SEED).shuffle(shuffled)

    total = len(shuffled)
    train_end = int(total * SPLIT_RATIOS["train"])
    val_end = train_end + int(total * SPLIT_RATIOS["val"])
    test_end = val_end + int(total * SPLIT_RATIOS["test"])

    return {
        "train": shuffled[:train_end],
        "val": shuffled[train_end:val_end],
        "test": shuffled[val_end:test_end],
        "demo": shuffled[test_end:],
    }


def prepare_output_dirs(splits):
    output_root = OUTPUT_ROOT.resolve()
    ANN_DIR.mkdir(parents=True, exist_ok=True)

    for split in splits:
        img_dir = OUTPUT_ROOT / f"{split}_imgs"
        resolved = img_dir.resolve()
        if output_root not in resolved.parents and resolved != output_root:
            raise RuntimeError(f"Refusing to clean outside output root: {resolved}")

        if CLEAN_OUTPUT_DIRS and img_dir.exists():
            shutil.rmtree(img_dir)
        img_dir.mkdir(parents=True, exist_ok=True)


def copy_split_images(split, images):
    img_dir = OUTPUT_ROOT / f"{split}_imgs"
    copied_images = []
    seen_outputs = {}
    missing = []

    for image in images:
        original_name = image["file_name"]
        src = SOURCE_ROOT / original_name
        if not src.exists():
            missing.append(original_name)
            continue

        new_name = safe_output_name(original_name)
        previous = seen_outputs.get(new_name)
        if previous is not None and previous != original_name:
            raise RuntimeError(
                f"Output filename collision: {previous} and {original_name} -> {new_name}"
            )
        seen_outputs[new_name] = original_name

        dst = img_dir / new_name
        shutil.copy2(src, dst)

        copied = deepcopy(image)
        copied["file_name"] = new_name
        copied_images.append(copied)

    if missing:
        preview = ", ".join(missing[:10])
        raise FileNotFoundError(
            f"{split} is missing {len(missing)} source images under {SOURCE_ROOT}: {preview}"
        )

    return copied_images


def save_coco(path, template, images, annotations):
    data = deepcopy(template)
    data["images"] = images
    data["annotations"] = annotations

    for category in data.get("categories", []):
        category["name"] = "cattle"

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)


def assert_no_overlap(split_to_images):
    split_to_names = {
        split: {image["file_name"] for image in images}
        for split, images in split_to_images.items()
    }

    for left_name, left_names in split_to_names.items():
        for right_name, right_names in split_to_names.items():
            if left_name >= right_name:
                continue
            overlap = left_names & right_names
            if overlap:
                examples = ", ".join(sorted(overlap)[:10])
                raise RuntimeError(
                    f"{left_name}/{right_name} overlap on {len(overlap)} files: {examples}"
                )


def main():
    with open(ANNOTATION_JSON, "r", encoding="utf-8") as f:
        coco = json.load(f)

    images = coco["images"]
    annotations = coco["annotations"]
    split_to_raw_images = split_images(images)

    prepare_output_dirs(split_to_raw_images)

    split_to_images = {}
    summary = {}
    for split, raw_images in split_to_raw_images.items():
        copied_images = copy_split_images(split, raw_images)
        image_ids = {image["id"] for image in copied_images}
        split_annotations = [
            ann for ann in annotations if ann["image_id"] in image_ids
        ]

        save_coco(ANN_DIR / f"{split}.json", coco, copied_images, split_annotations)
        split_to_images[split] = copied_images
        summary[split] = {
            "images": len(copied_images),
            "annotations": len(split_annotations),
            "image_dir": str(OUTPUT_ROOT / f"{split}_imgs"),
            "annotation": str(ANN_DIR / f"{split}.json"),
        }

    assert_no_overlap(split_to_images)

    with open(ANN_DIR / "split_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Created clean RGB train/val/test/demo split")
    for split, values in summary.items():
        print(
            f"{split}: images={values['images']}, "
            f"annotations={values['annotations']}, dir={values['image_dir']}"
        )
    print("No output filenames overlap between splits.")
    print("Summary:", ANN_DIR / "split_summary.json")


if __name__ == "__main__":
    main()
