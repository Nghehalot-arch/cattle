import argparse
import json
import random
import re
import shutil
from copy import deepcopy
from pathlib import Path


SPLIT_RATIOS = {
    "train": 0.70,
    "val": 0.15,
    "test": 0.10,
    "demo": 0.05,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Create clean COCO keypoint splits.")
    parser.add_argument("--annotation-json", required=True, type=Path)
    parser.add_argument("--source-root", default=Path("datasets/keypoints"), type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--keep-existing", action="store_true")
    return parser.parse_args()


def safe_output_name(file_name):
    normalized = file_name.replace("\\", "/").strip("/")
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", normalized)


def split_images(images, seed):
    shuffled = list(images)
    random.Random(seed).shuffle(shuffled)

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


def prepare_output_dirs(output_root, splits, keep_existing):
    resolved_root = output_root.resolve()
    annotations_dir = output_root / "annotations"
    annotations_dir.mkdir(parents=True, exist_ok=True)

    for split in splits:
        img_dir = output_root / f"{split}_imgs"
        resolved = img_dir.resolve()
        if resolved_root not in resolved.parents and resolved != resolved_root:
            raise RuntimeError(f"Refusing to clean outside output root: {resolved}")

        if not keep_existing and img_dir.exists():
            shutil.rmtree(img_dir)
        img_dir.mkdir(parents=True, exist_ok=True)


def copy_split_images(split, images, source_root, output_root):
    img_dir = output_root / f"{split}_imgs"
    copied_images = []
    seen_outputs = {}
    missing = []

    for image in images:
        original_name = image["file_name"]
        src = source_root / original_name
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

        shutil.copy2(src, img_dir / new_name)

        copied = deepcopy(image)
        copied["file_name"] = new_name
        copied_images.append(copied)

    if missing:
        preview = ", ".join(missing[:10])
        raise FileNotFoundError(
            f"{split} is missing {len(missing)} source images under {source_root}: {preview}"
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

    split_names = sorted(split_to_names)
    for index, left_name in enumerate(split_names):
        for right_name in split_names[index + 1:]:
            overlap = split_to_names[left_name] & split_to_names[right_name]
            if overlap:
                examples = ", ".join(sorted(overlap)[:10])
                raise RuntimeError(
                    f"{left_name}/{right_name} overlap on {len(overlap)} files: {examples}"
                )


def main():
    args = parse_args()
    annotations_dir = args.output_root / "annotations"

    with open(args.annotation_json, "r", encoding="utf-8") as f:
        coco = json.load(f)

    split_to_raw_images = split_images(coco["images"], args.seed)
    prepare_output_dirs(args.output_root, split_to_raw_images, args.keep_existing)

    split_to_images = {}
    summary = {}
    for split, raw_images in split_to_raw_images.items():
        copied_images = copy_split_images(
            split, raw_images, args.source_root, args.output_root
        )
        image_ids = {image["id"] for image in copied_images}
        split_annotations = [
            ann for ann in coco["annotations"] if ann["image_id"] in image_ids
        ]

        save_coco(annotations_dir / f"{split}.json", coco, copied_images, split_annotations)
        split_to_images[split] = copied_images
        summary[split] = {
            "images": len(copied_images),
            "annotations": len(split_annotations),
            "image_dir": str(args.output_root / f"{split}_imgs"),
            "annotation": str(annotations_dir / f"{split}.json"),
        }

    assert_no_overlap(split_to_images)

    with open(annotations_dir / "split_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Created clean keypoint split")
    for split, values in summary.items():
        print(
            f"{split}: images={values['images']}, "
            f"annotations={values['annotations']}, dir={values['image_dir']}"
        )
    print("No output filenames overlap between splits.")
    print("Summary:", annotations_dir / "split_summary.json")


if __name__ == "__main__":
    main()
