import argparse
import csv
import json
import random
import shutil
from pathlib import Path


SPLIT_RATIOS = {
    "train": 0.70,
    "val": 0.15,
    "test": 0.10,
    "demo": 0.05,
}


def load_coco(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def pair_key(image):
    return str(image.get("folder")), str(image.get("frame_id")).zfill(5)


def safe_name(prefix, folder, frame_id):
    return f"{prefix}_{folder}_{frame_id}.jpg"


def split_pairs(pairs, seed):
    rng = random.Random(seed)
    shuffled = list(pairs)
    rng.shuffle(shuffled)

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


def prepare_output(output_root, keep_existing):
    resolved_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    for subdir in ["annotations", "rgb", "thermal"]:
        path = output_root / subdir
        resolved = path.resolve()
        if resolved_root not in resolved.parents and resolved != resolved_root:
            raise RuntimeError(f"Refusing to clean outside output root: {resolved}")
        if path.exists() and not keep_existing:
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)

    for modality in ["rgb", "thermal"]:
        for split in SPLIT_RATIOS:
            (output_root / modality / f"{split}_imgs").mkdir(parents=True, exist_ok=True)


def annotation_index(coco):
    by_image_id = {}
    for annotation in coco["annotations"]:
        by_image_id.setdefault(annotation["image_id"], []).append(annotation)
    return by_image_id


def build_split_coco(coco, images, annotations_by_image_id, output_names):
    out_images = []
    out_annotations = []
    next_ann_id = 0

    for new_image_id, image in enumerate(images):
        original_image_id = image["id"]
        new_image = dict(image)
        new_image["id"] = new_image_id
        new_image["original_id"] = original_image_id
        new_image["file_name"] = output_names[original_image_id]
        out_images.append(new_image)

        for annotation in annotations_by_image_id.get(original_image_id, []):
            new_annotation = dict(annotation)
            new_annotation["id"] = next_ann_id
            new_annotation["image_id"] = new_image_id
            new_annotation["original_id"] = annotation.get("id")
            out_annotations.append(new_annotation)
            next_ann_id += 1

    return {
        "images": out_images,
        "annotations": out_annotations,
        "categories": coco.get("categories", []),
    }


def copy_images(source_root, output_root, split, modality, images, output_names):
    img_dir = output_root / modality / f"{split}_imgs"
    for image in images:
        source = source_root / image["file_name"]
        if not source.exists():
            raise FileNotFoundError(source)
        destination = img_dir / output_names[image["id"]]
        shutil.copy2(source, destination)


def write_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)


def write_pairs_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def assert_no_pair_overlap(split_to_pairs):
    seen = {}
    for split, pairs in split_to_pairs.items():
        for pair in pairs:
            key = pair["key"]
            if key in seen:
                raise RuntimeError(f"Pair {key} appears in both {seen[key]} and {split}")
            seen[key] = split


def main():
    parser = argparse.ArgumentParser(
        description="Create synchronized paired RGB/thermal train/val/test/demo splits by folder + frame_id."
    )
    parser.add_argument("--rgb-json", default="data/annotations/rgb_keypoints.json", type=Path)
    parser.add_argument("--thermal-json", default="data/annotations/thermal_keypoints.json", type=Path)
    parser.add_argument("--source-root", default="datasets/keypoints", type=Path)
    parser.add_argument("--output-root", default="datasets/keypoints/paired_rgb_thermal", type=Path)
    parser.add_argument("--seed", default=1337, type=int)
    parser.add_argument("--keep-existing", action="store_true")
    args = parser.parse_args()

    rgb_coco = load_coco(args.rgb_json)
    thermal_coco = load_coco(args.thermal_json)

    rgb_by_key = {pair_key(image): image for image in rgb_coco["images"]}
    thermal_by_key = {pair_key(image): image for image in thermal_coco["images"]}
    shared_keys = sorted(set(rgb_by_key) & set(thermal_by_key), key=lambda key: (int(key[0]), int(key[1])))

    pairs = [
        {
            "key": key,
            "folder": key[0],
            "frame_id": key[1],
            "rgb_image": rgb_by_key[key],
            "thermal_image": thermal_by_key[key],
        }
        for key in shared_keys
    ]
    split_to_pairs = split_pairs(pairs, args.seed)
    assert_no_pair_overlap(split_to_pairs)

    prepare_output(args.output_root, args.keep_existing)

    rgb_annotations = annotation_index(rgb_coco)
    thermal_annotations = annotation_index(thermal_coco)
    summary = {}

    for split, split_pairs_ in split_to_pairs.items():
        rgb_images = [pair["rgb_image"] for pair in split_pairs_]
        thermal_images = [pair["thermal_image"] for pair in split_pairs_]

        rgb_names = {
            image["id"]: safe_name("rgb", image["folder"], str(image["frame_id"]).zfill(5))
            for image in rgb_images
        }
        thermal_names = {
            image["id"]: safe_name("thermal", image["folder"], str(image["frame_id"]).zfill(5))
            for image in thermal_images
        }

        copy_images(args.source_root, args.output_root, split, "rgb", rgb_images, rgb_names)
        copy_images(args.source_root, args.output_root, split, "thermal", thermal_images, thermal_names)

        rgb_split = build_split_coco(rgb_coco, rgb_images, rgb_annotations, rgb_names)
        thermal_split = build_split_coco(thermal_coco, thermal_images, thermal_annotations, thermal_names)

        write_json(args.output_root / "annotations" / f"rgb_{split}.json", rgb_split)
        write_json(args.output_root / "annotations" / f"thermal_{split}.json", thermal_split)

        pair_rows = []
        for pair_id, pair in enumerate(split_pairs_):
            rgb_image = pair["rgb_image"]
            thermal_image = pair["thermal_image"]
            pair_rows.append(
                {
                    "pair_id": pair_id,
                    "split": split,
                    "folder": pair["folder"],
                    "frame_id": pair["frame_id"],
                    "rgb_file": rgb_names[rgb_image["id"]],
                    "thermal_file": thermal_names[thermal_image["id"]],
                    "rgb_original_file": rgb_image["file_name"],
                    "thermal_original_file": thermal_image["file_name"],
                    "rgb_original_id": rgb_image["id"],
                    "thermal_original_id": thermal_image["id"],
                }
            )
        write_pairs_csv(args.output_root / "annotations" / f"pairs_{split}.csv", pair_rows)

        summary[split] = {
            "pairs": len(split_pairs_),
            "rgb_images": len(rgb_split["images"]),
            "thermal_images": len(thermal_split["images"]),
            "rgb_annotations": len(rgb_split["annotations"]),
            "thermal_annotations": len(thermal_split["annotations"]),
        }

    summary["total"] = {
        "rgb_images": len(rgb_by_key),
        "thermal_images": len(thermal_by_key),
        "paired_images": len(shared_keys),
        "rgb_unpaired": len(set(rgb_by_key) - set(thermal_by_key)),
        "thermal_unpaired": len(set(thermal_by_key) - set(rgb_by_key)),
    }
    write_json(args.output_root / "annotations" / "split_summary.json", summary)

    print("Saved paired RGB/thermal split:", args.output_root)
    for split in SPLIT_RATIOS:
        print(f"{split}: {summary[split]['pairs']} pairs")
    print("Total paired images:", summary["total"]["paired_images"])
    print("RGB-only images:", summary["total"]["rgb_unpaired"])
    print("Thermal-only images:", summary["total"]["thermal_unpaired"])


if __name__ == "__main__":
    main()
