import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from common import RAW_NAME_RE


def raw_counts(raw_zip):
    import zipfile

    counts = Counter()
    frame_ids = defaultdict(set)
    with zipfile.ZipFile(raw_zip) as zf:
        for name in zf.namelist():
            match = RAW_NAME_RE.match(name)
            if not match:
                continue
            date, sequence_num, frame_id = match.groups()
            counts[(date, sequence_num)] += 1
            frame_ids[(date, sequence_num)].add(int(frame_id))
    return counts, frame_ids


def main():
    parser = argparse.ArgumentParser(description="Audit raw thermal temperature data availability.")
    parser.add_argument("--metadata", default="data/annotations/metadata.csv", type=Path)
    parser.add_argument("--raw-zip", default="data/thermal_raw.zip", type=Path)
    parser.add_argument("--thermal-annotations", default="data/annotations/thermal_keypoints.json", type=Path)
    args = parser.parse_args()

    with open(args.metadata, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    counts, raw_frame_ids = raw_counts(args.raw_zip)

    temp_rows = [row for row in rows if row.get("temperature_f")]
    usable_rows = [
        row
        for row in rows
        if row.get("temperature_f") and counts.get((row["date"], row["sequence_num"]), 0)
    ]

    print("Metadata rows:", len(rows))
    print("Rows with temperature_f:", len(temp_rows))
    print("Usable rows with temperature_f and raw TIFF:", len(usable_rows))
    for row in usable_rows:
        key = (row["date"], row["sequence_num"])
        print(
            "  {}/{} cow={} temp_f={} raw_frames={}".format(
                row["date"],
                row["sequence_num"],
                row["cow_tag"],
                row["temperature_f"],
                counts[key],
            )
        )

    if not args.thermal_annotations.exists():
        return

    with open(args.thermal_annotations, "r", encoding="utf-8") as f:
        coco = json.load(f)
    by_folder = defaultdict(set)
    for image in coco["images"]:
        by_folder[str(image.get("folder"))].add(int(image.get("frame_id")))

    by_sequence = defaultdict(list)
    for row in rows:
        by_sequence[row["sequence_num"].lstrip("0") or "0"].append(row)

    print()
    print("Annotated thermal JPG folder candidates:")
    for folder, frames in sorted(by_folder.items(), key=lambda item: int(item[0])):
        print(f"  folder={folder} annotated={len(frames)} range={min(frames)}-{max(frames)}")
        for row in by_sequence.get(folder, []):
            key = (row["date"], row["sequence_num"])
            overlap = len(frames & raw_frame_ids.get(key, set()))
            print(
                "    {}/{} cow={} temp_f={} raw_frames={} same_frame_ids={}".format(
                    row["date"],
                    row["sequence_num"],
                    row["cow_tag"],
                    row["temperature_f"] or "NULL",
                    counts.get(key, 0),
                    overlap,
                )
            )


if __name__ == "__main__":
    main()
