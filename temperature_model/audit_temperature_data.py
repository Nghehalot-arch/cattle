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


def sequence_int(value):
    try:
        return int(value)
    except ValueError:
        return None


def nearest_raw_candidates(row, raw_keys, limit=3):
    seq = sequence_int(row["sequence_num"])
    if seq is None:
        return ""
    same_date = [
        key
        for key in raw_keys
        if key[0] == row["date"] and sequence_int(key[1]) is not None
    ]
    ranked = sorted(same_date, key=lambda key: abs(sequence_int(key[1]) - seq))
    return "|".join(f"{date}/{sequence}" for date, sequence in ranked[:limit])


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields = list(rows[0].keys())
    for row in rows[1:]:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Audit raw thermal temperature data availability.")
    parser.add_argument("--metadata", default="data/annotations/metadata.csv", type=Path)
    parser.add_argument("--raw-zip", default="data/thermal_raw.zip", type=Path)
    parser.add_argument("--thermal-annotations", default="data/annotations/thermal_keypoints.json", type=Path)
    parser.add_argument("--output-dir", default="data/temperature_outputs/temperature_data_audit_v1", type=Path)
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
    raw_keys = sorted(counts)
    labeled_audit_rows = []
    for row in temp_rows:
        key = (row["date"], row["sequence_num"])
        raw_frame_count = counts.get(key, 0)
        labeled_audit_rows.append(
            {
                "date": row["date"],
                "sequence_num": row["sequence_num"],
                "cow_tag": row["cow_tag"],
                "temperature_f": row["temperature_f"],
                "metadata_has_raw_thermal": row.get("has_raw_thermal", ""),
                "raw_frame_count": raw_frame_count,
                "usable_for_temperature_model": "yes" if raw_frame_count else "no",
                "nearest_raw_same_date": "" if raw_frame_count else nearest_raw_candidates(row, raw_keys),
            }
        )
    missing_rows = [row for row in labeled_audit_rows if row["usable_for_temperature_model"] == "no"]
    raw_sequence_rows = [
        {
            "date": date,
            "sequence_num": sequence_num,
            "raw_frame_count": counts[(date, sequence_num)],
            "min_frame_id": min(raw_frame_ids[(date, sequence_num)]),
            "max_frame_id": max(raw_frame_ids[(date, sequence_num)]),
        }
        for date, sequence_num in raw_keys
    ]
    write_csv(args.output_dir / "labeled_raw_audit.csv", labeled_audit_rows)
    write_csv(args.output_dir / "missing_labeled_raw.csv", missing_rows)
    write_csv(args.output_dir / "raw_sequence_counts.csv", raw_sequence_rows)

    print("Metadata rows:", len(rows))
    print("Rows with temperature_f:", len(temp_rows))
    print("Usable rows with temperature_f and raw TIFF:", len(usable_rows))
    print("Missing labeled rows without raw TIFF:", len(missing_rows))
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
    if missing_rows:
        print("Missing labeled raw matches:")
        for row in missing_rows:
            print(
                "  {}/{} cow={} temp_f={} metadata_has_raw={} nearest_raw={}".format(
                    row["date"],
                    row["sequence_num"],
                    row["cow_tag"],
                    row["temperature_f"],
                    row["metadata_has_raw_thermal"],
                    row["nearest_raw_same_date"],
                )
            )
    print("Audit CSV:", args.output_dir)

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
