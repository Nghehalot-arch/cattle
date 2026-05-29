import argparse
import csv
import zipfile
from pathlib import Path


def read_mapping(path, min_score):
    best_by_folder = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if not row["temperature_f"] or not row["mean_score"]:
                continue
            score = float(row["mean_score"])
            if score < min_score:
                continue
            folder = row["folder"]
            previous = best_by_folder.get(folder)
            if previous is None or score > float(previous["mean_score"]):
                best_by_folder[folder] = row
    return best_by_folder


def read_pairs(paired_root):
    rows = []
    annotations = paired_root / "annotations"
    for split in ("train", "val", "test", "demo"):
        path = annotations / f"pairs_{split}.csv"
        if not path.exists():
            continue
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                row["split"] = split
                rows.append(row)
    return rows


def raw_tiff_name(date, sequence_num, frame_id):
    return f"thermal_raw/{date}/{sequence_num}_Video_Frame_{int(frame_id)}.tiff"


def main():
    parser = argparse.ArgumentParser(
        description="Create RGB + processed thermal + raw TIFF + rectal temperature triples from synced pairs."
    )
    parser.add_argument("--paired-root", default="datasets/keypoints/paired_rgb_thermal", type=Path)
    parser.add_argument("--mapping", default="data/temperature_outputs/processed_raw_mapping.csv", type=Path)
    parser.add_argument("--raw-zip", default="data/thermal_raw.zip", type=Path)
    parser.add_argument("--output", default="data/temperature_outputs/rgb_thermal_temperature_triples.csv", type=Path)
    parser.add_argument("--min-score", default=0.15, type=float)
    args = parser.parse_args()

    mapping = read_mapping(args.mapping, args.min_score)
    pairs = read_pairs(args.paired_root)
    with zipfile.ZipFile(args.raw_zip) as zf:
        raw_names = set(zf.namelist())

    triples = []
    for pair in pairs:
        map_row = mapping.get(pair["folder"])
        if not map_row:
            continue
        frame_id = int(pair["frame_id"])
        raw_tiff = raw_tiff_name(map_row["date"], map_row["sequence_num"], pair["frame_id"])
        if raw_tiff not in raw_names:
            continue

        triples.append(
            {
                "split": pair["split"],
                "folder": pair["folder"],
                "frame_id": pair["frame_id"],
                "rgb_file": pair["rgb_file"],
                "thermal_file": pair["thermal_file"],
                "rgb_original_file": pair["rgb_original_file"],
                "thermal_original_file": pair["thermal_original_file"],
                "raw_date": map_row["date"],
                "raw_sequence_num": map_row["sequence_num"],
                "raw_tiff": raw_tiff,
                "cow_tag": map_row["cow_tag"],
                "temperature_f": map_row["temperature_f"],
                "mapping_mean_score": map_row["mean_score"],
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        if triples:
            writer = csv.DictWriter(f, fieldnames=list(triples[0].keys()))
            writer.writeheader()
            writer.writerows(triples)

    counts = {}
    for row in triples:
        counts[row["split"]] = counts.get(row["split"], 0) + 1
    print("Saved:", args.output)
    print("Triple rows:", len(triples))
    print("By split:", counts)
    print("Accepted folder mappings:")
    for folder, row in sorted(mapping.items(), key=lambda item: int(item[0])):
        print(
            "  folder={} -> {}/{} temp={} score={}".format(
                folder,
                row["date"],
                row["sequence_num"],
                row["temperature_f"],
                row["mean_score"],
            )
        )


if __name__ == "__main__":
    main()
