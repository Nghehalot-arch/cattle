import argparse
import csv
from pathlib import Path

from common import write_csv


JOIN_KEYS = ("date", "sequence_num")
KEEP_KEYS = {
    "date",
    "sequence_num",
    "cow_tag",
    "temperature_f",
    "raw_frame_count",
    "sampled_frame_count",
    "detected_frame_count",
}


def read_rows(path, prefix):
    rows = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = tuple(row[name] for name in JOIN_KEYS)
            parsed = {}
            for name, value in row.items():
                if name in KEEP_KEYS:
                    parsed[name] = value
                else:
                    parsed[f"{prefix}_{name}"] = value
            rows[key] = parsed
    return rows


def main():
    parser = argparse.ArgumentParser(description="Combine multiple sequence-level temperature feature CSV files.")
    parser.add_argument("--left", required=True, type=Path)
    parser.add_argument("--right", required=True, type=Path)
    parser.add_argument("--left-prefix", default="raw")
    parser.add_argument("--right-prefix", default="roi")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    left = read_rows(args.left, args.left_prefix)
    right = read_rows(args.right, args.right_prefix)
    shared_keys = sorted(set(left) & set(right))
    if not shared_keys:
        raise RuntimeError("No matching date/sequence rows between feature files")

    combined = []
    for key in shared_keys:
        row = {}
        row.update(left[key])
        row.update(right[key])
        combined.append(row)

    write_csv(args.output, combined)
    print("Saved:", args.output)
    print("Combined rows:", len(combined))


if __name__ == "__main__":
    main()
