import argparse
from pathlib import Path
import zipfile

from common import (
    aggregate_feature_rows,
    choose_evenly_spaced,
    index_raw_zip,
    load_temperature_metadata,
    prefixed_stats,
    read_tiff_array,
    rect_values,
    train_random_forest,
    write_csv,
)


REGIONS = {
    "whole": (0.0, 0.0, 1.0, 1.0),
    "center": (0.2, 0.2, 0.8, 0.8),
    "lower_center": (0.3, 0.55, 0.7, 0.9),
    "nostril_proxy": (0.35, 0.45, 0.65, 0.75),
}


def frame_features(array):
    height, width = array.shape
    features = {}
    for name, (x0, y0, x1, y1) in REGIONS.items():
        rect = (x0 * width, y0 * height, x1 * width, y1 * height)
        features.update(prefixed_stats(name, rect_values(array, rect)))
    return features


def build_records(args):
    metadata_rows = load_temperature_metadata(args.metadata)
    raw_index = index_raw_zip(args.raw_zip)
    records = []
    skipped_unreadable = 0
    with zipfile.ZipFile(args.raw_zip) as zf:
        for row in metadata_rows:
            key = (row["date"], row["sequence_num"])
            raw_frames = raw_index.get(key, [])
            if not raw_frames:
                continue
            sampled = choose_evenly_spaced(raw_frames, args.max_frames)
            per_frame = []
            for _, zip_name in sampled:
                array = read_tiff_array(zf, zip_name)
                if array is None:
                    skipped_unreadable += 1
                    continue
                per_frame.append(frame_features(array))
            if not per_frame:
                continue
            record = {
                "date": row["date"],
                "sequence_num": row["sequence_num"],
                "cow_tag": row["cow_tag"],
                "temperature_f": row["temperature_f"],
                "raw_frame_count": len(raw_frames),
                "sampled_frame_count": len(sampled),
            }
            record.update(aggregate_feature_rows(per_frame))
            records.append(record)
    if skipped_unreadable:
        print("Skipped unreadable TIFF frames:", skipped_unreadable)
    return records


def main():
    parser = argparse.ArgumentParser(description="Train a raw TIFF temperature-statistics baseline.")
    parser.add_argument("--metadata", default="data/annotations/metadata.csv", type=Path)
    parser.add_argument("--raw-zip", default="data/thermal_raw.zip", type=Path)
    parser.add_argument("--output-dir", default="data/temperature_outputs/raw_baseline_v1", type=Path)
    parser.add_argument("--max-frames", default=80, type=int)
    parser.add_argument("--seed", default=1337, type=int)
    parser.add_argument("--test-size", default=0.2, type=float)
    args = parser.parse_args()

    records = build_records(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "features.csv", records)
    metrics = train_random_forest(records, args.output_dir, seed=args.seed, test_size=args.test_size)

    print("Saved:", args.output_dir)
    print("Usable sequences:", metrics["sample_count"])
    print("Feature count:", metrics["feature_count"])
    print("Random Forest test MAE:", metrics["random_forest_test"]["mae"])
    print("Random Forest test RMSE:", metrics["random_forest_test"]["rmse"])
    print("KFold MAE mean:", metrics["kfold_random_forest"]["mae_mean"])
    print("KFold RMSE mean:", metrics["kfold_random_forest"]["rmse_mean"])


if __name__ == "__main__":
    main()
