import argparse
from pathlib import Path

from common import read_feature_csv, train_random_forest


def main():
    parser = argparse.ArgumentParser(description="Train a Random Forest rectal-temperature regressor from features.csv.")
    parser.add_argument("--features", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", default=1337, type=int)
    parser.add_argument("--test-size", default=0.2, type=float)
    parser.add_argument(
        "--select-k",
        default=None,
        type=int,
        help="Keep only the top K univariate features before fitting Random Forest.",
    )
    args = parser.parse_args()

    records = read_feature_csv(args.features)
    metrics = train_random_forest(
        records,
        args.output_dir,
        seed=args.seed,
        test_size=args.test_size,
        select_k=args.select_k,
    )

    print("Saved:", args.output_dir)
    print("Samples:", metrics["sample_count"])
    print("Feature count:", metrics["feature_count"])
    print("Selected feature count:", metrics["selected_feature_count"])
    print("Random Forest test MAE:", metrics["random_forest_test"]["mae"])
    print("Random Forest test RMSE:", metrics["random_forest_test"]["rmse"])
    print("KFold MAE mean:", metrics["kfold_random_forest"]["mae_mean"])
    print("KFold RMSE mean:", metrics["kfold_random_forest"]["rmse_mean"])


if __name__ == "__main__":
    main()
