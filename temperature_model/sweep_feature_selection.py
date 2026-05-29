from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import read_feature_csv, train_random_forest, write_csv


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train Random Forest temperature models with several feature-selection sizes."
    )
    parser.add_argument("--features", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--ks", nargs="+", default=["10", "20", "40", "80", "160"], help="Feature counts to test.")
    parser.add_argument("--seed", default=1337, type=int)
    parser.add_argument("--test-size", default=0.2, type=float)
    args = parser.parse_args()

    records = read_feature_csv(args.features)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for k_text in args.ks:
        select_k = None if k_text.lower() in {"all", "none", "0"} else int(k_text)
        run_name = "all_features" if select_k is None else f"top_{select_k}"
        run_dir = args.output_dir / run_name
        metrics = train_random_forest(
            records,
            run_dir,
            seed=args.seed,
            test_size=args.test_size,
            select_k=select_k,
        )
        row = {
            "run": run_name,
            "sample_count": metrics["sample_count"],
            "feature_count": metrics["feature_count"],
            "selected_feature_count": metrics["selected_feature_count"],
            "test_mae": metrics["random_forest_test"]["mae"],
            "test_rmse": metrics["random_forest_test"]["rmse"],
            "kfold_mae_mean": metrics["kfold_random_forest"]["mae_mean"],
            "kfold_rmse_mean": metrics["kfold_random_forest"]["rmse_mean"],
        }
        rows.append(row)
        print(
            f"{run_name}: test MAE={row['test_mae']:.3f}, "
            f"kfold MAE={row['kfold_mae_mean']:.3f}"
        )

    rows.sort(key=lambda row: row["test_mae"])
    write_csv(args.output_dir / "sweep_summary.csv", rows)
    with (args.output_dir / "best_run.json").open("w", encoding="utf-8") as f:
        json.dump(rows[0], f, indent=2)
    print("Best by holdout MAE:", rows[0]["run"])
    print("Saved:", args.output_dir)


if __name__ == "__main__":
    main()
