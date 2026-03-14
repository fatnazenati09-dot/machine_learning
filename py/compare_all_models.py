from __future__ import annotations

import pandas as pd

from evaluate import save_results_csv
from processing import get_project_dir


def main() -> None:
    metrics_dir = get_project_dir() / "outputs" / "metrics"

    files = [
        metrics_dir / "classical_models_results.csv",
        metrics_dir / "mlp_results.csv",
        metrics_dir / "tuned_random_forest_results.csv",
    ]

    dfs = []

    for path in files:
        if path.exists():
            dfs.append(pd.read_csv(path))
        else:
            print(f"Missing file: {path}")

    if not dfs:
        raise FileNotFoundError("No metrics files found. Run the previous scripts first.")

    comparison_df = pd.concat(dfs, ignore_index=True)

    if "CV Mean F1" in comparison_df.columns:
        comparison_df["Rank by CV Mean F1"] = comparison_df["CV Mean F1"].rank(
            ascending=False,
            method="dense"
        )
        comparison_df = comparison_df.sort_values("Rank by CV Mean F1")

    comparison_df = comparison_df.reset_index(drop=True)

    filepath = save_results_csv(comparison_df.round(4), "final_model_comparison.csv")

    print("Final comparison:")
    print(comparison_df.round(4))
    print(f"\nSaved to: {filepath}")


if __name__ == "__main__":
    main()