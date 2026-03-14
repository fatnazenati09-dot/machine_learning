from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from evaluate import load_model, save_results_csv
from processing import get_project_dir, prepare_dataset


def main() -> None:
    df, split, cv = prepare_dataset()

    X_test = split.X_test
    y_test = split.y_test

    best_rf = load_model("tuned_random_forest.joblib")
    y_proba = best_rf.predict_proba(X_test)[:, 1]

    thresholds = [round(x, 2) for x in [
        0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45,
        0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90
    ]]

    results = []

    for threshold in thresholds:
        y_pred_thresh = (y_proba >= threshold).astype(int)

        results.append({
            "threshold": threshold,
            "accuracy": accuracy_score(y_test, y_pred_thresh),
            "precision": precision_score(y_test, y_pred_thresh),
            "recall": recall_score(y_test, y_pred_thresh),
            "f1": f1_score(y_test, y_pred_thresh),
        })

    threshold_df = pd.DataFrame(results)
    best_row = threshold_df.loc[threshold_df["f1"].idxmax()]

    best_threshold = best_row["threshold"]
    y_pred_optimal = (y_proba >= best_threshold).astype(int)

    print("Threshold comparison:")
    print(threshold_df.round(4))

    print("\nBest threshold row:")
    print(best_row.round(4))

    print(f"\nBest threshold: {best_threshold:.2f}")
    print(f"Accuracy: {accuracy_score(y_test, y_pred_optimal):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred_optimal):.4f}")
    print(f"Recall: {recall_score(y_test, y_pred_optimal):.4f}")
    print(f"F1-score: {f1_score(y_test, y_pred_optimal):.4f}")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred_optimal))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred_optimal))

    save_results_csv(threshold_df.round(4), "threshold_optimization_results.csv")

    figures_dir = get_project_dir() / "outputs" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.plot(threshold_df["threshold"], threshold_df["precision"], label="Precision")
    plt.plot(threshold_df["threshold"], threshold_df["recall"], label="Recall")
    plt.plot(threshold_df["threshold"], threshold_df["f1"], label="F1-score")
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.title("Effect of Classification Threshold on Performance")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    figure_path = figures_dir / "threshold_optimization.png"
    plt.savefig(figure_path, dpi=300)
    plt.show()

    print(f"\nThreshold plot saved to: {figure_path}")


if __name__ == "__main__":
    main()