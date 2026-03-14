from __future__ import annotations

import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from evaluate import (
    build_results_row,
    evaluate_model,
    evaluate_with_cv,
    print_evaluation,
    save_model,
    save_results_csv,
)
from processing import prepare_dataset


def main() -> None:
    df, split, cv = prepare_dataset()

    X_train, X_test = split.X_train, split.X_test
    y_train, y_test = split.y_train, split.y_test

    mlp_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            solver="adam",
            alpha=0.0001,
            batch_size=32,
            learning_rate_init=0.001,
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=42
        ))
    ])

    mlp_pipeline.fit(X_train, y_train)

    test_metrics = evaluate_model(mlp_pipeline, X_test, y_test)
    cv_mean, cv_std = evaluate_with_cv(mlp_pipeline, X_train, y_train, cv, scoring="f1")

    print_evaluation("MLP", test_metrics, cv_mean, cv_std)

    results_df = pd.DataFrame([
        build_results_row("MLP", test_metrics, cv_mean, cv_std)
    ])

    results_path = save_results_csv(results_df, "mlp_results.csv")
    model_path = save_model(mlp_pipeline, "mlp_pipeline.joblib")

    print("\nMLP results:")
    print(results_df.round(4))
    print(f"\nResults saved to: {results_path}")
    print(f"Model saved to: {model_path}")


if __name__ == "__main__":
    main()