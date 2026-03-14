from __future__ import annotations

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV

from evaluate import (
    build_results_row,
    evaluate_model,
    print_evaluation,
    save_model,
    save_results_csv,
)
from processing import prepare_dataset


def main() -> None:
    df, split, cv = prepare_dataset()

    X_train, X_test = split.X_train, split.X_test
    y_train, y_test = split.y_train, split.y_test

    rf_param_grid = {
        "n_estimators": [100, 200, 300, 500],
        "max_depth": [None, 5, 10, 20, 30],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2", None],
    }

    rf_random_search = RandomizedSearchCV(
        estimator=RandomForestClassifier(random_state=42),
        param_distributions=rf_param_grid,
        n_iter=20,
        scoring="f1",
        cv=cv,
        verbose=1,
        n_jobs=-1,
        random_state=42
    )

    rf_random_search.fit(X_train, y_train)

    best_rf = rf_random_search.best_estimator_

    print("\nBest parameters found:")
    print(rf_random_search.best_params_)
    print(f"Best CV F1: {rf_random_search.best_score_:.4f}")

    test_metrics = evaluate_model(best_rf, X_test, y_test)
    print_evaluation("Tuned Random Forest", test_metrics, rf_random_search.best_score_, None)

    results_df = pd.DataFrame([
        build_results_row(
            "Tuned Random Forest",
            test_metrics,
            rf_random_search.best_score_,
            None
        )
    ])

    results_path = save_results_csv(results_df, "tuned_random_forest_results.csv")
    model_path = save_model(best_rf, "tuned_random_forest.joblib")

    print("\nTuned Random Forest results:")
    print(results_df.round(4))
    print(f"\nResults saved to: {results_path}")
    print(f"Model saved to: {model_path}")


if __name__ == "__main__":
    main()