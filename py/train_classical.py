from __future__ import annotations

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from evaluate import (
    build_results_row,
    evaluate_model,
    evaluate_with_cv,
    print_evaluation,
    save_results_csv,
)
from processing import prepare_dataset


def main() -> None:
    df, split, cv = prepare_dataset()

    X_train, X_test = split.X_train, split.X_test
    y_train, y_test = split.y_train, split.y_test

    models = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000, random_state=42))
        ]),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            n_jobs=-1
        ),
    }

    results = []

    for name, model in models.items():
        model.fit(X_train, y_train)

        test_metrics = evaluate_model(model, X_test, y_test)
        cv_mean, cv_std = evaluate_with_cv(model, X_train, y_train, cv, scoring="f1")

        print_evaluation(name, test_metrics, cv_mean, cv_std)

        results.append(build_results_row(name, test_metrics, cv_mean, cv_std))

    results_df = (
        pd.DataFrame(results)
        .sort_values("CV Mean F1", ascending=False)
        .reset_index(drop=True)
    )

    filepath = save_results_csv(results_df, "classical_models_results.csv")

    print("\nFinal comparison table:")
    print(results_df.round(4))
    print(f"\nResults saved to: {filepath}")


if __name__ == "__main__":
    main()