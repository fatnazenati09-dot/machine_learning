from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split


@dataclass
class DataSplit:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series


def get_project_dir() -> Path:
    """
    Return the root project directory.
    """
    return Path(__file__).resolve().parent.parent


def get_data_path(filename: str = "winequality-red.csv") -> Path:
    """
    Return the full path to the raw dataset.
    """
    return get_project_dir() / "data" / "raw" / filename


def load_data(filepath: str | Path | None = None) -> pd.DataFrame:
    """
    Load the red wine dataset from a CSV file.
    """
    if filepath is None:
        filepath = get_data_path()

    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"Dataset not found at: {filepath}")

    df = pd.read_csv(filepath, sep=",")
    df.columns = df.columns.str.strip().str.lower()

    return df


def create_binary_target(
    df: pd.DataFrame,
    quality_col: str = "quality",
    threshold: int = 6
) -> pd.DataFrame:
    """
    Create binary target:
    1 if quality >= threshold, 0 otherwise.
    """
    df = df.copy()
    df["good_quality"] = (df[quality_col] >= threshold).astype(int)
    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicated rows.
    """
    return df.drop_duplicates().copy()


def prepare_features_target(
    df: pd.DataFrame,
    target_col: str = "good_quality",
    quality_col: str = "quality"
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Split dataset into features X and target y.
    """
    X = df.drop(columns=[quality_col, target_col])
    y = df[target_col]
    return X, y


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42
) -> DataSplit:
    """
    Perform stratified train-test split.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    return DataSplit(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test
    )


def get_cv(random_state: int = 42, n_splits: int = 5) -> StratifiedKFold:
    """
    Return StratifiedKFold object.
    """
    return StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state
    )


def prepare_dataset(
    filepath: str | Path | None = None,
    threshold: int = 6,
    test_size: float = 0.2,
    random_state: int = 42
) -> tuple[pd.DataFrame, DataSplit, StratifiedKFold]:
    """
    Full preprocessing pipeline:
    - load data
    - create binary target
    - remove duplicates
    - split into X/y
    - train/test split
    - create CV object
    """
    df = load_data(filepath)
    df = create_binary_target(df, threshold=threshold)
    df = remove_duplicates(df)

    X, y = prepare_features_target(df)
    split = split_data(X, y, test_size=test_size, random_state=random_state)
    cv = get_cv(random_state=random_state)

    return df, split, cv


if __name__ == "__main__":
    data_path = get_data_path()

    print("SCRIPT EXECUTED:", __file__)
    print("CURRENT WORKING DIR:", Path.cwd())
    print("PROJECT DIR:", get_project_dir())
    print("DATA PATH:", data_path)
    print("EXISTS:", data_path.exists())

    df, split, cv = prepare_dataset(data_path)

    print("\nDataset shape after preprocessing:", df.shape)
    print("X_train shape:", split.X_train.shape)
    print("X_test shape:", split.X_test.shape)
    print("\nTrain target distribution:")
    print(split.y_train.value_counts(normalize=True).round(3))
    print("\nTest target distribution:")
    print(split.y_test.value_counts(normalize=True).round(3))