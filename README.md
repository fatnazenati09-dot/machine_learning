# Red Wine Quality Prediction

## Project overview

This project investigates whether the physicochemical properties of a red wine can be used to predict its quality.

The study compares several supervised learning approaches in order to identify which model provides the best balance between predictive performance and robustness on a tabular dataset of modest size.

The original quality score was transformed into a binary target:

- `1` if `quality >= 6`
- `0` otherwise

This makes it possible to study the problem as a binary classification task: predicting whether a wine is of good quality or not.

## Research question

To what extent can the physicochemical characteristics of a red wine predict its quality, and which type of model offers the best compromise between performance and robustness?

## Dataset

The project uses the **Red Wine Quality** dataset.

Initial characteristics:
- 1,599 observations
- 12 variables
- 11 numerical explanatory variables
- 1 target variable: `quality`

After duplicate removal:
- 1,359 observations

Main explanatory variables:
- fixed acidity
- volatile acidity
- citric acid
- residual sugar
- chlorides
- free sulfur dioxide
- total sulfur dioxide
- density
- pH
- sulphates
- alcohol

## Repository structure

```text
machine_learning/
├── data/
│   └── raw/
│       └── winequality-red.csv
├── notebooks/
│   └── eda_red_wine.ipynb
├── py/
│   ├── processing.py
│   ├── evaluate.py
│   ├── train_classical.py
│   ├── train_deep_learning.py
│   ├── tune_random_forest.py
│   ├── custom_threshold.py
│   └── compare_all_models.py
├── reports/
│   └── projet.pdf
└── README.md
