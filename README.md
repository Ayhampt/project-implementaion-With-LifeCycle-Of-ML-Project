# ML Project: Life Cycle Implementation

![GitHub last commit](https://img.shields.io/github/last-commit/Ayhampt/project-implementaion-With-LifeCycle-Of-ML-Project?style=for-the-badge)

## Project Overview

This repository contains a machine learning project demonstrating the end-to-end lifecycle of building, evaluating, and documenting models. The work includes data cleaning, exploratory data analysis, feature preparation, model training, evaluation, and serialization.

## Key Components

- `Algerian_forest_fires_dataset_UPDATE.csv`: Raw forest fire dataset used for cleaning, preprocessing, and model development.
- `Algerian_forest_fires_dataset_cleaned.csv`: Cleaned version of the forest fire dataset produced by the preprocessing workflow.
- `height-weight.csv`: Dataset used for a simple linear regression example.
- `scalar.pkl`: Serialized scaler object for feature scaling.
- `lasso_cv.pkl`: Serialized Lasso model or cross-validation model artifact.
- `requirements.txt`: Python package dependencies used by the notebooks.

## Notebooks

The project is organized into several Jupyter notebooks, each focusing on a different part of the machine learning workflow.

### `model training.ipynb`

- Loads the cleaned Algerian forest fires dataset.
- Prepares features and target variables.
- Removes redundant correlated features.
- Performs feature scaling with `StandardScaler`.
- Trains a linear regression model for forecasting Fire Weather Index (FWI).
- Includes visualization of correlation and feature distributions.

### `simple_linear_regression_model_training.ipynb`

- Loads `height-weight.csv`.
- Visualizes the relationship between weight and height.
- Performs train-test splitting.
- Applies standard scaling.
- Trains a simple linear regression model.
- Evaluates the model using MAE, MSE, RMSE, and R² score.

### `multiple_linear_regression_model_traning.ipynb`

- Loads the California housing dataset from `sklearn.datasets`.
- Converts the dataset into a pandas DataFrame.
- Explores data quality, correlations, and descriptive statistics.
- Splits data into training and test sets.
- Applies standard scaling.
- Trains a multiple linear regression model to predict housing prices.

### `Ridge,Lasso Regression.ipynb`

- Loads the raw Algerian forest fire dataset with header correction.
- Performs data cleaning and null-value handling.
- Converts categorical values into numeric representations.
- Conducts exploratory data analysis with histograms, correlation matrices, and class distribution visualization.
- Prepares data for regularized regression modeling.

## Installation

1. Create a Python virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

> Note: `requirements.txt` contains spelling errors in package names. Use the corrected package list below if installation fails.

Correct dependencies:

- `numpy`
- `pandas`
- `scikit-learn`
- `matplotlib`
- `seaborn`

## Usage

1. Open the project folder in VS Code or JupyterLab.
2. Open a notebook file, for example `model training.ipynb`.
3. Run the notebook cells sequentially to reproduce the data workflow and model training steps.

## Notes

- The forest fire dataset is used for both classification-style preprocessing and regression modeling.
- The notebooks demonstrate common ML lifecycle stages: data ingestion, cleaning, visualization, feature scaling, training, and evaluation.
- Model artifacts like `scalar.pkl` and `lasso_cv.pkl` can be used to load trained components in a loader script or production workflow.
