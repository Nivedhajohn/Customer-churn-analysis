# src/data_preprocessing.py
"""
Preprocessing utilities.
We create a ColumnTransformer (numeric impute+scale, categorical impute+onehot).
This file only *constructs* the preprocessor; it does not persist it.
The final pipeline (preprocessor + model) will be saved by train_model.py.
"""

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer


def create_preprocessor(X: pd.DataFrame):
    """
    Build a ColumnTransformer based on the columns present in X.
    X: DataFrame of features (no target column).
    Returns: an sklearn ColumnTransformer (unfitted).
    """
    # numeric columns: ints + floats
    numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

    # categorical = everything else (object, category, bool, etc.)
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    # numeric transformer: fill missing with median, then scale
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # categorical transformer: fill missing with most frequent, then one-hot (ignore unknowns)
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_cols),
        ('cat', categorical_transformer, categorical_cols)
    ], remainder='drop')  # drop any other columns

    return preprocessor
