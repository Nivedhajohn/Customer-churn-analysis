# src/train_model.py
"""
Train script:
- loads data/data_churn.csv
- builds preprocessing pipeline
- trains a few models (LogisticRegression, RandomForest, XGBoost if available)
- evaluates and picks the best by ROC-AUC
- saves the full pipeline (preprocessor + classifier) to models/churn_model.pkl
"""

import os
import sys
import joblib
import pandas as pd
import numpy as np

# ensure src dir is on path so local imports work when running `python src/train_model.py` from project root
sys.path.append(os.path.dirname(__file__))

from data_preprocessing import create_preprocessor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, classification_report)
from sklearn.pipeline import Pipeline

# optional xgboost
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False


def prepare_target(y: pd.Series) -> pd.Series:
    """Try to convert common target representations to 0/1 ints."""
    if y.dtype.kind in 'biu':  # already numeric
        return y.astype(int)
    lowered = y.astype(str).str.strip().str.lower()
    if set(lowered.unique()) <= {'yes', 'no'}:
        return lowered.map({'yes': 1, 'no': 0}).astype(int)
    if set(lowered.unique()) <= {'y', 'n'}:
        return lowered.map({'y': 1, 'n': 0}).astype(int)
    if set(lowered.unique()) <= {'true', 'false'}:
        return lowered.map({'true': 1, 'false': 0}).astype(int)
    # try numeric conversion (strings like '1'/'0')
    try:
        arr = lowered.astype(float)
        uniq = set(arr.unique())
        if uniq <= {0.0, 1.0}:
            return arr.astype(int)
    except Exception:
        pass
    raise ValueError("Cannot interpret target column as binary (0/1 or Yes/No). "
                     "Please ensure the target uses 'Yes'/'No' or 1/0.")


def main():
    # paths
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    data_path = os.path.join(project_root, 'data', 'customer churn data.csv')
    model_out = os.path.join(project_root, 'models', 'churn_model.pkl')

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}. Put your dataset there.")

    # load
    df = pd.read_csv(data_path)
    print("Loaded dataset:", data_path)
    print("Shape:", df.shape)
    print(df.head(3))

    # Expect a target column named 'Churn' (case-sensitive). If your dataset uses another name, change this.
    target_col = 'Churn'
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found. Rename your churn column to 'Churn' or change target_col.")

    # prepare target y and features X
    y = prepare_target(df[target_col])
    X = df.drop(columns=[target_col])

    print("Target distribution (after mapping):")
    print(y.value_counts())

    # quick missing values check
    print("Missing values per column (top 10):")
    print(df.isna().sum().sort_values(ascending=False).head(10))

    # split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # build preprocessor using training features
    preprocessor = create_preprocessor(X_train)

    # prepare candidate models
    models = {}
    models['logreg'] = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
    models['rf'] = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced')

    if XGBOOST_AVAILABLE:
        # compute scale_pos_weight for xgboost to handle imbalance
        pos = y_train.sum()
        neg = len(y_train) - pos
        scale_pos_weight = float(neg) / float(pos) if pos > 0 else 1.0
        models['xgb'] = XGBClassifier(use_label_encoder=False, eval_metric='logloss',
                                      random_state=42, scale_pos_weight=scale_pos_weight)

    results = {}

    for name, clf in models.items():
        print(f"\nTraining {name} ...")
        pipe = Pipeline(steps=[('preprocessor', preprocessor), ('clf', clf)])
        pipe.fit(X_train, y_train)

        # predictions
        y_pred = pipe.predict(X_test)
        try:
            y_proba = pipe.predict_proba(X_test)[:, 1]
            roc = roc_auc_score(y_test, y_proba)
        except Exception:
            y_proba = None
            roc = None

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        print(f"{name} metrics: accuracy={acc:.4f}, precision={prec:.4f}, recall={rec:.4f}, f1={f1:.4f}, roc_auc={roc}")
        print("Classification report:")
        print(classification_report(y_test, y_pred, zero_division=0))

        # store
        results[name] = {
            'pipeline': pipe,
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1': f1,
            'roc_auc': roc if roc is not None else -1.0
        }

    # choose best by roc_auc (fallback to f1)
    best_name = max(results.items(), key=lambda kv: (kv[1].get('roc_auc', -1), kv[1].get('f1', 0)))[0]
    best_pipeline = results[best_name]['pipeline']
    print(f"\nBest model: {best_name}")

    # persist best pipeline
    os.makedirs(os.path.join(project_root, 'models'), exist_ok=True)
    joblib.dump(best_pipeline, model_out)
    print(f"Saved best pipeline ({best_name}) to: {model_out}")
    print("Done.")


if __name__ == '__main__':
    main()
