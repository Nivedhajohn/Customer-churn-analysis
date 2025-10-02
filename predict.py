# src/predict.py
"""
Predict script:
- If you pass --file new_customers.csv, it predicts churn for each row and writes new_customers_predictions.csv.
- If you run without args, it shows an example of how to pass a single sample dictionary.
Usage:
    python src/predict.py --file path/to/new_customers.csv
"""

import os
import sys
import argparse
import joblib
import pandas as pd

# ensure src dir on path
sys.path.append(os.path.dirname(__file__))

def load_pipeline():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    model_path = os.path.join(project_root, 'models', 'churn_model.pkl')
    if not os.path.exists(model_path):
        raise FileNotFoundError("Model not found. Train first by running: python src/train_model.py")
    return joblib.load(model_path)

def predict_file(pipeline, file_path):
    df = pd.read_csv(file_path)
    # predictions
    try:
        proba = pipeline.predict_proba(df)[:, 1]
        preds = pipeline.predict(df)
    except Exception as e:
        raise RuntimeError(f"Prediction failed. Make sure uploaded file has same feature columns as training set. Error: {e}")

    out = df.copy()
    out['churn_probability'] = proba
    out['churn_pred'] = preds
    out['churn_label'] = out['churn_pred'].map({1: 'Churn', 0: 'Not Churn'})

    out_path = file_path.replace('.csv', '_predictions.csv')
    out.to_csv(out_path, index=False)
    print(f"Saved predictions to {out_path}")
    print(out[['churn_probability', 'churn_label']].head())

def predict_single(pipeline, sample_dict):
    df = pd.DataFrame([sample_dict])
    proba = pipeline.predict_proba(df)[:, 1]
    pred = pipeline.predict(df)[0]
    label = 'Churn' if pred == 1 else 'Not Churn'
    print("Prediction:")
    print("Probability of churn:", float(proba[0]))
    print("Label:", label)
    return float(proba[0]), label

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', help='CSV file with rows to predict (must contain same features as training X).')
    args = parser.parse_args()

    pipeline = load_pipeline()

    if args.file:
        predict_file(pipeline, args.file)
    else:
        # Example dictionary - you must replace keys with your real feature names and values
        print("No file provided. Showing example of single-sample prediction.")
        example = {}
        # Build a minimal example: get expected feature names from preprocessor if possible
        try:
            preprocessor = pipeline.named_steps['preprocessor']
            # attempt to obtain required columns
            cols = []
            for trans in preprocessor.transformers_:
                # trans is (name, transformer, columns)
                if len(trans) >= 3:
                    cols.extend(list(trans[2]))
            if cols:
                print("Detected feature columns:", cols)
                for c in cols:
                    example[c] = 0  # naive defaults; replace with realistic values
            else:
                print("Could not auto-detect feature names. Fill example manually.")
                example = {'feature1': 0}
        except Exception:
            example = {'feature1': 0}

        print("Example input (please modify keys/values to match your dataset):")
        print(example)
        predict_single(pipeline, example)

if __name__ == '__main__':
    main()
