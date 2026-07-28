"""
test_app_logic.py
------------------
Validates the prediction pipeline used by app.py WITHOUT needing Streamlit
running. This is what you should run first on BITS Virtual Lab / locally,
before ever touching `streamlit run`, to catch bugs in the model-loading
and scoring logic early.

Run with:
    python test_app_logic.py
Expected: metrics printed here should match models/metrics.json (produced by
train_models.py) almost exactly (test_data.csv IS the held-out split).
"""

import json
import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef,
)

MODEL_FILES = {
    "Logistic Regression": "models/logistic_regression.joblib",
    "Decision Tree": "models/decision_tree.joblib",
    "kNN": "models/knn.joblib",
    "Naive Bayes": "models/naive_bayes.joblib",
    "Random Forest (Ensemble)": "models/random_forest_ensemble.joblib",
}

print("=== TEST 1: Loading artifacts ===")
scaler = joblib.load("models/scaler.joblib")
feature_names = joblib.load("models/feature_names.joblib")
target_names = joblib.load("models/target_names.joblib")
models = {name: joblib.load(path) for name, path in MODEL_FILES.items()}
with open("models/metrics.json") as f:
    training_metrics = json.load(f)
print(f"Loaded {len(models)} models, scaler, {len(feature_names)} feature names.")
assert len(models) == 5, "Expected exactly 5 saved models"
print("PASS\n")

print("=== TEST 2: Loading test_data.csv ===")
df = pd.read_csv("test_data.csv")
assert "target" in df.columns, "test_data.csv must contain a 'target' column"
missing = [c for c in feature_names if c not in df.columns]
assert not missing, f"Missing feature columns: {missing}"
print(f"test_data.csv shape: {df.shape}")
print("PASS\n")

print("=== TEST 3: Scaling + inference for each model ===")
X = df[feature_names]
y_true = df["target"]
X_scaled = scaler.transform(X)

all_passed = True
for name, model in models.items():
    y_pred = model.predict(X_scaled)
    y_proba = model.predict_proba(X_scaled)[:, 1]

    live_metrics = {
        "Accuracy": round(accuracy_score(y_true, y_pred), 4),
        "AUC": round(roc_auc_score(y_true, y_proba), 4),
        "Precision": round(precision_score(y_true, y_pred), 4),
        "Recall": round(recall_score(y_true, y_pred), 4),
        "F1": round(f1_score(y_true, y_pred), 4),
        "MCC": round(matthews_corrcoef(y_true, y_pred), 4),
    }
    trained = training_metrics[name]

    print(f"--- {name} ---")
    print("Live (this script):     ", live_metrics)
    print("Training-time (json):   ", trained)

    for k in live_metrics:
        if abs(live_metrics[k] - trained[k]) > 1e-4:
            print(f"  MISMATCH on {k}!")
            all_passed = False
    print()

print("=== RESULT ===")
if all_passed:
    print("PASS: All 5 models reproduce training-time metrics on test_data.csv.")
    print("The app.py prediction pipeline logic is verified correct.")
else:
    print("FAIL: Investigate mismatches above before deploying.")
