"""
train_models.py
----------------
Trains 5 classification models (Logistic Regression, Decision Tree, kNN,
Gaussian Naive Bayes, Random Forest) on the Breast Cancer Wisconsin
(Diagnostic) dataset, evaluates each with 6 metrics, and persists:
  - each fitted model (model/*.joblib)
  - the fitted StandardScaler (model/scaler.joblib)
  - the held-out test split, UNSCALED, with the target column (test_data.csv)
  - a metrics comparison table (metrics_comparison.csv / metrics.json)

Run this once from the project-folder directory:
    python model/train_models.py

NOTE: This uses sklearn's bundled Breast Cancer Wisconsin dataset (569
instances, 30 numeric features, binary target) so it needs no internet
access and satisfies the assignment's minimum size requirements
(>=500 instances, >=12 features). Swap load_breast_cancer() below for
pandas.read_csv("your_dataset.csv") if you choose a different Kaggle/UCI
dataset -- the rest of the pipeline is dataset-agnostic as long as the
target column is named "target" and all other columns are numeric features.
"""

import json
import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    classification_report,
    confusion_matrix,
)

RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# 1. Load dataset
# ---------------------------------------------------------------------------
data = load_breast_cancer(as_frame=True)
df = data.frame.copy()
df.rename(columns={"target": "target"}, inplace=True)  # already named 'target'

print(f"Dataset shape: {df.shape[0]} instances x {df.shape[1] - 1} features")
print(f"Class distribution:\n{df['target'].value_counts()}\n")

X = df.drop(columns=["target"])
y = df["target"]

# ---------------------------------------------------------------------------
# 2. Train/test split (80/20, stratified)
# ---------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
)

# ---------------------------------------------------------------------------
# 3. Scale features (fit on train only, to avoid leakage)
# ---------------------------------------------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------------------------------------------------------------
# 4. Save the UNSCALED test split as test_data.csv
#    (this is what gets uploaded to the Streamlit app / submitted to GitHub)
# ---------------------------------------------------------------------------
test_data = X_test.copy()
test_data["target"] = y_test.values
test_data.to_csv("test_data.csv", index=False)
print(f"Saved test_data.csv with {len(test_data)} rows.\n")

# ---------------------------------------------------------------------------
# 5. Define models
# ---------------------------------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=5000, random_state=RANDOM_STATE),
    "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
    "kNN": KNeighborsClassifier(n_neighbors=5),
    "Naive Bayes": GaussianNB(),
    "Random Forest (Ensemble)": RandomForestClassifier(
        n_estimators=200, random_state=RANDOM_STATE
    ),
}

results = {}

# ---------------------------------------------------------------------------
# 6. Train, evaluate, and save each model
# ---------------------------------------------------------------------------
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]

    metrics = {
        "Accuracy": round(accuracy_score(y_test, y_pred), 4),
        "AUC": round(roc_auc_score(y_test, y_proba), 4),
        "Precision": round(precision_score(y_test, y_pred), 4),
        "Recall": round(recall_score(y_test, y_pred), 4),
        "F1": round(f1_score(y_test, y_pred), 4),
        "MCC": round(matthews_corrcoef(y_test, y_pred), 4),
    }
    results[name] = metrics

    print(f"--- {name} ---")
    print(pd.Series(metrics))
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred, target_names=data.target_names))
    print()

    # Save model
    filename = "model/" + name.lower().replace(" ", "_").replace("(", "").replace(")", "") + ".joblib"
    joblib.dump(model, filename)

# Save scaler (needed by the Streamlit app to transform uploaded data)
joblib.dump(scaler, "model/scaler.joblib")
joblib.dump(list(X.columns), "model/feature_names.joblib")
joblib.dump(list(data.target_names), "model/target_names.joblib")

# ---------------------------------------------------------------------------
# 7. Save comparison table
# ---------------------------------------------------------------------------
comparison_df = pd.DataFrame(results).T
comparison_df.index.name = "ML Model Name"
comparison_df.to_csv("metrics_comparison.csv")

with open("model/metrics.json", "w") as f:
    json.dump(results, f, indent=2)

print("=== Final Comparison Table ===")
print(comparison_df.to_string())
print("\nAll models, scaler, and metrics saved under model/. Done.")
