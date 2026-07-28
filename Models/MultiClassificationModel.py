"""
This script trains five classification models—Logistic Regression, Decision Tree, k-Nearest Neighbors, Gaussian Naive Bayes, and Random Forest—using the Breast Cancer Wisconsin (Diagnostic) dataset. 

It evaluates each model with six performance metrics and saves the following outputs:

Each trained model as a .joblib file in the models/ folder

The fitted StandardScaler as scaler.joblib

The unscaled test set (including the target column) as test_data.csv

A comparison table of metrics as both metrics_comparison.csv and metrics.json
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
"""
Brief Description of Models:
1. Logistic Regression: Predicts the probability that an instance belongs to a certain class using a logistic (sigmoid) function.
Key terms:
Sigmoid function – maps any real-valued number to a probability between 0 and 1.
Decision boundary – a linear threshold that separates classes.
Log-odds – the log of the odds ratio; the model learns linear relationships between features and the log-odds of the target.
Best for: Linearly separable data, interpretability, and when features are roughly independent.

2. Decision Tree: Splits the data recursively based on feature values, creating a tree-like structure where each leaf node represents a class decision.
Key terms:
Splitting criterion – uses metrics like Gini impurity or entropy to choose the best feature to split on.
Root/Internal/Leaf nodes – root is the first split; internal nodes are decision points; leaves are final predictions.
Overfitting – tends to memorize training data if the tree grows too deep without pruning.
Best for: Easy visualization, handling non-linear relationships, and when interpretability matters.

3. k-Nearest Neighbors (kNN): Classifies a new instance by looking at the k closest training points (neighbors) and taking a majority vote.
Key terms:
Distance metric – typically Euclidean distance; defines "closeness" between points.
k (number of neighbors) – a hyperparameter; small k can be noisy, large k smooths decisions.
Lazy learner – doesn't build an explicit model; just stores all training data and makes predictions on the fly.
Best for: Low-dimensional data, small datasets, and when decision boundaries are irregular.

4. Gaussian Naive Bayes: Applies Bayes' theorem with the "naive" assumption that features are conditionally independent given the class, and uses Gaussian distributions to model each feature's likelihood.
Key terms:
Bayes' theorem – computes posterior probability using prior probability and likelihood.
Conditional independence – assumes features don't influence each other (which is rarely true but works well in practice).
Gaussian distribution – models each feature per class as a normal (bell-curve) distribution with its own mean and variance.
Best for: Text classification, high-dimensional data, and when training speed is critical.

5. Random Forest (Ensemble): Builds many decision trees (typically hundreds) on random subsets of data and features, then averages their predictions for a final decision.
Key terms:
Ensemble learning – combines multiple weak learners to create a strong, robust model.
Bagging (Bootstrap Aggregating) – trains each tree on a random sample (with replacement) of the training data.
Feature randomness – each tree only considers a random subset of features at each split, reducing correlation between trees.
n_estimators – the number of trees (here, 200); more trees generally improve stability but increase computation.
Best for: High-dimensional data, handling non-linear relationships, reducing overfitting (compared to single trees), and achieving high accuracy with minimal tuning.
"""
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

"""
Brief Classification:
1. Accuracy: The overall correctness of the model — the proportion of all predictions that are correct.
Formula: (TP + TN) / (TP + TN + FP + FN)
Interpretation:
--Simple and intuitive: "How often did the model get it right?"
--Limitation: Can be misleading for imbalanced datasets. If 95% of cases are benign, a model that always predicts "benign" gets 95% accuracy but is useless.

2. AUC (Area Under the ROC Curve): The model's ability to distinguish between positive and negative classes across all possible classification thresholds.
Interpretation: 
--Ranges from 0 to 1:
0.5 = Random guessing (no discrimination)
0.7-0.8 = Acceptable
0.8-0.9 = Excellent
>0.9 = Outstanding
--Key insight: Unlike accuracy, AUC is threshold-independent — it evaluates the model's ranking ability: "How well does the model rank malignant cases higher than benign ones?"
--Limitation: Doesn't tell you which threshold to use in practice.

3. Precision: Of all cases predicted as positive, how many were actually positive? (How precise/trustworthy are your positive predictions?)
Formula: TP / (TP + FP)
Interpretation:
--High precision = Few false positives (fewer "false alarms")
--Important when false positives are costly — e.g., telling a healthy patient they have cancer (causing unnecessary stress and procedures)

4. Recall (Sensitivity / True Positive Rate): What it measures: Of all actual positive cases, how many did the model correctly identify?
Formula: TP / (TP + FN)
Interpretation:
--High recall = Few false negatives (fewer missed cases)
--Important when false negatives are costly — e.g., missing a cancer diagnosis (could be life-threatening)

5. F1 Score: What it measures: The harmonic mean of precision and recall — a single metric that balances both.
Formula: 2 × (Precision × Recall) / (Precision + Recall)
Interpretation:
--Ranges from 0 to 1
--Useful when: You need a single number that balances precision and recall, especially when there's a trade-off between them.
--Why harmonic mean? It penalizes extreme values more heavily than arithmetic mean — both precision and recall need to be decent for a good F1.

6. MCC (Matthews Correlation Coefficient):What it measures: A balanced correlation coefficient that takes all four confusion matrix entries into account.
Formula:(TP × TN - FP × FN) / √[(TP+FP)(TP+FN)(TN+FP)(TN+FN)]
Interpretation:
--Ranges from -1 to +1:
+1 = Perfect prediction
0 = Random guessing
-1 = Complete disagreement (worse than random)
--Key advantage: Works well even with imbalanced datasets (unlike accuracy)
--Considered a "gold standard" metric for binary classification because it gives a balanced view of all four quadrants of the confusion matrix.
"""

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
    filename = "models/" + name.lower().replace(" ", "_").replace("(", "").replace(")", "") + ".joblib"
    joblib.dump(model, filename)

# Save scaler (needed by the Streamlit app to transform uploaded data)
joblib.dump(scaler, "models/scaler.joblib")
joblib.dump(list(X.columns), "models/feature_names.joblib")
joblib.dump(list(data.target_names), "models/target_names.joblib")

# ---------------------------------------------------------------------------
# 7. Save comparison table
# ---------------------------------------------------------------------------
comparison_df = pd.DataFrame(results).T
comparison_df.index.name = "ML Model Name"
comparison_df.to_csv("metrics_comparison.csv")

with open("models/metrics.json", "w") as f:
    json.dump(results, f, indent=2)

print("=== Final Comparison Table ===")
print(comparison_df.to_string())
print("\nAll models, scaler, and metrics saved under models/. Done.")
