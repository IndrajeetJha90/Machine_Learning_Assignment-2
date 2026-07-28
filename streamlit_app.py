"""
streamlit_app.py
---------------
Interactive web application for comparing breast cancer classification models.

Features:
  - Upload custom test data (CSV) or fall back to bundled test_data.csv
  - Select from 5 pre-trained classifiers via dropdown
  - Display all 6 evaluation metrics for the chosen model
  - Visualize confusion matrix and full classification report
  - Side-by-side comparison of all models

Run locally:
    streamlit run streamlit_app.py
"""

import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report,
)

# Page configuration
st.set_page_config(
    page_title="Breast Cancer Classifier Comparison",
    page_icon="🔬",
    layout="wide"
)

# Model registry
MODEL_PATHS = {
    "Logistic Regression": "models/logistic_regression.joblib",
    "Decision Tree": "models/decision_tree.joblib",
    "kNN": "models/knn.joblib",
    "Naive Bayes": "models/naive_bayes.joblib",
    "Random Forest (Ensemble)": "models/random_forest_ensemble.joblib",
}

# Constants
RANDOM_STATE = 42
TEST_SIZE = 0.2

"""
    Load all pre-trained models and supporting artifacts from disk.
    Cached to avoid reloading on every interaction.
"""

@st.cache_resource
def load_artifacts():

    scaler = joblib.load("models/scaler.joblib")
    feature_names = joblib.load("models/feature_names.joblib")
    target_names = joblib.load("models/target_names.joblib")
    
    models = {
        name: joblib.load(path) 
        for name, path in MODEL_PATHS.items()
    }
    
    with open("models/metrics.json", "r") as f:
        training_metrics = json.load(f)
    
    return scaler, feature_names, target_names, models, training_metrics


# Load all artifacts
scaler, feature_names, target_names, models, training_metrics = load_artifacts()

# App header
st.title("🔬 Breast Cancer Diagnosis — Classifier Comparison")
st.caption(
    "Assignment 2 — Machine Learning | BITS Pilani WILP M.Tech (AIML/DSE) | "
    "Dataset: Breast Cancer Wisconsin (Diagnostic), 569 instances, 30 features"
)

# ---------------------------------------------------------------------------
# Sidebar Configuration
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Controls")
    
    uploaded_file = st.file_uploader(
        "Upload test data (CSV format with 'target' column)",
        type=["csv"]
    )
    
    selected_model = st.selectbox(
        "Select a classification model",
        list(models.keys())
    )
    
    show_comparison = st.checkbox(
        "Compare all models side-by-side",
        value=False
    )
    
    st.divider()
    st.caption("Built with ❤️ using Streamlit")

# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.sidebar.success(f"✅ Loaded {df.shape[0]} rows from uploaded file")
else:
    df = pd.read_csv("test_data.csv")
    st.sidebar.info("📂 Using default test_data.csv (no file uploaded)")

# Validate data
if "target" not in df.columns:
    st.error("❌ CSV must contain a 'target' column with true labels")
    st.stop()

missing_features = [col for col in feature_names if col not in df.columns]
if missing_features:
    st.error(f"❌ Missing required feature columns: {missing_features[:5]}...")
    st.stop()

# Prepare test data
X = df[feature_names]
y_true = df["target"]
X_scaled = scaler.transform(X)

# Display data preview
st.subheader("📋 Test Data Preview")
st.dataframe(df.head(10), use_container_width=True)

# ---------------------------------------------------------------------------
# Evaluation Function
# ---------------------------------------------------------------------------
def evaluate_model(model, X_scaled, y_true):
    """
    Evaluate a classifier and return predictions plus all metrics.
    """
    y_pred = model.predict(X_scaled)
    y_proba = model.predict_proba(X_scaled)[:, 1]
    
    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_proba),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1 Score": f1_score(y_true, y_pred),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }
    
    return y_pred, metrics


def display_confusion_matrix(y_true, y_pred, target_names):
    """Generate and display confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=target_names,
        yticklabels=target_names,
        ax=ax,
        cbar_kws={"label": "Count"}
    )
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    return fig


def display_classification_report(y_true, y_pred, target_names):
    """Generate and display classification report as DataFrame."""
    report = classification_report(
        y_true, 
        y_pred, 
        target_names=target_names, 
        output_dict=True
    )
    return pd.DataFrame(report).T.round(3)


# ---------------------------------------------------------------------------
# Single Model View
# ---------------------------------------------------------------------------
if not show_comparison:
    model = models[selected_model]
    y_pred, metrics = evaluate_model(model, X_scaled, y_true)
    
    # Metrics display
    st.subheader(f"📊 Performance Metrics — {selected_model}")
    cols = st.columns(6)
    for col, (metric_name, value) in zip(cols, metrics.items()):
        col.metric(metric_name, f"{value:.4f}")
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Confusion Matrix")
        fig = display_confusion_matrix(y_true, y_pred, target_names)
        st.pyplot(fig)
        plt.close(fig)
    
    with col2:
        st.subheader("Classification Report")
        report_df = display_classification_report(y_true, y_pred, target_names)
        st.dataframe(report_df, use_container_width=True)

# ---------------------------------------------------------------------------
# Model Comparison View
# ---------------------------------------------------------------------------
else:
    st.subheader("📊 Model Comparison on Test Data")
    
    # Compute metrics for all models
    comparison_data = {}
    for name, model in models.items():
        _, metrics = evaluate_model(model, X_scaled, y_true)
        comparison_data[name] = metrics
    
    comparison_df = pd.DataFrame(comparison_data).T.round(4)
    
    # Display comparison table
    st.dataframe(comparison_df, use_container_width=True)
    
    # Highlight best model
    best_model = comparison_df["F1 Score"].idxmax()
    st.success(f"🏆 Best performing model (F1 Score): **{best_model}**")
    
    # Visual comparison
    st.subheader("Performance Visualization")
    metrics_to_plot = ["Accuracy", "AUC", "F1 Score", "MCC"]
    st.bar_chart(comparison_df[metrics_to_plot])

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.divider()
st.caption(
    "💡 **Note**: Metrics are computed live on the current test data. "
    "Results may differ from the training-time comparison due to different "
    "test splits or uploaded custom data."
)