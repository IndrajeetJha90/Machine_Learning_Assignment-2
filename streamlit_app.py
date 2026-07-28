"""
streamlit_app.py
---------------
Interactive web application for comparing breast cancer classification models.
"""

# ========== SUPPRESS SKLEARN VERSION WARNINGS ==========
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
# ======================================================

import json
from pathlib import Path
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
import plotly.express as px
import plotly.graph_objects as go

# ========== PAGE CONFIGURATION ==========
st.set_page_config(
    page_title="🔬 Breast Cancer Classifier Comparison",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========== CUSTOM CSS FOR ENHANCED UI ==========
st.markdown("""
    <style>
    /* Main container styling */
    .main {
        padding: 0rem 1rem;
    }
    
    /* Gradient header */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.3);
        border: 1px solid rgba(255,255,255,0.1);
    }
    
    /* Metric cards with gradient backgrounds */
    .metric-card {
        padding: 1.2rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
        cursor: pointer;
    }
    .metric-card:hover {
        transform: translateY(-8px) scale(1.02);
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.2);
    }
    .metric-card .metric-value {
        font-size: 2.2rem;
        font-weight: bold;
        margin: 0.3rem 0;
        text-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .metric-card .metric-label {
        font-size: 0.85rem;
        opacity: 0.9;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        font-weight: 300;
    }
    
    /* Individual colors for different metrics */
    .metric-accuracy { background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); }
    .metric-auc { background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); }
    .metric-precision { background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%); }
    .metric-recall { background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); }
    .metric-f1 { background: linear-gradient(135deg, #a18cd1 0%, #fbc2eb 100%); }
    .metric-mcc { background: linear-gradient(135deg, #fccb90 0%, #d57eeb 100%); }
    
    /* Data preview box */
    .data-preview-box {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.06);
        margin-bottom: 2rem;
        border: 1px solid #f0f0f0;
    }
    
    /* Success message */
    .custom-success {
        background: linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%);
        padding: 1rem 1.5rem;
        border-radius: 12px;
        color: #1a1a2e;
        font-weight: 500;
        border: none;
        box-shadow: 0 4px 15px rgba(132, 250, 176, 0.3);
    }
    
    /* Custom divider */
    .custom-divider {
        height: 4px;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        margin: 2rem 0;
        border-radius: 4px;
    }
    
    /* Sidebar styling */
    .sidebar-section {
        background: linear-gradient(180deg, #f8f9fa 0%, #ffffff 100%);
        padding: 1rem;
        border-radius: 12px;
        margin-bottom: 1rem;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 1.5rem;
        color: #6c757d;
        font-size: 0.9rem;
        border-top: 2px solid #f0f0f0;
        margin-top: 2rem;
        background: #fafafa;
        border-radius: 12px;
    }
    
    /* Button hover effects */
    .stButton button {
        transition: all 0.3s ease !important;
    }
    .stButton button:hover {
        transform: scale(1.05) !important;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4) !important;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
    }
    </style>
""", unsafe_allow_html=True)

# ========== MODEL REGISTRY ==========
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

MODEL_PATHS = {
    "Logistic Regression": MODEL_DIR / "logistic_regression.joblib",
    "Decision Tree": MODEL_DIR / "decision_tree.joblib",
    "kNN": MODEL_DIR / "knn.joblib",
    "Naive Bayes": MODEL_DIR / "naive_bayes.joblib",
    "Random Forest (Ensemble)": MODEL_DIR / "random_forest_ensemble.joblib",
}

MODEL_EMOJIS = {
    "Logistic Regression": "📊",
    "Decision Tree": "🌳",
    "kNN": "👥",
    "Naive Bayes": "📈",
    "Random Forest (Ensemble)": "🌲"
}

MODEL_COLORS = {
    "Logistic Regression": "#FF6B6B",
    "Decision Tree": "#4ECDC4",
    "kNN": "#45B7D1",
    "Naive Bayes": "#96CEB4",
    "Random Forest (Ensemble)": "#FFEAA7"
}

# ========== LOAD ARTIFACTS ==========
@st.cache_resource
def load_artifacts():
    """Load all pre-trained models and supporting artifacts from disk."""
    try:
        scaler = joblib.load(MODEL_DIR / "scaler.joblib")
        feature_names = joblib.load(MODEL_DIR / "feature_names.joblib")
        target_names = joblib.load(MODEL_DIR / "target_names.joblib")
        
        models = {
            name: joblib.load(path)
            for name, path in MODEL_PATHS.items()
        }
        
        with open(MODEL_DIR / "metrics.json", "r") as f:
            training_metrics = json.load(f)
        
        return scaler, feature_names, target_names, models, training_metrics
    except Exception as e:
        st.error(f"❌ Error loading models: {str(e)}")
        st.stop()

scaler, feature_names, target_names, models, training_metrics = load_artifacts()

# ========== HEADER ==========
st.markdown("""
    <div class="main-header">
        <h1 style="margin: 0; font-size: 2.8rem; font-weight: 700;">
            🔬 Breast Cancer Diagnosis
        </h1>
        <p style="margin: 0.5rem 0 0 0; opacity: 0.9; font-size: 1.2rem; font-weight: 300;">
            Interactive Classifier Comparison & Evaluation Platform
        </p>
        <div style="margin-top: 1rem; display: flex; gap: 2rem; flex-wrap: wrap;">
            <span style="opacity: 0.8;">📚 BITS Pilani WILP M.Tech (AIML/DSE)</span>
            <span style="opacity: 0.8;">📊 Assignment 2 — Machine Learning</span>
            <span style="opacity: 0.8;">📈 Wisconsin Dataset • 569 instances • 30 features</span>
        </div>
    </div>
""", unsafe_allow_html=True)

# ========== SIDEBAR ==========
with st.sidebar:
    st.markdown("""
        <div class="sidebar-section">
            <h3 style="margin-top: 0; color: #667eea;">⚙️ Controls</h3>
        </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "📁 Upload Custom Test Data",
        type=["csv"],
        help="Upload your own CSV file with a 'target' column"
    )
    
    st.markdown("---")
    
    selected_model = st.selectbox(
        "🤖 Select Classification Model",
        list(models.keys()),
        format_func=lambda x: f"{MODEL_EMOJIS.get(x, '')} {x}"
    )
    
    st.markdown("---")
    
    show_comparison = st.toggle(
        "📊 Compare All Models",
        value=False,
        help="Toggle to view side-by-side comparison of all models"
    )
    
    st.markdown("---")
    
    # Show data info in sidebar
    if uploaded_file is not None:
        st.success(f"✅ Loaded: {uploaded_file.name}")
        # Reading the file to show info (cached to avoid re-reading)
        @st.cache_data
        def get_file_info(file):
            df_temp = pd.read_csv(file)
            return df_temp.shape[0]
        try:
            rows = get_file_info(uploaded_file)
            st.info(f"📊 Rows: {rows}")
        except:
            pass
    else:
        st.info("📂 Using default test_data.csv")

# ========== DATA LOADING ==========
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    df = pd.read_csv(BASE_DIR / "test_data.csv")

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

# ========== DATA PREVIEW ==========
st.markdown("""
    <div class="data-preview-box">
        <h3 style="margin-top: 0; color: #667eea;">📋 Test Data Overview</h3>
    </div>
""", unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("📊 Total Samples", df.shape[0])
with col2:
    st.metric("🔢 Features", df.shape[1] - 1)
with col3:
    benign = (df["target"] == 0).sum()
    st.metric("✅ Benign", benign)
with col4:
    malignant = (df["target"] == 1).sum()
    st.metric("⚠️ Malignant", malignant)

st.dataframe(df.head(8), use_container_width=True)

# ========== EVALUATION FUNCTIONS ==========
def evaluate_model(model, X_scaled, y_true):
    """Evaluate a classifier and return predictions plus all metrics."""
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

def create_confusion_matrix_plotly(y_true, y_pred, target_names):
    """Create interactive confusion matrix using Plotly."""
    cm = confusion_matrix(y_true, y_pred)
    
    fig = px.imshow(
        cm,
        text_auto=True,
        x=target_names,
        y=target_names,
        color_continuous_scale="Blues",
        aspect="auto",
        labels=dict(x="Predicted Label", y="True Label", color="Count")
    )
    
    fig.update_layout(
        title=dict(text="🔍 Confusion Matrix", font=dict(size=22, color="#667eea")),
        xaxis=dict(title="Predicted Label", title_font=dict(size=14)),
        yaxis=dict(title="True Label", title_font=dict(size=14)),
        height=450,
        width=550,
        margin=dict(l=50, r=50, t=60, b=50),
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

def create_radar_chart(metrics, model_name):
    """Create radar chart for model performance."""
    categories = list(metrics.keys())
    values = list(metrics.values())
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values + [values[0]],
        theta=categories + [categories[0]],
        fill='toself',
        name=model_name,
        line=dict(color="#667eea", width=3),
        fillcolor='rgba(102, 126, 234, 0.3)',
        marker=dict(size=8, color="#667eea")
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                tickfont=dict(size=11),
                gridcolor='rgba(0,0,0,0.1)'
            ),
            angularaxis=dict(
                tickfont=dict(size=12, color="#333")
            )
        ),
        showlegend=True,
        title=dict(text="📈 Performance Radar", font=dict(size=20, color="#667eea")),
        height=450,
        width=550,
        margin=dict(l=80, r=80, t=60, b=80),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
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

# ========== MAIN VIEW ==========
st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

if not show_comparison:
    # ===== SINGLE MODEL VIEW =====
    model = models[selected_model]
    y_pred, metrics = evaluate_model(model, X_scaled, y_true)
    
    st.markdown(f"""
        <h2 style="text-align: center; color: #667eea; margin-bottom: 1.5rem;">
            {MODEL_EMOJIS.get(selected_model, '')} {selected_model} — Performance Analysis
        </h2>
    """, unsafe_allow_html=True)
    
    # Metric cards with different colors
    cols = st.columns(6)
    metric_classes = ["metric-accuracy", "metric-auc", "metric-precision", 
                     "metric-recall", "metric-f1", "metric-mcc"]
    
    for col, (metric_name, value), css_class in zip(cols, metrics.items(), metric_classes):
        with col:
            st.markdown(f"""
                <div class="metric-card {css_class}">
                    <div class="metric-label">{metric_name}</div>
                    <div class="metric-value">{value:.4f}</div>
                </div>
            """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        fig_cm = create_confusion_matrix_plotly(y_true, y_pred, target_names)
        st.plotly_chart(fig_cm, use_container_width=True)
    
    with col2:
        fig_radar = create_radar_chart(metrics, selected_model)
        st.plotly_chart(fig_radar, use_container_width=True)
    
    # Classification Report
    st.markdown("---")
    st.subheader("📋 Detailed Classification Report")
    report_df = display_classification_report(y_true, y_pred, target_names)
    
    # Styled dataframe with gradient
    st.dataframe(
        report_df.style
        .background_gradient(cmap="Blues", subset=["precision", "recall", "f1-score"])
        .format("{:.3f}", subset=["precision", "recall", "f1-score", "support"])
        .set_properties(**{'font-size': '14px'}),
        use_container_width=True
    )

else:
    # ===== COMPARISON VIEW =====
    st.markdown("""
        <h2 style="text-align: center; color: #667eea; margin-bottom: 1.5rem;">
            📊 Model Comparison Dashboard
        </h2>
    """, unsafe_allow_html=True)
    
    # Compute metrics for all models
    comparison_data = {}
    for name, model in models.items():
        _, metrics = evaluate_model(model, X_scaled, y_true)
        comparison_data[name] = metrics
    
    comparison_df = pd.DataFrame(comparison_data).T.round(4)
    
    # Comparison table
    st.subheader("📊 Performance Comparison Table")
    
    # Color-coded table
    styled_df = comparison_df.style.background_gradient(
        cmap="RdYlGn", 
        subset=list(comparison_df.columns),
        vmin=0.5,
        vmax=1.0
    ).format("{:.4f}")
    
    st.dataframe(styled_df, use_container_width=True)
    
    # Best model highlights
    st.markdown("<br>", unsafe_allow_html=True)
    cols = st.columns(3)
    
    best_accuracy = comparison_df["Accuracy"].idxmax()
    best_f1 = comparison_df["F1 Score"].idxmax()
    best_auc = comparison_df["AUC"].idxmax()
    
    with cols[0]:
        st.markdown(f"""
            <div class="custom-success">
                🏆 <strong>Best Accuracy</strong><br>
                {best_accuracy}<br>
                <span style="font-size: 1.4rem; font-weight: bold;">
                    {comparison_df.loc[best_accuracy, "Accuracy"]:.4f}
                </span>
            </div>
        """, unsafe_allow_html=True)
    
    with cols[1]:
        st.markdown(f"""
            <div class="custom-success">
                🏆 <strong>Best F1 Score</strong><br>
                {best_f1}<br>
                <span style="font-size: 1.4rem; font-weight: bold;">
                    {comparison_df.loc[best_f1, "F1 Score"]:.4f}
                </span>
            </div>
        """, unsafe_allow_html=True)
    
    with cols[2]:
        st.markdown(f"""
            <div class="custom-success">
                🏆 <strong>Best AUC</strong><br>
                {best_auc}<br>
                <span style="font-size: 1.4rem; font-weight: bold;">
                    {comparison_df.loc[best_auc, "AUC"]:.4f}
                </span>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Visual comparisons
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Performance Bar Chart")
        metrics_to_plot = ["Accuracy", "AUC", "F1 Score", "MCC"]
        df_plot = comparison_df[metrics_to_plot].reset_index()
        df_plot = df_plot.melt(id_vars=["index"], var_name="Metric", value_name="Score")
        df_plot.rename(columns={"index": "Model"}, inplace=True)
        
        fig = px.bar(
            df_plot,
            x="Model",
            y="Score",
            color="Metric",
            barmode="group",
            text_auto=".3f",
            color_discrete_sequence=px.colors.qualitative.Set2,
            height=400
        )
        fig.update_layout(
            xaxis_title="Model",
            yaxis_title="Score",
            yaxis_range=[0, 1],
            legend_title="Metric",
            xaxis_tickangle=-30,
            bargap=0.15,
            plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🌡️ Performance Heatmap")
        fig = px.imshow(
            comparison_df,
            text_auto=".3f",
            aspect="auto",
            color_continuous_scale="RdYlGn",
            title="",
            zmin=0.5,
            zmax=1.0
        )
        fig.update_layout(
            xaxis_title="Metrics",
            yaxis_title="Models",
            height=400,
            xaxis=dict(tickangle=0),
            plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig, use_container_width=True)

# ========== FOOTER ==========
st.markdown("""
    <div class="footer">
        💡 <strong>Note:</strong> Metrics are computed live on the current test data. 
        Results may differ from training-time evaluation due to different test splits.
        <br><br>
        <span style="font-size: 0.85rem;">
            Built with ❤️ using Streamlit • Plotly • Scikit-learn
        </span>
        <br>
        <span style="font-size: 0.8rem; opacity: 0.6;">
            BITS Pilani WILP — M.Tech in AI/ML & Data Science Engineering
        </span>
    </div>
""", unsafe_allow_html=True)