# ML Assignment 2 — Classification Model Comparison & Streamlit Deployment

## a. Problem Statement

Breast cancer diagnosis requires distinguishing malignant tumors from benign
ones based on measurements taken from digitized images of a fine needle
aspirate (FNA) of a breast mass. This project frames diagnosis as a **binary
classification problem**: given 30 numeric features describing cell nuclei
characteristics, predict whether a tumor is **malignant** or **benign**. Six
classification models are trained, evaluated, and compared, and the best
model is served through an interactive Streamlit web app.

## b. Dataset Description

- **Name:** Breast Cancer Wisconsin (Diagnostic) Data Set
- **Source:** UCI Machine Learning Repository (bundled in
  `sklearn.datasets.load_breast_cancer`)
- **Instances:** 569 (212 malignant, 357 benign)
- **Features:** 30 numeric features (mean, standard error, and "worst"
  values of 10 real-valued measurements per cell nucleus, e.g. radius,
  texture, perimeter, area, smoothness, concavity)
- **Target:** Binary — `0 = malignant`, `1 = benign`
- **Split used:** 80% train / 20% test, stratified by class (455 train / 114 test)
- **Preprocessing:** Features standardized with `StandardScaler` (fit on
  train only, to avoid data leakage)

> This dataset meets the assignment's minimum requirements (≥500 instances,
> ≥12 features). To substitute a different Kaggle/UCI dataset, replace the
> `load_breast_cancer()` call in `models/MultiClassificationModel.py` with
> `pd.read_csv("your_dataset.csv")` — the rest of the pipeline is dataset-agnostic
> as long as the target column is named `target`.

## c. GitHub Repository Link

[Machine_Learning_Assignment-2](https://github.com/IndrajeetJha90/Machine_Learning_Assignment-2)

## d. Models Used

### Comparison Table (test set, 114 held-out samples)

| ML Model Name             | Accuracy | AUC    | Precision | Recall | F1     | MCC    |
|----------------------------|----------|--------|-----------|--------|--------|--------|
| Logistic Regression        | 0.9825   | 0.9954 | 0.9861    | 0.9861 | 0.9861 | 0.9623 |
| Decision Tree               | 0.9123   | 0.9157 | 0.9559    | 0.9028 | 0.9286 | 0.8174 |
| kNN                         | 0.9561   | 0.9788 | 0.9589    | 0.9722 | 0.9655 | 0.9054 |
| Naive Bayes                 | 0.9298   | 0.9868 | 0.9444    | 0.9444 | 0.9444 | 0.8492 |
| Random Forest (Ensemble)    | 0.9561   | 0.9932 | 0.9589    | 0.9722 | 0.9655 | 0.9054 |

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Best overall performer across every metric (Accuracy 0.983, MCC 0.962). After scaling, the classes are close to linearly separable, which favors a linear decision boundary and gives it a strong edge. |
| Decision Tree | Weakest performer (Accuracy 0.912, MCC 0.817). A single unpruned tree overfits the training split and is sensitive to small variations in the data, hurting recall on the malignant class in particular. |
| kNN | Strong performer (Accuracy 0.956, F1 0.966). Distance-based classification works well once features are standardized, since all features then contribute comparably to the distance metric. |
| Naive Bayes | Moderate accuracy (0.930) but a high AUC (0.987), showing it ranks predictions well even though its independence assumption between the 30 correlated features costs it some precision/recall at the default 0.5 threshold. |
| Random Forest (Ensemble) | Matches kNN on Accuracy/F1 (0.956/0.966) but has the highest AUC among tree-based models (0.993). Ensembling many trees reduces the variance/overfitting problem seen in the single Decision Tree. |
| **Overall Winner for this dataset** | **Logistic Regression** — highest score on 5 of 6 metrics (Accuracy, Precision, Recall, F1, MCC) and second-highest AUC, at a fraction of the computational cost of the ensemble model. |

## Repository Structure

```
project-folder/
│-- streamlit_app.py                  # Streamlit app
│-- requirements.txt
│-- README.md
│-- test_data.csv                     # held-out test split used for evaluation/upload
│-- metrics_comparison.csv            # raw comparison table (machine-readable)
└── models/
    │-- MultiClassificationModel.py   # trains all 5 models + saves artifacts
    │-- logistic_regression.joblib
    │-- decision_tree.joblib
    │-- knn.joblib
    │-- naive_bayes.joblib
    │-- random_forest_ensemble.joblib
    │-- scaler.joblib
    │-- feature_names.joblib
    │-- target_names.joblib
    └── metrics.json
```

## How to Run Locally

```bash
pip install -r requirements.txt
python models/MultiClassificationModel.py   # regenerates models/test_data.csv/metrics (already included)
streamlit run streamlit_app.py
```

## Live App

`<< PASTE YOUR STREAMLIT COMMUNITY CLOUD APP URL HERE >>`
