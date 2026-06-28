# 🚀 AI4I Predictive Maintenance using Machine Learning

## 📌 Project Overview

This project develops a machine learning model to predict machine failures using the AI4I 2020 Predictive Maintenance dataset. Early prediction of machine failures helps industries reduce downtime, optimize maintenance schedules, and improve operational efficiency.

The project includes complete data preprocessing, feature engineering, handling class imbalance using SMOTE, model comparison, hyperparameter tuning, model interpretation using SHAP, and robustness evaluation.

---

## 📂 Dataset

**Dataset:** AI4I 2020 Predictive Maintenance Dataset

### Features

- UDI
- Product ID
- Type
- Air Temperature [K]
- Process Temperature [K]
- Rotational Speed [rpm]
- Torque [Nm]
- Tool Wear [min]

### Target

- **Machine Failure**
  - 0 → No Failure
  - 1 → Machine Failure

---

# Project Workflow

```
Dataset
    │
    ▼
Data Preprocessing
    │
    ▼
Feature Engineering
    │
    ▼
SMOTE (Class Balancing)
    │
    ▼
Train-Test Split
    │
    ▼
Random Forest & LightGBM
    │
    ▼
Hyperparameter Tuning
    │
    ▼
Model Evaluation
    │
    ▼
SHAP Analysis
    │
    ▼
Robustness Testing
```

---

# Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- LightGBM
- SHAP
- Matplotlib
- Joblib

---

# Data Preprocessing

- Removed unnecessary columns
- Label Encoding
- Feature selection
- Cleaned feature names
- Train-Test Split (80:20)

---

# Feature Engineering

Additional features created include:

- Temperature Difference
- Power Feature
- Wear Rate
- Temperature Ratio
- Torque-RPM Ratio

---

# Handling Class Imbalance

SMOTE (Synthetic Minority Oversampling Technique) was applied only to the training dataset to improve the prediction of machine failure cases.

---

# Machine Learning Models

- Random Forest
- LightGBM

---

# Hyperparameter Tuning

GridSearchCV was used for hyperparameter optimization.

Optimized parameters include:

- Number of Estimators
- Learning Rate
- Maximum Depth
- Number of Leaves
- Subsample Ratio

---

# Model Performance

| Metric | Random Forest | LightGBM |
|---------|--------------:|---------:|
| Accuracy | 99.90% | 99.90% |
| Precision | 97% | 97% |
| Recall | 97% | 97% |
| F1 Score | 97% | 97% |

---

# Cross-Validation (Robustness)

| Metric | Value |
|---------|-------|
| Mean Accuracy | **99.90%** |
| Accuracy Std. Dev. | **0.0000** |
| Mean F1 Score | **98.50%** |
| F1 Std. Dev. | **0.00009** |

The cross-validation results demonstrate that the tuned LightGBM model maintains highly consistent performance across different folds, indicating good robustness and generalization capability.

---

# SHAP Analysis

SHAP (SHapley Additive exPlanations) was used to interpret the predictions made by the tuned LightGBM model.

Generated visualizations include:

- SHAP Summary Plot
- Feature Importance Plot
- Beeswarm Plot

---

# Project Structure

```
AI4I-Predictive-Maintenance/
│
├── dataset/
│      project1_feature_engineered.csv
│
├── hyperparameter_tuning.py
├── final_evaluation_lightgbm.py
├── shap_analysis_lightgbm.py
├── robustness_testing_lightgbm.py
│
├── lightgbm_tuned_model.pkl
│
├── README.md
│
└── requirements.txt
```

---

# Results

- Excellent predictive performance
- High F1-score for failure detection
- Stable performance across cross-validation folds
- Model interpretation using SHAP
- Robust evaluation using cross-validation

---

# Future Improvements

- XGBoost implementation
- CatBoost comparison
- Deep Learning models
- Real-time predictive maintenance dashboard
- Model deployment using Flask or Streamlit

---

# Author

**Sherin John D C**

M.Tech in Electronics

Diploma in Data Science & AI

Machine Learning | Data Science | Predictive Analytics

---

## ⭐ If you found this project useful, please consider giving it a star.
