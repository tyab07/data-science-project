# NHANES Depression Prediction Replication

This repository contains a replication of the journal paper:
**"Prediction of depressive disorder using machine learning approaches: findings from the NHANES"** by Thien Vu et al. (2025).

## Objective
The goal of this project is to replicate the findings of the original study using machine learning to predict depressive disorders based on NHANES data.

## Methodology
- **Data Source:** NHANES (surrogate 2005-2018 dataset).
- **Models:** Logistic Regression, Random Forest, Naive Bayes, SVM, XGBoost, and LightGBM.
- **Preprocessing:** Median imputation, One-hot encoding, and Random Undersampling for class imbalance.
- **Explainability:** SHAP (Shapley Additive Explanations).

## Features Used
- PIR (Family income to poverty ratio)
- Gender, BMI, Age, Marital Status
- Blood Glucose, Creatinine
- Hypertension (derived)
