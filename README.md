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

## Results Summary
The replication achieved a peak AUC of **0.702 (SVM)** and **0.695 (LightGBM)**, aligning with the findings of the original paper which reported stability around the ~0.7 AUC constraint using clinical and socio-demographic parameters.

## Project Structure
- `task2/download_dataset.py`: Script to fetch/prepare data.
- `task2/replication_pipeline.py`: Main ML pipeline.
- `task2/reproduction_report.md`: Detailed methodology and results report.
- `task2/generate_pdf.py`: Comparison generator script.

## How to Run
```bash
python task2/replication_pipeline.py
```
Check the `results/` directory for metrics and SHAP plots.
