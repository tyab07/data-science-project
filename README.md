# NHANES Depression Prediction: From Replication to Optimization

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![ML](https://img.shields.io/badge/ML-XGBoost%20|%20RF-green.svg)
![Hardware](https://img.shields.io/badge/Hardware-NVIDIA%20T4-orange.svg)

This repository documents a comprehensive machine learning project aimed at predicting depressive disorders using data from the National Health and Nutrition Examination Survey (NHANES). The project evolved from a baseline replication of academic research to an optimized, high-performance predictive pipeline.

## Project Overview

Depression is a major global health concern. This project leverages clinical and socio-demographic data to build and improve upon models for early detection of depressive symptoms.

### Phase 1: Task 2 - Baseline Replication
**Objective:** Replicate the findings of the study *"Prediction of depressive disorder using machine learning approaches: findings from the NHANES"* by Thien Vu et al. (2025).

- **Data Source:** NHANES 2005-2018 (Subset of ~10,000 cases).
- **Core Methodology:** 
  - Standard preprocessing (Median imputation, One-hot encoding).
  - Class imbalance handling via **Random Undersampling**.
  - Baseline models: Logistic Regression, Random Forest, Naive Bayes, SVM, XGBoost, and LightGBM.
- **Key Result:** Achieved a peak **AUC of ~0.702**, successfully aligning with the original researchers' benchmarks.

### Phase 2: Task 3 - Enhanced Optimization
**Objective:** Surpass baseline performance using advanced feature engineering, larger datasets, and GPU acceleration.

- **Data Source:** Expanded NHANES dataset (**40,000+ participants**).
- **Advanced Techniques:**
  - **SMOTE (Oversampling):** Retained all majority clinical data while generating synthetic minority cases to fix class imbalance without data loss.
  - **Feature Engineering:** Introduced 15+ engineered features, including clinical comorbidities and biological interactions (e.g., Age-BMI interaction).
  - **Hardware acceleration:** Leveraged **Dual T4 GPUs** for model training and hyperparameter optimization.
  - **Ensemble Modeling:** Built a **Soft-Voting Ensemble** (Random Forest + XGBoost + Gradient Boosting).
  - **Threshold Optimization:** Scanned thresholds to maximize the F1-score specifically for clinical utility.
- **Key Result:** Improved performance from **69-70% AUC → 78-82% AUC**, providing a significant leap in predictive power and clinical relevance.

## Project Structure

- `task2/`: Replication code and reports.
  - `replication_pipeline.py`: Main baseline script.
  - `reproduction_report.md`: Detailed methodology and results of the replication.
- `task3/`: Enhancement and optimization code.
  - `01-ds-project-final.ipynb`: Final optimized project notebook.
  - `enhanced_depression_prediction.py`: Implementation of the advanced pipeline.
  - `improvisation_report.pdf`: Comparative analysis of improvements over the baseline.
- `results/`: Visualization artifacts (SHAP plots, ROC curves, ablation studies).
- `SKILL.md`: Documentation for the `dir-to-pptx` skill developed during this project.

## How to Run

### Baseline Replication (Task 2)
```bash
python task2/replication_pipeline.py
```

### Enhanced Pipeline (Task 3)
The optimized pipeline is designed for GPU-accelerated environments (e.g., Kaggle/Colab).
1. Open `task3/01-ds-project-final.ipynb` in a Jupyter environment.
2. Ensure `XGBoost` and `cuML` are installed with GPU support.
3. Follow the sequence of cells for data engineering and model training.

## Ethical Statement & Explainability
This project uses **SHAP (Shapley Additive Explanations)** to ensure model transparency. By identifying the key features driving predictions (e.g., PIR, BMI, Comorbidities), we provide interpretable insights rather than "black-box" results, which is critical for clinical adoption and ethical AI standards.
