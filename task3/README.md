# Task 3: Enhanced Depression Prediction - Comparative Technical Breakdown

This document outlines the key technical differences between the **Task 2 Replication Code** and the **Task 3 Enhanced Pipeline**.

## 1. Data Processing Improvements
| Feature | Task 2 (Replication) | Task 3 (Enhanced) | Benefit |
|---------|-----------------------|-------------------|---------|
| **Dataset Size** | ~10,000 participants | **40,000+ participants** | Better generalization and more complex patterns. |
| **Imbalance Handling** | Random Undersampling | **SMOTE (Oversampling)** | Retains all majority clinical data while generating synthetic minority cases. |
| **Data Split** | Standard Split | **No-Leakage Stratified Split** | Prevents imputer/scaler information from leaking into the test set. |

## 2. Feature Engineering (The Major Shift)
Task 2 used only raw NHANES variables. Task 3 introduced **15+ engineered features**:
- **Interactions**: Age-BMI, Glucose-Age, Cholesterol-BMI.
- **Categorical Binning**: BMI Categories (Obese/Underweight), Age Groups (Senior/Adult).
- **Clinical Comorbidity**: A summation of 7 chronic conditions (Asthma, Arthritis, etc.) into a single `comorbidity_count` score.

## 3. Modeling & Hardware Optimization
| Component | Task 2 (Replication) | Task 3 (Enhanced) |
|-----------|-----------------------|-------------------|
| **Hardware** | Standard CPU | **Dual T4 GPU (Kaggle)** |
| **Algorithms** | Sklearn (Sequential) | **XGBoost (GPU_Hist) & cuML (Random Forest GPU)** |
| **Ensemble** | Single Models | **Soft-Voting Ensemble (RF + XGB + GB)** |
| **Optimization** | Minimal Tuning | **GPU-Accelerated RandomizedSearchCV (200+ fits)** |

## 4. Evaluation Efficiency
Task 3 introduced a **Post-Training Threshold Search**. While Task 2 relied on the standard 0.5 probability cutoff, Task 3 scans 100+ thresholds to pinpoint the "Sweet Spot" that maximizes both Accuracy and F1-Score simultaneously.

---
## 5. Techniques Comparison: Original vs Enhanced

| Technique | Original Paper | Task 3 (Enhanced) | Why It's Better |
|-----------|----------------|-------------------|-----------------|
| **Data Imbalance** | Undersampling / None | **SMOTE + Dynamic Weighting** | SMOTE prevents data loss (undersampling throws away data). Dynamic weighting forces the model to focus on the 8% minority signal. |
| **Feature Depth** | 8-10 raw features | **25+ Engineered Features** | Captures non-linear biological intersections (e.g., how Age affects the impact of BMI on Depression). |
| **Optimization** | Grid Search (CPU) | **Randomized Search (GPU)** | GPU allows 10x more iterations (200+ fits), finding nuances in high-dimensional space that CPU grid search misses. |
| **Ensemble Logic**| Single Best Model | **Soft-Voting Ensemble** | Combines the stability of RF, the speed of XGB, and the precision of GB to reduce variance. |
| **Classification** | Fixed Threshold (0.5) | **Threshold Optimization** | Clinically, a 0.5 cutoff is often wrong for rare diseases. Our scan finds the "Maximum F1" point, boosting recall by ~20%. |

*For a detailed statistical analysis and visualizations, see the "Improvisation Report" PDF.*
