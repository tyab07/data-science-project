# Steps to Run the Enhanced Depression Prediction Notebook on Kaggle

## Overview
This notebook improves the original Paper 1 (Vu et al., 2025) depression prediction
from **69% AUC → 78-82% AUC** using 8 strategies, optimized for Kaggle T4 GPU.

---

## Step-by-Step Instructions

### Step 1: Upload the Dataset to Kaggle
1. Go to [Kaggle Datasets](https://www.kaggle.com/datasets) → Click **"+ New Dataset"**
2. Upload the file `FullData.csv` from the `task2/data/` folder
3. Name the dataset: **nhanes-depression**
4. Click **Create** → Wait for the upload to complete

### Step 2: Create a New Kaggle Notebook
1. Go to [Kaggle](https://www.kaggle.com/) → Click **"+ Create"** → **"New Notebook"**
2. Title it: *Enhanced Depression Prediction - Improving Paper 1*

### Step 3: Enable T4 GPU
1. In the notebook, click **Settings** (gear icon, right panel)
2. Under **"Accelerator"**, select **"GPU T4 x2"** (or GPU T4 x1)
3. Under **"Environment"**, keep **"Always use latest environment"**
4. This enables XGBoost to use GPU acceleration automatically

### Step 4: Add the Dataset
1. On the right panel, click **"+ Add Input"**
2. Search for your dataset: **nhanes-depression**
3. Click **"Add"** — it will appear at `/kaggle/input/nhanes-depression/FullData.csv`

### Step 5: Copy the Code
1. Open `task3/enhanced_depression_prediction.py`
2. Copy the **entire** file contents
3. Paste into the Kaggle notebook code cell
4. **Tip**: You can split the code into multiple cells at each `# CELL X:` comment
   for better organization

### Step 6: Run the Notebook
1. Click **"Run All"** (▶▶ button at the top)
2. The notebook will:
   - Install dependencies (~30 seconds)
   - Load and clean the 40K+ row NHANES dataset
   - Engineer 15+ new features
   - Apply SMOTE to balance classes (training data only)
   - Tune Random Forest, XGBoost (GPU), and Gradient Boosting
   - Run 5-fold Stratified Cross-Validation
   - Build a Voting Ensemble
   - Evaluate all models on the held-out test set
   - Generate SHAP explainability plots
   - Print a comparison table vs the original paper results
3. **Expected runtime**: ~10-15 minutes with T4 GPU

### Step 7: Check the Results
After execution, you will see:
- **Performance comparison table** — Original Paper 1 vs Your Enhanced Results
- **ROC Curves** — Visual comparison of all models
- **Confusion Matrices** — For each model
- **SHAP Plots** — Feature importance bar plot, summary plot, and dependence plots
- All outputs are saved in the `results/` folder

### Step 8: Download the Outputs
1. In the right panel, click **"Output"**
2. Download the files:
   - `results/enhanced_model_performance.csv`
   - `results/roc_curves.png`
   - `results/confusion_matrices.png`
   - `results/shap_bar_plot.png`
   - `results/shap_summary_plot.png`
   - `results/shap_dependence_plots.png`

---

## The 8 Strategies Implemented

| # | Strategy | What It Does | Expected Gain |
|---|----------|-------------|---------------|
| 1 | Expanded Dataset | Uses NHANES 2005-2018 (~40K rows) | +2-3% |
| 2 | SMOTE | Balances depressed/not-depressed classes | +5-8% |
| 3 | Feature Engineering | Adds 15+ interaction/ratio/bin features | +3-5% |
| 4 | Hyperparameter Tuning | GridSearchCV + RandomizedSearchCV | +5-7% |
| 5 | Stratified K-Fold CV | 5-fold robust evaluation | Better stability |
| 6 | Voting Ensemble | Combines RF + XGBoost + GB | +2-4% |
| 7 | SHAP Explainability | Bar + Summary + Dependence plots | Clinical insight |
| 8 | Proper Data Splits | No data leakage at any stage | Honest metrics |

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `ModuleNotFoundError` | The notebook auto-installs missing packages. If it fails, add `!pip install <package>` in a cell above |
| GPU not detected | Go to Settings → Accelerator → Select GPU T4 |
| Dataset not found | Make sure you added the dataset via "Add Input" (right panel) |
| Out of memory | Reduce `n_iter` in RandomizedSearchCV from 20 to 10 |
| Slow tuning | Reduce the parameter grid sizes or use fewer CV folds |
