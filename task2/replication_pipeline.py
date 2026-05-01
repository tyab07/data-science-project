import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, roc_auc_score, f1_score, confusion_matrix
import shap
import matplotlib.pyplot as plt
import os

def load_and_preprocess(filepath='data/nhanes_2005_2018.csv'):
    df = pd.read_csv(filepath, low_memory=False)
    df.columns = df.columns.str.strip()
    
    desired_features = [
        'ratio_family_income_poverty', 'household_income', 'PIR',
        'gender', 'Gender',
        'BMI', 'bmi',
        'education', 'education_level', 'Education',
        'glucose', 'Glucose',
        'age', 'Age',
        'marital_status', 'MaritalStatus',
        'creatinine', 'Creatinine'
    ]
    
    depression_col = 'depression' if 'depression' in df.columns else 'Depression'
    df = df.dropna(subset=[depression_col])
    
    sys_col = next((c for c in ['systolic_bp', 'SystolicBP'] if c in df.columns), None)
    dia_col = next((c for c in ['diastolic_bp', 'DiastolicBP'] if c in df.columns), None)
    
    if sys_col and dia_col:
        df['hypertension'] = ((df[sys_col] >= 130) | (df[dia_col] >= 80)).astype(int)
        desired_features.append('hypertension')
    elif 'health_problem_Blood Pressure' in df.columns:
        # Use existing feature if sys/dia not found
        df['hypertension'] = df['health_problem_Blood Pressure']
        desired_features.append('hypertension')
    
    available_feat = [c for c in desired_features if c in df.columns]
    
    X = df[available_feat].copy()
    X = pd.get_dummies(X, drop_first=True)
    y = (df[depression_col].astype(str).str.lower() != 'not depressed').astype(int)
    
    # Handle missing values: median imputation for numerical
    imputer = SimpleImputer(strategy='median')
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns, index=X.index)
    
    # Random undersampling to handle class imbalance
    rus = RandomUnderSampler(random_state=42)
    X_resampled, y_resampled = rus.fit_resample(X_imputed, y)
    
    # Standarize features
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X_resampled), columns=X.columns, index=X_resampled.index)
    
    return X_scaled, y_resampled

def evaluate_models(X, y):
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42),
        'Naive Bayes': GaussianNB(),
        'SVM': SVC(probability=True, random_state=42),
        'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
        'LightGBM': LGBMClassifier(random_state=42, verbose=-1)
    }
    
    results = []
    
    # Split for SHAP evaluation later
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else y_pred
        
        acc = accuracy_score(y_test, y_pred)
        sens = recall_score(y_test, y_pred)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        spec = tn / (tn + fp)
        prec = precision_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        f1 = f1_score(y_test, y_pred)
        
        results.append({
            'Model': name,
            'Accuracy': acc,
            'Sensitivity': sens,
            'Specificity': spec,
            'Precision': prec,
            'AUC': auc,
            'F1_Score': f1
        })
        
    df_results = pd.DataFrame(results)
    
    print("\n--- Model Performance Comparison ---")
    print(df_results.to_string(index=False))
    
    # Save the results
    os.makedirs('results', exist_ok=True)
    df_results.to_csv('results/model_performance.csv', index=False)
    
    return df_results, models['LightGBM'], X_train, X_test

def generate_shap_plots(best_model, X_test, output_dir='results'):
    os.makedirs(output_dir, exist_ok=True)
    print("Generating SHAP plots...")
    
    explainer = shap.TreeExplainer(best_model)
    shap_values = explainer(X_test)
    
    # SHAP feature importance plot (bar)
    plt.figure()
    shap.plots.bar(shap_values, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'shap_bar_plot.png'))
    plt.close()
    
    # SHAP summary plot (beeswarm)
    plt.figure()
    shap.summary_plot(shap_values, X_test, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'shap_summary_plot.png'))
    plt.close()
    print("SHAP plots saved in 'results' folder.")

if __name__ == "__main__":
    X, y = load_and_preprocess()
    results_df, xgb_model, X_train, X_test = evaluate_models(X, y)
    generate_shap_plots(xgb_model, X_test)
