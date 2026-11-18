# untuk menyimpan file latih.py ke disk Colab
import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

# berfungsi untuk mengimpor fungsi bersihkan_data_awal dan
# class BreastCancerFeatureEngineer dari file fitur.py. Saat Perintah `!python latih.py`
from fitur import bersihkan_data_awal, BreastCancerFeatureEngineer

# 1. Load & Audit Awal Data
DATA_PATH = os.path.join("dataset", "kasus02_breast_cancer.csv")
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "model_breast_cancer.joblib")

df = pd.read_csv(DATA_PATH)
df = bersihkan_data_awal(df)

# Encoding target: Ganas = 1 (kelas positif medis), Jinak = 0
target_col = "diagnosis"
y = df[target_col].map({"Ganas": 1, "Jinak": 0})
X = df.drop(columns=[target_col])

# 2. Split Train & Test SEBELUM preprocessing (mencegah data leakage)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Definisi Pipeline untuk Minimal Dua Algoritma
pipelines = {
    "Logistic Regression": Pipeline([
        ("fe", BreastCancerFeatureEngineer(drop_redundant=True)),
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, random_state=42))
    ]),
    "Random Forest": Pipeline([
        ("fe", BreastCancerFeatureEngineer(drop_redundant=True)),
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(n_estimators=200, random_state=42))
    ])
}

# 4. Pemilihan Model Menggunakan Cross-Validation di Data Train SAJA
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scoring = ["accuracy", "precision", "recall", "f1", "roc_auc"]

cv_summary = {}
best_model_name = None
best_cv_recall = -1.0

print("=== HASIL 5-FOLD CROSS-VALIDATION (DATA TRAIN) ===")
for name, pipe in pipelines.items():
    scores = cross_validate(pipe, X_train, y_train, cv=cv, scoring=scoring)
    mean_acc = scores["test_accuracy"].mean()
    mean_prec = scores["test_precision"].mean()
    mean_rec = scores["test_recall"].mean()
    mean_f1 = scores["test_f1"].mean()
    mean_auc = scores["test_roc_auc"].mean()

    cv_summary[name] = {
        "Accuracy": round(float(mean_acc), 4),
        "Precision": round(float(mean_prec), 4),
        "Recall": round(float(mean_rec), 4),
        "F1-Score": round(float(mean_f1), 4),
        "ROC-AUC": round(float(mean_auc), 4)
    }
    print(f"{name:20s} -> Recall: {mean_rec:.4f} | F1: {mean_f1:.4f} | ROC-AUC: {mean_auc:.4f} | Acc: {mean_acc:.4f}")

    # Pilih model terbaik berdasarkan Recall tertinggi (prioritas medis)
    if mean_rec > best_cv_recall:
        best_cv_recall = mean_rec
        best_model_name = name

print(f"\nModel Terpilih Berdasarkan CV Recall Tertinggi: {best_model_name}")

# 5. Latih Model Terbaik pada Seluruh Data Train & Evaluasi 1x pada Data Test
best_pipeline = pipelines[best_model_name]
best_pipeline.fit(X_train, y_train)

y_pred = best_pipeline.predict(X_test)
y_proba = best_pipeline.predict_proba(X_test)[:, 1]

test_metrics = {
    "Accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
    "Precision": round(float(precision_score(y_test, y_pred)), 4),
    "Recall": round(float(recall_score(y_test, y_pred)), 4),
    "F1-Score": round(float(f1_score(y_test, y_pred)), 4),
    "ROC-AUC": round(float(roc_auc_score(y_test, y_proba)), 4),
    "Confusion_Matrix": confusion_matrix(y_test, y_pred).tolist()
}

print("\n=== HASIL EVALUASI FINAL PADA DATA TEST (1x EVALUASI) ===")
print(classification_report(y_test, y_pred, target_names=["Jinak (0)", "Ganas (1)"]))
print("Ringkasan Metrik Test:", test_metrics)

# 6. Simpan Pipeline Lengkap + Metadata ke models/
os.makedirs(MODEL_DIR, exist_ok=True)
artifact = {
    "model_name": best_model_name,
    "pipeline": best_pipeline,
    "feature_columns": list(X_train.columns),
    "feature_defaults": X_train.median().to_dict(),
    "cv_results": cv_summary,
    "test_metrics": test_metrics,
    "label_mapping": {0: "Jinak", 1: "Ganas"}
}

joblib.dump(artifact, MODEL_PATH)
print(f"\nPipeline & metadata berhasil disimpan ke: {MODEL_PATH}")
