# untuk menyimpan file fitur.py ke disk Colab
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

def bersihkan_data_awal(df: pd.DataFrame) -> pd.DataFrame:
    """
    Membersihkan baris duplikat dan membuang kolom ID/kosong jika ada.
    """
    df_clean = df.copy()
    df_clean = df_clean.drop(columns=["id", "Unnamed: 32"], errors="ignore")
    df_clean = df_clean.drop_duplicates().reset_index(drop=True)
    return df_clean

class BreastCancerFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Transformer Feature Engineering untuk Kasus 2 (Diagnosis Kanker Payudara).
    """
    def __init__(self, drop_redundant: bool = True):
        self.drop_redundant = drop_redundant
        self.redundant_cols = [
            "mean_perimeter", "mean_area",
            "perimeter_error", "area_error",
            "worst_perimeter", "worst_area"
        ]

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_out = X.copy()

        # 1. Buat fitur baru: Rasio keparahan sel terburuk dibanding rata-rata
        if "worst_radius" in X_out.columns and "mean_radius" in X_out.columns:
            X_out["radius_worst_to_mean_ratio"] = X_out["worst_radius"] / (X_out["mean_radius"] + 1e-5)

        if "worst_concavity" in X_out.columns and "mean_concavity" in X_out.columns:
            X_out["concavity_worst_to_mean_ratio"] = X_out["worst_concavity"] / (X_out["mean_concavity"] + 1e-5)

        # 2. Hapus fitur redundan akibat korelasi tinggi (> 0.95)
        if self.drop_redundant:
            cols_to_drop = [col for col in self.redundant_cols if col in X_out.columns]
            X_out = X_out.drop(columns=cols_to_drop)

        return X_out
