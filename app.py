import os
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Diagnosis Kanker Payudara - UTS AI", layout="wide")

MODEL_PATH = os.path.join("models", "model_breast_cancer.joblib")

@st.cache_resource
def load_artifact():
    return joblib.load(MODEL_PATH)

st.title("Sistem Klasifikasi Diagnosis Kanker Payudara")
st.caption(" Aditya Beckham - NIM 24552011066 - UTS Intelegensi Buatan (Kasus 2)")

if not os.path.exists(MODEL_PATH):
    st.error("File model tidak ditemukan. Jalankan `python latih.py` terlebih dahulu.")
    st.stop()

artifact = load_artifact()
pipeline = artifact["pipeline"]
feature_cols = artifact["feature_columns"]
defaults = artifact["feature_defaults"]
test_metrics = artifact["test_metrics"]
cv_results = artifact["cv_results"]

tab1, tab2 = st.tabs(["Prediksi Diagnosis Pasien", "Perbandingan Model & Metrik Evaluasi"])

with tab1:
    st.subheader("Input Data Pengukuran Sel (Fitur Mentah)")
    st.info("Nilai default diisi dengan median data latih. Ubah nilai sesuai hasil pengukuran sel pasien.")

    with st.form("form_prediksi"):
        cols = st.columns(3)
        input_data = {}
        for idx, col_name in enumerate(feature_cols):
            with cols[idx % 3]:
                input_data[col_name] = st.number_input(
                    label=col_name,
                    value=float(defaults[col_name]),
                    format="%.5f"
                )
        submitted = st.form_submit_button("Prediksi Diagnosis", type="primary")

    if submitted:
        df_input = pd.DataFrame([input_data])
        pred_class = pipeline.predict(df_input)[0]
        pred_proba = pipeline.predict_proba(df_input)[0]

        label_hasil = artifact["label_mapping"][pred_class]
        prob_jinak = pred_proba[0] * 100
        prob_ganas = pred_proba[1] * 100

        st.divider()
        st.subheader("Hasil Diagnosis Model")
        c1, c2, c3 = st.columns(3)
        c1.metric("Prediksi Kelas", label_hasil)
        c2.metric("Probabilitas Jinak (0)", f"{prob_jinak:.2f}%")
        c3.metric("Probabilitas Ganas (1)", f"{prob_ganas:.2f}%")

        if pred_class == 1:
            st.error("Hasil memprediksi **GANAS (Malignant)**. Segera lakukan konsultasi dan pemeriksaan medis lanjutan.")
        else:
            st.success("Hasil memprediksi **JINAK (Benign)**.")

with tab2:
    st.subheader(f"Model Terpilih: {artifact['model_name']}")

    st.markdown("**1. Perbandingan 5-Fold Cross-Validation (Data Train)**")
    df_cv = pd.DataFrame(cv_results).T
    st.dataframe(df_cv, use_container_width=True)

    st.markdown("**2. Hasil Evaluasi Akhir pada Data Test (1x Evaluasi)**")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Recall (Utama)", f"{test_metrics['Recall']*100:.2f}%")
    m2.metric("F1-Score", f"{test_metrics['F1-Score']*100:.2f}%")
    m3.metric("Precision", f"{test_metrics['Precision']*100:.2f}%")
    m4.metric("Accuracy", f"{test_metrics['Accuracy']*100:.2f}%")
    m5.metric("ROC-AUC", f"{test_metrics['ROC-AUC']:.4f}")

    st.markdown("**Confusion Matrix (Data Test)**")
    cm = test_metrics["Confusion_Matrix"]
    df_cm = pd.DataFrame(
        cm,
        index=["Aktual: Jinak (0)", "Aktual: Ganas (1)"],
        columns=["Prediksi: Jinak (0)", "Prediksi: Ganas (1)"]
    )
    st.table(df_cm)
