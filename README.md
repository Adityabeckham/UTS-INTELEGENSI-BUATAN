# LAPORAN UJIAN TENGAH SEMESTER (UTS)
**Mata Kuliah:** Intelegensi Buatan (3 SKS) — Kelas TIF RP 23D  
**Nama:** Aditya Beckham 
**NIM:** 24552011066  
**Kasus Terpilih:** Kasus 2 — Diagnosis Kanker Payudara (`kasus02_breast_cancer.csv`)  

---

## 1. Penentuan Kasus & Audit Data
Berdasarkan perhitungan NIM `24552011066`:
* Jumlah digit NIM: $2 + 4 + 5 + 5 + 2 + 0 + 1 + 1 + 0 + 6 + 6 = 32$
* Modulo 10: $32 \pmod{10} = 2 \rightarrow$ **Kasus 2 (`kasus02_breast_cancer.csv`)**.

**Hasil Audit Dataset:**
* **Dimensi & Tipe Data:** 569 baris dan 31 kolom (30 fitur numerik `float64` dan 1 kolom target kategorik `diagnosis`).
* **Kebersihan Data:** Tidak ditemukan *missing value* (0) maupun baris duplikat (0), serta tidak ada kolom ID yang perlu dibuang.
* **Distribusi Target:** Kelas `Jinak` sebanyak 357 sampel ($62,74\%$) dan kelas `Ganas` sebanyak 212 sampel ($37,26\%$).
* **Multikolinearitas:** Ditemukan korelasi linear sangat tinggi ($r > 0,95$) antarfitur geometris sel, seperti `mean_perimeter` dengan `mean_radius` ($r = 0,9979$), `worst_perimeter` dengan `worst_radius` ($r = 0,9937$), dan `mean_area` dengan `mean_radius` ($r = 0,9874$).

---

## 2. Keputusan Feature Engineering (`fitur.py`)
Seluruh transformasi dibungkus di dalam `Pipeline` scikit-learn dan diterapkan **setelah** pembagian data *train-test* (80:20 *stratified*) untuk mencegah *data leakage*:
1. **Seleksi Fitur (Menangani Korelasi Tinggi):** Membuang 6 fitur redundan (`mean_perimeter`, `mean_area`, `perimeter_error`, `area_error`, `worst_perimeter`, `worst_area`) karena informasinya sudah terwakili oleh fitur `radius`. Strategi seleksi fitur dipilih dibandingkan reduksi dimensi (PCA) agar makna klinis setiap variabel tetap dapat diinterpretasikan secara medis.
2. **Pembuatan Fitur Baru (Rasio Keparahan Sel):**
   * `radius_worst_to_mean_ratio` ($\frac{\text{worst\_radius}}{\text{mean\_radius}}$): Mengukur seberapa ekstrem pembesaran sel terburuk dibandingkan ukuran rata-rata sel pasien.
   * `concavity_worst_to_mean_ratio` ($\frac{\text{worst\_concavity}}{\text{mean\_concavity}}$): Mengukur tingkat keparahan cekungan kontur sel terburuk terhadap kondisi rata-rata.
3. **Standardisasi Skala:** Menggunakan `StandardScaler` karena rentang nilai fitur sangat beragam (misalnya `area` bernilai ratusan, sedangkan `smoothness` berada di kisaran $0,05 - 0,16$).

---

## 3. Perbandingan Model & Pemilihan Metrik Medis
Evaluasi pemilihan model dilakukan menggunakan **5-Fold Stratified Cross-Validation** pada data *train* (455 sampel):

| Algoritma | Accuracy | Precision | Recall (Utama) | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 96,92% | — | 94,12% | 95,80% | 0,9954 |
| **Random Forest** | **97,36%** | — | **94,71%** | **96,39%** | 0,9904 |

**Alasan Pemilihan Metrik & Model:**
* Dalam diagnosis medis kanker payudara, kesalahan **False Negative (FN)**—yakni pasien penderita kanker **Ganas** yang diprediksi **Jinak**—merupakan kesalahan paling fatal karena menunda pengobatan dan membahayakan nyawa pasien.
* Oleh karena itu, **Recall** pada kelas `Ganas` dipilih sebagai metrik utama untuk meminimalkan *False Negative*. Berdasarkan hasil *cross-validation*, **Random Forest** terpilih sebagai model terbaik dengan rata-rata Recall tertinggi (**94,71%**) dan F1-Score (**96,39%**).

---

## 4. Hasil Evaluasi Akhir (Data Test) & Implementasi Aplikasi
Model **Random Forest** terbaik dilatih ulang pada seluruh data *train* dan diuji **satu kali** pada data *test* (114 sampel: 72 Jinak, 42 Ganas):
* **Accuracy:** $97,37\%$
* **Precision (Ganas):** $100,00\%$ (tidak ada pasien Jinak yang salah didiagnosis Ganas / *False Positive* = 0)
* **Recall (Ganas):** $92,86\%$ (39 dari 42 kasus Ganas berhasil terdeteksi, hanya 3 *False Negative*)
* **F1-Score:** $96,30\%$
* **ROC-AUC:** $0,9974$ (kemampuan pemisahan probabilitas antar kelas sangat tinggi)

**Confusion Matrix (Data Test):**
* *True Negative* (Aktual Jinak, Prediksi Jinak): **72**
* *False Positive* (Aktual Jinak, Prediksi Ganas): **0**
* *False Negative* (Aktual Ganas, Prediksi Jinak): **3**
* *True Positive* (Aktual Ganas, Prediksi Ganas): **39**

Model lengkap beserta metadata telah disimpan menggunakan `joblib` ke `models/model_breast_cancer.joblib` dan diintegrasikan ke dalam aplikasi web interaktif berbasis **Streamlit** (`app.py`) yang dijalankan melalui Google Colab (`notebook/kasus02_colab.ipynb`).

### Screenshoot Stremlit
- <img width="1920" height="1395" alt="ss-perbandingan-model-dan-matriks-evaluasi" src="https://github.com/user-attachments/assets/d202e084-7dcb-4772-9515-d9cffe326ab9" />

- <img width="1920" height="2226" alt="ss-prediksi-diagnosis" src="https://github.com/user-attachments/assets/890d173a-c4df-42a5-a166-29ed0b348cad" />
