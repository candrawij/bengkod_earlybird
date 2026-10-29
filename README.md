# 🏠 Prediksi Harga Rumah Kota Depok (EarlyBird Bengkod)

Proyek *Machine Learning* terintegrasi (*end-to-end*) untuk estimasi harga penawaran rumah tinggal di wilayah Kota Depok berbasis data historis listing properti.

---

## 📌 Ringkasan Proyek
- **Tipe Masalah**: Regresi (*Supervised Machine Learning*)
- **Dataset**: `Raw-Final-Rumah.csv` (Listing properti Kota Depok hasil web scraping)
- **Model Terbaik**: Tuned **Random Forest Regressor** dengan **Transformasi Target Logaritmik** (`TransformedTargetRegressor`)
- **Hasil Evaluasi**:
  - **$R^2$ Score**: **0.8448** (84,5% variansi harga berhasil dijelaskan)
  - **MAE**: **Rp 419.824.542**
  - **MAPE**: **20.00%** (standar industri penilaian properti)
- **Aplikasi Web**: Dashboard interaktif berbasis **Streamlit**

---

## 📁 Struktur Berkas Utama
```text
Bengkod/
├── app.py                         # Kode aplikasi web interaktif Streamlit
├── EarlyBird_Bengkod_Final.ipynb  # Notebook analisis dan pemodelan Machine Learning
├── model_harga_rumah.joblib       # Model Machine Learning terlatih siap pakai
├── Raw-Final-Rumah.csv            # Dataset mentah listing properti Kota Depok
├── requirements.txt               # Daftar dependensi Python untuk deployment
├── README.md                      # Dokumentasi utama proyek
└── .gitignore                     # Konfigurasi file yang diabaikan oleh Git
```

---

## 🚀 Panduan Menjalankan Aplikasi
1. **Pasang Dependensi**:
   ```bash
   pip install -r requirements.txt
   ```
2. **Jalankan Aplikasi Web Streamlit**:
   ```bash
   python -m streamlit run app.py
   ```
3. Buka peramban di `http://localhost:8501`.
