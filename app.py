import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# --- Konfigurasi Halaman ---
st.set_page_config(
    page_title="Prediksi Harga Rumah Depok",
    page_icon="🏠",
    layout="centered",
)

# --- Load Model ---
MODEL_PATH = Path(__file__).parent / "model_harga_rumah.joblib"


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None
    try:
        return joblib.load(MODEL_PATH)
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        return None


model = load_model()

# --- Header ---
st.title("🏠 Prediksi Harga Rumah Kota Depok")
st.caption("Estimasi harga penawaran rumah berdasarkan model Random Forest — EarlyBird Bengkod Project")

if model is None:
    st.error("File model `model_harga_rumah.joblib` tidak ditemukan. Jalankan notebook terlebih dahulu.")
    st.stop()

st.divider()

# --- Form Input ---
st.subheader("Masukkan Spesifikasi Rumah")

col1, col2 = st.columns(2)

with col1:
    kamar_tidur = st.number_input("Kamar Tidur", min_value=0, max_value=12, value=3, step=1)
    kamar_mandi = st.number_input("Kamar Mandi", min_value=0, max_value=10, value=2, step=1)
    garasi = st.number_input("Garasi", min_value=0, max_value=10, value=1, step=1)

with col2:
    luas_tanah = st.number_input("Luas Tanah (m²)", min_value=10.0, max_value=2000.0, value=100.0, step=5.0)
    luas_bangunan = st.number_input("Luas Bangunan (m²)", min_value=10.0, max_value=1500.0, value=80.0, step=5.0)

    daftar_kecamatan = [
        "Beji", "Cilodong", "Cimanggis", "Cinere", "Cipayung",
        "Limo", "Pancoran Mas", "Sawangan", "Sukmajaya", "Tapos",
    ]
    kecamatan = st.selectbox("Kecamatan", daftar_kecamatan, index=3)

st.divider()

# --- Prediksi ---
if st.button("🔍 Prediksi Harga", use_container_width=True, type="primary"):
    input_data = pd.DataFrame([{
        "Kamar Tidur_num": kamar_tidur,
        "Kamar Mandi_num": kamar_mandi,
        "Garasi_num": garasi,
        "Luas Tanah_num": luas_tanah,
        "Luas Bangunan_num": luas_bangunan,
        "Kecamatan": kecamatan,
    }])

    try:
        prediksi = float(model.predict(input_data)[0])

        if prediksi <= 0:
            st.warning("Model menghasilkan estimasi tidak valid. Coba ubah spesifikasi.")
        else:
            # Format harga
            harga_rp = f"Rp {prediksi:,.0f}".replace(",", ".")

            if prediksi >= 1_000_000_000:
                harga_singkat = f"≈ {prediksi / 1_000_000_000:.2f} Miliar"
            else:
                harga_singkat = f"≈ {prediksi / 1_000_000:.0f} Juta"

            st.success("Estimasi berhasil!")
            st.metric(label="Estimasi Harga Properti", value=harga_rp, delta=harga_singkat)

            # Ringkasan spesifikasi
            st.caption(
                f"{kamar_tidur} KT · {kamar_mandi} KM · {garasi} Garasi · "
                f"LT {luas_tanah:.0f} m² · LB {luas_bangunan:.0f} m² · {kecamatan}"
            )

            st.info("ℹ️ Hasil ini merupakan estimasi model, bukan jaminan harga transaksi.")

    except Exception as e:
        st.error(f"Prediksi gagal: {e}")
