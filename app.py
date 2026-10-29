import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# --- Page Configuration ---
st.set_page_config(
    page_title="Prediksi Harga Rumah Depok | EarlyBird Bengkod",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom Styling ---
st.markdown("""
<style>
    /* Global font & background enhancements */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    .main-header {
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 50%, #06b6d4 100%);
        padding: 2.2rem 2.5rem;
        border-radius: 16px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 10px 25px -5px rgba(30, 58, 138, 0.25);
    }
    .main-header h1 {
        color: white !important;
        font-weight: 800;
        font-size: 2.2rem;
        margin-bottom: 0.4rem;
        letter-spacing: -0.5px;
    }
    .main-header p {
        color: #e0f2fe;
        font-size: 1.05rem;
        margin-bottom: 0;
        max-width: 800px;
        line-height: 1.5;
    }

    /* Metric cards */
    .metric-card {
        background: #ffffff;
        border-radius: 14px;
        padding: 1.3rem 1.5rem;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.08);
    }
    .metric-label {
        font-size: 0.85rem;
        font-weight: 600;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .metric-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #0f172a;
        margin-top: 0.25rem;
    }

    /* Result Card */
    .result-box {
        background: linear-gradient(145deg, #0f172a 0%, #1e293b 100%);
        color: white;
        padding: 2rem;
        border-radius: 16px;
        border: 1px solid #334155;
        box-shadow: 0 20px 25px -5px rgba(15, 23, 42, 0.3);
        margin-top: 1.5rem;
    }
    .result-title {
        color: #94a3b8;
        font-size: 0.95rem;
        font-weight: 600;
        letter-spacing: 0.5px;
        text-transform: uppercase;
    }
    .result-price {
        font-size: 2.5rem;
        font-weight: 800;
        color: #38bdf8;
        margin: 0.5rem 0 0.8rem 0;
    }
    .result-terbilang {
        font-size: 1.1rem;
        color: #f1f5f9;
        font-weight: 600;
        background: rgba(255, 255, 255, 0.08);
        padding: 0.4rem 0.8rem;
        border-radius: 8px;
        display: inline-block;
        margin-bottom: 1rem;
    }

    /* Badge */
    .badge {
        display: inline-block;
        padding: 0.25rem 0.6rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        background: rgba(56, 189, 248, 0.2);
        color: #38bdf8;
        border: 1px solid rgba(56, 189, 248, 0.3);
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# --- Helper Functions ---
def format_rupiah_full(angka):
    """Format angka ke Rupiah standar dengan pemisah titik."""
    return f"Rp{angka:,.0f}".replace(",", ".")

def format_rupiah_singkat(angka):
    """Format angka ke format pembacaan singkat (Miliar / Juta)."""
    if angka >= 1_000_000_000:
        return f"{angka / 1_000_000_000:.2f} Miliar"
    elif angka >= 1_000_000:
        return f"{angka / 1_000_000:.1f} Juta"
    else:
        return f"Rp{angka:,.0f}".replace(",", ".")

# --- Model Loading with Fallback / Cache ---
MODEL_PATH = "model_harga_rumah.joblib"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"File model '{MODEL_PATH}' tidak ditemukan di direktori kerja.")
        return None
    try:
        loaded = joblib.load(MODEL_PATH)
        return loaded
    except Exception as err:
        st.error(f"Gagal memuat model: {err}")
        return None

model = load_model()

# --- Sidebar ---
with st.sidebar:
    st.image("https://images.unsplash.com/photo-1560518883-ce09059eeffa?w=600&auto=format&fit=crop&q=80", 
             use_container_width=True, caption="Kota Depok Residential Analytics")
    
    st.markdown("### 📋 Tentang Project")
    st.info(
        "**EarlyBird Bengkod Project**\n\n"
        "Model ini memprediksi estimasi harga penawaran rumah tinggal di wilayah **Kota Depok** "
        "berdasarkan data historis listing properti yang telah melalui pembersihan data, penanganan outlier, "
        "serta tuning algoritma **Random Forest Regressor**."
    )

    st.markdown("---")
    st.markdown("### 💡 Preset Contoh Input")
    st.caption("Pilih salah satu profil rumah untuk mengisi form secara otomatis:")
    
    preset_choice = st.selectbox(
        "Pilih Contoh Spesifikasi",
        ["Default (Notebook)", "Rumah Minimalis Sawangan", "Townhouse Mewah Cinere", "Cluster Keluarga Beji"]
    )

    preset_values = {
        "Default (Notebook)": {"kt": 3, "km": 2, "garasi": 1, "lt": 100.0, "lb": 80.0, "kec": "Cinere"},
        "Rumah Minimalis Sawangan": {"kt": 2, "km": 1, "garasi": 1, "lt": 72.0, "lb": 45.0, "kec": "Sawangan"},
        "Townhouse Mewah Cinere": {"kt": 4, "km": 3, "garasi": 2, "lt": 180.0, "lb": 220.0, "kec": "Cinere"},
        "Cluster Keluarga Beji": {"kt": 3, "km": 2, "garasi": 1, "lt": 90.0, "lb": 70.0, "kec": "Beji"}
    }
    active_preset = preset_values[preset_choice]

    st.markdown("---")
    st.caption("EarlyBird Bengkod © 2026 | Machine Learning Housing Estimator")


# --- Main Header ---
st.markdown("""
<div class="main-header">
    <span class="badge">MACHINE LEARNING WEB APP</span>
    <h1>🏠 Prediksi Harga Rumah Kota Depok</h1>
    <p>Masukkan karakteristik rumah untuk memperoleh estimasi harga penawaran berdasarkan model Machine Learning Random Forest dari project EarlyBird Bengkod.</p>
</div>
""", unsafe_allow_html=True)


# --- Tabs ---
tab_prediksi, tab_analisis, tab_kecamatan = st.tabs([
    "🎯 Kalkulator Prediksi", 
    "📊 Metrik & Insight Model", 
    "📍 Profil Pasar Kecamatan"
])

# ==========================================
# TAB 1: KALKULATOR PREDIKSI (Sesuai Sel Notebook)
# ==========================================
with tab_prediksi:
    st.subheader("Karakteristik Spesifikasi Rumah")
    st.write("Silakan sesuaikan variabel di bawah lalu klik **Prediksi Harga**.")

    with st.form("form_prediksi"):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("##### 🛏️ Ruangan & Fasilitas")
            kamar_tidur = st.number_input(
                "Kamar tidur",
                min_value=0,
                max_value=30,
                value=int(active_preset["kt"]),
                step=1,
                help="Jumlah kamar tidur utama dan anak"
            )

            kamar_mandi = st.number_input(
                "Kamar mandi",
                min_value=0,
                max_value=30,
                value=int(active_preset["km"]),
                step=1,
                help="Jumlah kamar mandi"
            )

            garasi = st.number_input(
                "Jumlah garasi",
                min_value=0,
                max_value=30,
                value=int(active_preset["garasi"]),
                step=1,
                help="Kapasitas garasi / carport kendaraan roda empat"
            )

        with col2:
            st.markdown("##### 📐 Dimensi & Wilayah")
            luas_tanah = st.number_input(
                "Luas tanah (m²)",
                min_value=1.0,
                value=float(active_preset["lt"]),
                step=5.0,
                help="Luas total bidang tanah dalam meter persegi"
            )

            luas_bangunan = st.number_input(
                "Luas bangunan (m²)",
                min_value=1.0,
                value=float(active_preset["lb"]),
                step=5.0,
                help="Luas lantai bangunan dalam meter persegi"
            )

            daftar_kecamatan = [
                "Cinere",
                "Sawangan",
                "Cimanggis",
                "Pancoran Mas",
                "Cilodong",
                "Beji",
                "Sukmajaya",
                "Limo",
                "Tapos",
                "Cipayung",
                "Lainnya"
            ]
            default_kec_index = daftar_kecamatan.index(active_preset["kec"]) if active_preset["kec"] in daftar_kecamatan else 0

            kecamatan = st.selectbox(
                "Kecamatan",
                daftar_kecamatan,
                index=default_kec_index,
                help="Lokasi kecamatan properti di Kota Depok"
            )

        st.markdown("<br>", unsafe_allow_html=True)
        submitted = st.form_submit_button("🔍 Prediksi Harga", use_container_width=True)

    # Output Section
    if submitted:
        if model is None:
            st.error("Model machine learning belum berhasil dimuat. Periksa ketersediaan file 'model_harga_rumah.joblib'.")
        else:
            input_data = pd.DataFrame([{
                "Kamar Tidur_num": kamar_tidur,
                "Kamar Mandi_num": kamar_mandi,
                "Garasi_num": garasi,
                "Luas Tanah_num": luas_tanah,
                "Luas Bangunan_num": luas_bangunan,
                "Kecamatan": kecamatan
            }])

            try:
                prediksi = float(model.predict(input_data)[0])

                if prediksi <= 0:
                    st.warning(
                        "⚠️ Model menghasilkan estimasi yang tidak positif. "
                        "Periksa input dan evaluasi model."
                    )
                else:
                    harga_formatted = format_rupiah_full(prediksi)
                    harga_singkat = format_rupiah_singkat(prediksi)

                    # Menghitung estimasi per meter persegi
                    harga_per_m2_tanah = prediksi / luas_tanah if luas_tanah > 0 else 0
                    harga_per_m2_bangunan = prediksi / luas_bangunan if luas_bangunan > 0 else 0

                    st.markdown(f"""
                    <div class="result-box">
                        <div class="result-title">Estimasi Nilai Pasar Properti (Kota Depok)</div>
                        <div class="result-price">{harga_formatted}</div>
                        <div class="result-terbilang">≈ {harga_singkat}</div>
                        <p style="color: #cbd5e1; font-size: 0.9rem; margin: 0;">
                            Spesifikasi: <strong>{kamar_tidur} KT</strong> | <strong>{kamar_mandi} KM</strong> | <strong>{garasi} Garasi</strong> | 
                            LT: <strong>{luas_tanah:.0f} m²</strong> | LB: <strong>{luas_bangunan:.0f} m²</strong> | Kecamatan: <strong>{kecamatan}</strong>
                        </p>
                    </div>
                    """, unsafe_allow_html=True)

                    st.markdown("<br>", unsafe_allow_html=True)
                    
                    # Quick analytics breakdown
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-label">Estimasi / m² Tanah</div>
                            <div class="metric-value">{format_rupiah_singkat(harga_per_m2_tanah)}/m²</div>
                        </div>
                        """, unsafe_allow_html=True)
                    with c2:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-label">Estimasi / m² Bangunan</div>
                            <div class="metric-value">{format_rupiah_singkat(harga_per_m2_bangunan)}/m²</div>
                        </div>
                        """, unsafe_allow_html=True)
                    with c3:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-label">Kecamatan Terpilih</div>
                            <div class="metric-value" style="font-size: 1.3rem;">{kecamatan}</div>
                        </div>
                        """, unsafe_allow_html=True)

                    st.markdown("<br>", unsafe_allow_html=True)
                    st.caption("ℹ️ *Hasil ini merupakan estimasi model, bukan jaminan harga transaksi.*")

            except Exception as e:
                st.error(f"Prediksi gagal: {e}")


# ==========================================
# TAB 2: METRIK & INSIGHT MODEL (Notebook Context)
# ==========================================
with tab_analisis:
    st.subheader("Evaluasi Performa Model Machine Learning")
    st.write(
        "Berdasarkan pembaruan arsitektur pada notebook **EarlyBird_Bengkod.ipynb**, model terbaik diperoleh dari "
        "algoritma **Random Forest Regressor** dengan **Transformasi Target Logaritmik**, standarisasi fitur, "
        "serta pemfilteran outlier pasar residensial Kota Depok."
    )

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Test MAE", "Rp 419,8 Juta", delta="-1,77 M vs Versi Lama", delta_color="inverse")
    with m2:
        st.metric("Test R² Score", "0.8448", delta="+0.617 vs Versi Lama")
    with m3:
        st.metric("Test MAPE", "20.00%", delta="-3178% vs Versi Lama", delta_color="inverse")
    with m4:
        st.metric("Estimator", "Random Forest + Log Transform")

    st.markdown("---")
    st.subheader("Faktor Paling Berpengaruh (Permutation Feature Importance)")
    st.write(
        "Berdasarkan analisis *Permutation Feature Importance* pada data uji, faktor fisik tanah, bangunan, "
        "dan lokasi kecamatan merupakan penentu utama pembentukan nilai properti di Kota Depok:"
    )

    fi_data = pd.DataFrame({
        "Fitur": [
            "Luas Tanah (m²)",
            "Luas Bangunan (m²)",
            "Kecamatan",
            "Kamar Tidur",
            "Kamar Mandi",
            "Garasi"
        ],
        "Tingkat Pengaruh (Neg-MAE Impact)": [
            5.60e9,
            2.24e9,
            1.21e9,
            2.74e8,
            2.55e8,
            1.20e8
        ]
    })
    st.bar_chart(data=fi_data.set_index("Fitur"), use_container_width=True)

    with st.expander("🔎 Lihat Arsitektur & Hyperparameter Model"):
        st.json({
            "target_transformation": "TransformedTargetRegressor(func=np.log1p, inverse_func=np.expm1)",
            "model__n_estimators": 150,
            "model__max_depth": 25,
            "model__min_samples_split": 2,
            "numeric_preprocessor": "SimpleImputer(strategy='median') + StandardScaler()",
            "categorical_preprocessor": "SimpleImputer(strategy='constant', 'Unknown') + OneHotEncoder(ignore)"
        })


# ==========================================
# TAB 3: PROFIL PASAR KECAMATAN KOTA DEPOK
# ==========================================
with tab_kecamatan:
    st.subheader("Karakteristik & Distribusi Listing di Kota Depok")
    st.write(
        "Kecamatan di Kota Depok memiliki variasi harga median yang sangat beragam karena faktor kedekatan dengan akses "
        "DKI Jakarta (seperti Cinere dan Pangkalan Jati) dibandingkan dengan kawasan perluasan hunian (seperti Sawangan dan Cilodong)."
    )

    kec_stats = pd.DataFrame([
        {"Kecamatan": "Cinere", "Kategori": "Premium / Bordering Jakarta", "Median Estimasi (Rp)": "Rp 3,00 Miliar", "Catatan Pasar": "Kawasan hunian mapan dekat Jakarta Selatan"},
        {"Kecamatan": "Pancoran Mas", "Kategori": "Pusat Kota / Stasiun", "Median Estimasi (Rp)": "Rp 1,45 Miliar", "Catatan Pasar": "Akses transportasi KRL & pusat pemerintahan"},
        {"Kecamatan": "Cimanggis", "Kategori": "Koridor Tol Jagorawi", "Median Estimasi (Rp)": "Rp 1,50 Miliar", "Catatan Pasar": "Akses tol Jakarta-Bogor & industri/komersial"},
        {"Kecamatan": "Sawangan", "Kategori": "Area Pertumbuhan Cepat", "Median Estimasi (Rp)": "Rp 950 Juta", "Catatan Pasar": "Pusat ekspansi cluster perumahan baru"},
        {"Kecamatan": "Beji", "Kategori": "Kawasan Pendidikan (UI)", "Median Estimasi (Rp)": "Rp 1,35 Miliar", "Catatan Pasar": "Pasar sewa kos, mahasiswa, dan hunian keluarga"},
        {"Kecamatan": "Sukmajaya", "Kategori": "Permukiman Padat", "Median Estimasi (Rp)": "Rp 1,10 Miliar", "Catatan Pasar": "Kawasan perumnas lama & perumahan mandiri"},
        {"Kecamatan": "Cilodong", "Kategori": "Residensial Terjangkau", "Median Estimasi (Rp)": "Rp 850 Juta", "Catatan Pasar": "Pilihan favorit first-time home buyers"},
        {"Kecamatan": "Limo", "Kategori": "Perkembangan Tol Desari", "Median Estimasi (Rp)": "Rp 1,75 Miliar", "Catatan Pasar": "Konektivitas meningkat pasca dibukanya Tol Desari"},
        {"Kecamatan": "Tapos", "Kategori": "Timur Depok / Bogor", "Median Estimasi (Rp)": "Rp 900 Juta", "Catatan Pasar": "Banyak pilihan perumahan subsidi & komersial"},
        {"Kecamatan": "Cipayung", "Kategori": "Residensial Berkembang", "Median Estimasi (Rp)": "Rp 800 Juta", "Catatan Pasar": "Kawasan permukiman asri di sekitar stasiun Citayam"}
    ])

    st.dataframe(kec_stats, use_container_width=True, hide_index=True)
