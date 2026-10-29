# 📚 PANDUAN LENGKAP & BEDAH KODE PER SEL: `EarlyBird_Bengkod.ipynb`
### Proyek Machine Learning Prediksi Harga Rumah Kota Depok (EarlyBird Bengkod)
---

Dokumen ini disusun sebagai panduan belajar komprehensif bagi Anda untuk memahami **seluruh alur, konsep data science, fungsi, dan setiap baris kode** di dalam notebook `EarlyBird_Bengkod.ipynb`.

---

## 📑 DAFTAR ISI ALUR 9 BAGIAN
1. [Sel 0 [Markdown]: Business Understanding & Problem Framing](#sel-0-markdown-business-understanding--problem-framing)
2. [Sel 1 [Markdown] & Sel 2 [Code]: Import Library & Konfigurasi Lingkungan](#sel-1-markdown--sel-2-code-import-library--konfigurasi-lingkungan)
3. [Sel 3 [Code]: Memuat Dataset Mentah (Raw Data Loading)](#sel-3-code-memuat-dataset-mentah-raw-data-loading)
4. [Sel 4 [Markdown] & Sel 5 [Code]: Audit Missing Values, Suspects, dan Duplikat](#sel-4-markdown--sel-5-code-audit-missing-values-suspects-dan-duplikat)
5. [Sel 6 [Markdown] & Sel 7 [Code]: Parsing Nominal Harga Awal](#sel-6-markdown--sel-7-code-parsing-nominal-harga-awal)
6. [Sel 8 [Markdown] & Sel 9 [Code]: Visualisasi Distribusi Target (Normal vs Log10)](#sel-8-markdown--sel-9-code-visualisasi-distribusi-target-normal-vs-log10)
7. [Sel 10 [Markdown] & Sel 11 [Code]: Analisis Statistik Pasar per Kecamatan](#sel-10-markdown--sel-11-code-analisis-statistik-pasar-per-kecamatan)
8. [Sel 12 [Markdown] & Sel 13 [Code]: Regex Data Recovery (Penyelamatan Kolom Tertukar)](#sel-12-markdown--sel-13-code-regex-data-recovery-penyelamatan-kolom-tertukar)
9. [Sel 14 [Markdown] & Sel 15 [Code]: Eliminasi Outlier Ekstrem (Kunci Peningkatan Performa)](#sel-14-markdown--sel-15-code-eliminasi-outlier-ekstrem-kunci-peningkatan-performa)
10. [Sel 16 [Markdown] & Sel 17 [Code]: Pemisahan Data (Train-Test Split 80:20)](#sel-16-markdown--sel-17-code-pemisahan-data-train-test-split-8020)
11. [Sel 18 [Markdown] & Sel 19 [Code]: Arsitektur Preprocessing Pipeline (ColumnTransformer)](#sel-18-markdown--sel-19-code-arsitektur-preprocessing-pipeline-columntransformer)
12. [Sel 20 [Markdown] & Sel 21 [Code]: Model Benchmarking 5 Algoritma Regresi](#sel-20-markdown--sel-21-code-model-benchmarking-5-algoritma-regresi)
13. [Sel 22 [Markdown] & Sel 23 [Code]: Hyperparameter Tuning (RandomizedSearchCV)](#sel-22-markdown--sel-23-code-hyperparameter-tuning-randomizedsearchcv)
14. [Sel 24 [Markdown] & Sel 25 [Code]: Permutation Feature Importance](#sel-24-markdown--sel-25-code-permutation-feature-importance)
15. [Sel 26 [Markdown] & Sel 27 [Code]: Evaluasi Model Akhir & Analisis Diagnostik Residual](#sel-26-markdown--sel-27-code-evaluasi-model-akhir--analisis-diagnostik-residual)
16. [Sel 28 [Markdown], Sel 29 [Code], & Sel 30 [Markdown]: Model Persistence & Deployment](#sel-28-markdown-sel-29-code--sel-30-markdown-model-persistence--deployment)

---

## Sel 0 [Markdown]: Business Understanding & Problem Framing
- **Tujuan**: Merumuskan konteks bisnis, latar belakang masalah, dan sasaran proyek.
- **Konsep**: Proyek data science yang baik harus selalu dimulai dari pemahaman domain (*domain knowledge*). Kota Depok adalah kawasan penyangga Jakarta dengan disparitas harga properti yang lebar. Tujuannya adalah menciptakan model estimasi harga wajar (*fair valuation*) untuk membantu calon pembeli dan investor menghindari properti yang *overpriced*.

---

## Sel 1 [Markdown] & Sel 2 [Code]: Import Library & Konfigurasi Lingkungan

### Kode Sel 2:
```python
import re
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import kruskal
from sklearn.model_selection import train_test_split, KFold, RandomizedSearchCV
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.inspection import permutation_importance
import joblib

warnings.filterwarnings("ignore")
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
pd.set_option("display.max_columns", 50)
pd.set_option("display.float_format", lambda x: "%.2f" % x)

print("Semua pustaka berhasil dimuat.")
```

### Bedah Kode Baris per Baris:
1. `import re`: Modul *Regular Expressions* bawaan Python untuk pencarian pola teks kompleks (digunakan saat mem-parsing angka, satuan "Miliar/Juta", dan label "LT/LB").
2. `import warnings; warnings.filterwarnings("ignore")`: Menonaktifkan peringatan minor (*warning*) dari library agar output notebook rapi dan mudah dibaca.
3. `import numpy as np`: Library komputasi numerik dan manipulasi array (digunakan untuk fungsi matematis seperti `np.log1p`, `np.expm1`, dan pembuatan mask boolean).
4. `import pandas as pd`: Library pengolahan data tabular utama (DataFrame dan Series).
5. `import matplotlib.pyplot as plt` & `import seaborn as sns`: Library untuk pembuatan visualisasi data (histogram, scatter plot, diagram batang, heatmap).
6. `from sklearn.model_selection import ...`:
   - `train_test_split`: Membagi data menjadi data latih (*training set*) dan data uji (*testing set*).
   - `KFold`: Melakukan validasi silang (*cross-validation*) $K$-lipat secara acak.
   - `RandomizedSearchCV`: Mencari kombinasi hyperparameter terbaik secara acak dan efisien.
7. `from sklearn.compose import ColumnTransformer, TransformedTargetRegressor`:
   - `ColumnTransformer`: Mengaplikasikan transformasi preprocessing yang berbeda untuk kolom numerik dan kolom kategorikal secara simultan.
   - `TransformedTargetRegressor`: **Kunci penting!** Membungkus model agar target harga dilatih dalam skala logaritmik $\log(1+y)$ secara otomatis, namun saat memprediksi otomatis dikembalikan ke satuan Rupiah via $\exp(y)-1$.
8. `from sklearn.pipeline import Pipeline`: Menggabungkan langkah *imputer*, *scaler*, dan *estimator* menjadi satu kesatuan alur kerja tertutup untuk mencegah kebocoran data (*data leakage*).
9. `from sklearn.preprocessing import OneHotEncoder, StandardScaler`:
   - `OneHotEncoder`: Mengubah kolom teks kategorikal (`Kecamatan`) menjadi representasi kolom biner (0 atau 1).
   - `StandardScaler`: Mengubah skala fitur numerik agar berpusat pada rata-rata 0 dengan standar deviasi 1 ($z = \frac{x - \mu}{\sigma}$).
10. `from sklearn.impute import SimpleImputer`: Mengisi nilai yang kosong (*missing values*) dengan strategi tertentu (median untuk numerik, konstanta untuk kategori).
11. `from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score`: Tiga metrik evaluasi regresi utama:
    - **MAE**: Rata-rata selisih absolut antara harga aktual dan prediksi dalam Rupiah.
    - **MAPE**: Rata-rata persentase kesalahan relatif.
    - **$R^2$**: Seberapa besar variansi target berhasil dijelaskan oleh fitur-fitur model (nilai 1.0 = sempurna).
12. `import joblib`: Library untuk menyimpan (*serialize*) dan memuat (*deserialize*) objek model machine learning ke/dari berkas disk (`.joblib`).
13. `pd.set_option(...)`: Mengatur format tampilan tabel Pandas agar menampilkan maksimal 50 kolom dan angka desimal dibatasi 2 angka di belakang koma.

---

## Sel 3 [Code]: Memuat Dataset Mentah (Raw Data Loading)

### Kode:
```python
df_raw = pd.read_csv("Raw-Final-Rumah.csv")

print("Dimensi dataset:", df_raw.shape)
display(df_raw.head())
df_raw.info()
```

### Bedah Kode Baris per Baris:
1. `df_raw = pd.read_csv("Raw-Final-Rumah.csv")`: Membaca file CSV data mentah properti dan menyimpannya ke dalam variabel DataFrame `df_raw`.
2. `print("Dimensi dataset:", df_raw.shape)`: Mencetak ukuran matriks data `(baris, kolom)`. Dataset ini memiliki **40.200 baris dan 8 kolom**.
3. `display(df_raw.head())`: Menampilkan 5 baris pertama data untuk inspeksi visual awal kolom `Harga`, `Kamar Tidur`, `Kamar Mandi`, `Garasi`, `Luas Tanah`, `Luas Bangunan`, `Kecamatan`, dan `Page`.
4. `df_raw.info()`: Memeriksa tipe data teknis setiap kolom. Sebagian besar kolom bertipe `object` (string) karena berisi teks satuan seperti `"2,5 Miliar"` atau `"LB : 110 m²"`.

---

## Sel 4 [Markdown] & Sel 5 [Code]: Audit Missing Values, Suspects, dan Duplikat

### Kode Sel 5:
```python
print("Jumlah Missing Values per Kolom:")
display(df_raw.isnull().sum().to_frame("Jumlah Missing"))

suspects = ["?", "??", "NA", "N/A", "na", "None", "null", "Null", "Unknown", "UNK", "-", "--", " ", "Nan", "XXX"]
suspect_summary = {val: (df_raw == val).sum().sum() for val in suspects}
print("\nRingkasan Nilai Suspect:", {k: v for k, v in suspect_summary.items() if v > 0})

print("\nJumlah Duplikat Persis:", df_raw.duplicated().sum())
```

### Bedah Kode Baris per Baris:
1. `df_raw.isnull().sum()`: Menghitung total data kosong standar (`NaN`/`None`) pada setiap kolom.
2. `suspects = [...]`: Daftar karakter yang sering digunakan pengguna atau program scraping sebagai penanda data tidak diketahui.
3. `suspect_summary = {val: (df_raw == val).sum().sum() for val in suspects}`: Memeriksa apakah ada string suspect tersebut di dalam tabel.
4. `df_raw.duplicated().sum()`: Menghitung baris yang terduplikasi secara persis di seluruh kolom.

---

## Sel 6 [Markdown] & Sel 7 [Code]: Parsing Nominal Harga Awal

### Kode Sel 7:
```python
def parse_harga(value):
    if pd.isna(value):
        return np.nan
    text = str(value).strip().lower().replace("\u00a0", " ")
    match = re.search(r"(\d[\d.,]*)\s*(miliar|milyar|juta|jt)\b", text)
    if not match:
        return np.nan
    angka_text, satuan = match.group(1), match.group(2)
    if "," in angka_text:
        angka_text = angka_text.replace(".", "").replace(",", ".")
    elif angka_text.count(".") > 1:
        angka_text = angka_text.replace(".", "")
    angka = float(angka_text)
    if satuan in ["miliar", "milyar"]:
        return angka * 1_000_000_000
    return angka * 1_000_000

df_eda = df_raw.copy()
df_eda["Harga_Rp"] = df_eda["Harga"].apply(parse_harga)

print("Ringkasan Statistik Harga Kasar (Sebelum Pembersihan):")
display(df_eda["Harga_Rp"].describe().to_frame())
```

### Bedah Kode Baris per Baris:
1. `def parse_harga(value)`: Fungsi konversi string harga penawaran menjadi angka floating point Rupiah.
2. `text = str(value).strip().lower().replace("\u00a0", " ")`: Membersihkan spasi di awal/akhir, mengubah ke huruf kecil, serta mengganti karakter non-breaking space Unicode (`\u00a0`) menjadi spasi biasa.
3. `re.search(r"(\d[\d.,]*)\s*(miliar|milyar|juta|jt)\b", text)`:
   - Pola Regex untuk menangkap angka (misal `2,5` atau `850`) di grup 1.
   - Menangkap satuan (`miliar`/`juta`) di grup 2.
4. Normalisasi tanda baca: Mengubah format penulisan desimal Indonesia (koma `,` menjadi titik `.`).
5. `if satuan in ["miliar", "milyar"]: return angka * 1_000_000_000`: Mengalikan angka dengan 1 Miliar ($10^9$) jika satuannya miliar, atau 1 Juta ($10^6$) jika satuannya juta.
6. `display(df_eda["Harga_Rp"].describe())`: Menampilkan statistika dasar. Di sini terlihat anomali data mentah: nilai minimum **Rp 1.030.000** (harga sewa) dan nilai maksimum **Rp 999.000.000.000** (outlier ekstrem).

---

## Sel 8 [Markdown] & Sel 9 [Code]: Visualisasi Distribusi Target (Normal vs Log10)

### Kode Sel 9:
```python
harga_valid = df_eda["Harga_Rp"].dropna()
harga_positif = harga_valid[harga_valid > 0]

fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Plot distribusi asli
sns.histplot(harga_positif, bins=40, kde=True, ax=axes[0], color="#2563eb")
axes[0].set_title("Distribusi Harga Asli (Sangat Skewed)", fontsize=13, fontweight="bold")
axes[0].set_xlabel("Harga (Rp)")

# Plot distribusi log10
sns.histplot(np.log10(harga_positif), bins=40, kde=True, ax=axes[1], color="#059669")
axes[1].set_title("Distribusi Log10 Harga (Mendekati Normal/Gaussian)", fontsize=13, fontweight="bold")
axes[1].set_xlabel("Log10 Harga (Rp)")

plt.tight_layout()
plt.show()
```

### Bedah Kode Baris per Baris:
1. `fig, axes = plt.subplots(1, 2, figsize=(15, 5))`: Membuat kanvas visualisasi dengan 1 baris dan 2 kolom grafik berdampingan.
2. `sns.histplot(harga_positif, ...)`: Menampilkan histogram harga nominal mentah. Grafiknya terjepit di sisi kiri (*heavy right-skewed*) karena ditarik oleh beberapa rumah mewah ratusan miliar.
3. `sns.histplot(np.log10(harga_positif), ...)`: Mengaplikasikan transformasi logaritmik $\log_{10}$. Hasilnya menunjukkan kurva berbentuk lonceng (*Gaussian Bell Curve*) yang simetris. Ini membuktikan secara empiris bahwa pemodelan regresi **wajib menggunakan transformasi log** agar algoritma bekerja optimal.

---

## Sel 10 [Markdown] & Sel 11 [Code]: Analisis Statistik Pasar per Kecamatan

### Kode Sel 11:
```python
ringkasan_kecamatan = (
    df_eda.dropna(subset=["Harga_Rp"])
    .groupby("Kecamatan")["Harga_Rp"]
    .agg(Jumlah_Listing="count", Median="median", Rata_Rata="mean", Min="min", Max="max")
    .sort_values("Median", ascending=False)
)
display(ringkasan_kecamatan)

plt.figure(figsize=(10, 6))
ringkasan_kecamatan["Median"].sort_values().plot(kind="barh", color="#3b82f6")
plt.title("Median Harga Properti Berdasarkan Kecamatan (Rp)", fontsize=14, fontweight="bold")
plt.xlabel("Median Harga (Rp)")
plt.ylabel("Kecamatan")
plt.tight_layout()
plt.show()
```

### Bedah Kode Baris per Baris:
1. `.groupby("Kecamatan")["Harga_Rp"].agg(...)`: Mengelompokkan data berdasarkan kecamatan dan menghitung 5 statistik deskriptif sekaligus (Jumlah, Median, Rata-rata, Min, Max).
2. `ringkasan_kecamatan["Median"].sort_values().plot(kind="barh")`: Membuat diagram batang horizontal terurut. Dari sini kita melihat kecamatan berbatasan Jakarta seperti **Cinere** memiliki median harga tertinggi (~Rp 3 Miliar), sedangkan kawasan ekspansi baru seperti **Sawangan** dan **Cilodong** berada di kisaran Rp 850 Juta – Rp 950 Juta.

---

## Sel 12 [Markdown] & Sel 13 [Code]: Regex Data Recovery (Penyelamatan Kolom Tertukar)

### Kode Sel 13:
```python
def parse_angka(value):
    if pd.isna(value): return np.nan
    text = str(value).strip()
    if re.fullmatch(r"\d+(?:[.,]\d+)?", text):
        return float(text.replace(",", "."))
    return np.nan

def parse_area_label(value):
    if pd.isna(value): return None, np.nan
    text = str(value).strip().lower()
    match = re.search(r"\b(lt|lb)\b\s*:?\s*(\d+(?:[.,]\d+)?)\s*m(?:²|2)?\b", text)
    if not match: return None, np.nan
    return match.group(1).upper(), float(match.group(2).replace(",", "."))

def repair_area_row(row):
    sumber = ["Kamar Tidur", "Kamar Mandi", "Garasi", "Luas Tanah", "Luas Bangunan"]
    kandidat_lt, kandidat_lb = [], []
    for col in sumber:
        label, nilai = parse_area_label(row[col])
        if label == "LT": kandidat_lt.append(nilai)
        elif label == "LB": kandidat_lb.append(nilai)
    
    def pilih_nilai(kandidat):
        nilai_unik = sorted(set(kandidat))
        if len(nilai_unik) == 1: return nilai_unik[0], False
        if len(nilai_unik) == 0: return np.nan, False
        return np.nan, True
        
    lt, _ = pilih_nilai(kandidat_lt)
    lb, _ = pilih_nilai(kandidat_lb)
    return pd.Series({"LT_recovered": lt, "LB_recovered": lb})

hasil_repair = df_raw.apply(repair_area_row, axis=1)

df_clean = df_raw.copy()
df_clean["Harga_Rp"] = df_clean["Harga"].apply(parse_harga)
for col in ["Kamar Tidur", "Kamar Mandi", "Garasi"]:
    df_clean[col + "_num"] = df_clean[col].apply(parse_angka)

df_clean["Luas Tanah_num"] = hasil_repair["LT_recovered"]
df_clean["Luas Bangunan_num"] = hasil_repair["LB_recovered"]
```

### Bedah Kode Baris per Baris:
1. `parse_angka`: Mengambil nilai numerik murni dari kolom kamar dan garasi (mengabaikan teks campuran).
2. `parse_area_label`: Menggunakan regex `r"\b(lt|lb)\b\s*:?\s*(\d+(?:[.,]\d+)?)\s*m(?:²|2)?\b"` untuk mencari apakah ada teks berlabel `LT` (Luas Tanah) atau `LB` (Luas Bangunan) lengkap dengan angka luasannya.
3. `repair_area_row`: Fungsi canggih untuk memindai ke-5 kolom properti. Jika ada luasan tanah yang tidak sengaja tertulis di kolom `Kamar Tidur`, fungsi ini mengekstrak nilai tersebut dan mengumpulkannya ke `kandidat_lt`.
4. `pilih_nilai`: Memverifikasi integritas nilai. Jika ada nilai tunggal yang valid, nilai tersebut ditetapkan sebagai nilai definitif. Jika terdapat konflik antar kolom, ditetapkan `NaN` agar tidak menebak data yang salah.
5. Hasil: Ribuan data luas tanah dan luas bangunan yang sebelumnya hilang (*missing*) berhasil diselamatkan kembali!

---

## Sel 14 [Markdown] & Sel 15 [Code]: Eliminasi Outlier Ekstrem (Kunci Peningkatan Performa)

### Kode Sel 15:
```python
# Filter Domain Properti Wajar
mask_harga = (df_clean["Harga_Rp"] >= 150_000_000) & (df_clean["Harga_Rp"] <= 20_000_000_000)
mask_lt = df_clean["Luas Tanah_num"].isna() | ((df_clean["Luas Tanah_num"] >= 25) & (df_clean["Luas Tanah_num"] <= 1200))
mask_lb = df_clean["Luas Bangunan_num"].isna() | ((df_clean["Luas Bangunan_num"] >= 20) & (df_clean["Luas Bangunan_num"] <= 800))
mask_kt = df_clean["Kamar Tidur_num"].isna() | ((df_clean["Kamar Tidur_num"] >= 1) & (df_clean["Kamar Tidur_num"] <= 12))
mask_km = df_clean["Kamar Mandi_num"].isna() | ((df_clean["Kamar Mandi_num"] >= 1) & (df_clean["Kamar Mandi_num"] <= 10))

df_filtered = df_clean[mask_harga & mask_lt & mask_lb & mask_kt & mask_km].drop_duplicates().copy()

print(f"Data awal sebelum filtering : {df_raw.shape[0]} baris")
print(f"Data bersih setelah filtering: {df_filtered.shape[0]} baris")
print(f"Total baris outlier/duplikat dibuang: {df_raw.shape[0] - df_filtered.shape[0]} baris")
```

### Bedah Kode Baris per Baris:
- **Mengapa ini adalah kunci perbaikan model?**
  Pada versi lama, data listing sewa Rp 1 Juta dan luas tanah typo 818 juta $m^2$ dibiarkan masuk ke model, menyebabkan MAPE meledak 3.198%.
- `mask_harga`: Menetapkan batas bawah harga wajar rumah jual beli di Depok minimal **Rp 150 Juta** dan maksimal **Rp 20 Miliar**.
- `mask_lt` & `mask_lb`: Membatasi luas tanah antara 25 s.d. 1.200 $m^2$ dan luas bangunan 20 s.d. 800 $m^2$.
- `mask_kt` & `mask_km`: Membatasi jumlah kamar tidur (1–12) dan kamar mandi (1–10).
- `.drop_duplicates()`: Membuang duplikasi listing hasil scraping ulang.
- Hasil: Dataset menyaring 34.782 baris data bersih yang representatif untuk pasar hunian residensial.

---

## Sel 16 [Markdown] & Sel 17 [Code]: Pemisahan Data (Train-Test Split 80:20)

### Kode Sel 17:
```python
fitur = [
    "Kamar Tidur_num",
    "Kamar Mandi_num",
    "Garasi_num",
    "Luas Tanah_num",
    "Luas Bangunan_num",
    "Kecamatan"
]

X = df_filtered[fitur].copy()
y = df_filtered["Harga_Rp"].copy()

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42
)
```

### Bedah Kode Baris per Baris:
1. `fitur = [...]`: Memilih 6 variabel prediktor utama (5 fitur numerik fisik dan 1 fitur kategorikal lokasi).
2. `X` dan `y`: Memisahkan matriks fitur independen ($X$) dan vektor target dependen ($y$).
3. `train_test_split(..., test_size=0.20, random_state=42)`: Membagi data secara tegas:
   - **80% data training** ($27.825$ baris) untuk melatih model.
   - **20% data testing** ($6.957$ baris) sebagai data ujian murni yang tidak pernah dilihat model saat proses belajar.
   - `random_state=42`: Menjamin replikasi hasil yang konsisten setiap kali kode dijalankan.

---

## Sel 18 [Markdown] & Sel 19 [Code]: Arsitektur Preprocessing Pipeline (ColumnTransformer)

### Kode Sel 19:
```python
kolom_numerik = ["Kamar Tidur_num", "Kamar Mandi_num", "Garasi_num", "Luas Tanah_num", "Luas Bangunan_num"]
kolom_kategori = ["Kecamatan"]

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]),
            kolom_numerik
        ),
        (
            "cat",
            Pipeline([
                ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
                ("onehot", OneHotEncoder(handle_unknown="ignore"))
            ]),
            kolom_kategori
        )
    ]
)
```

### Bedah Kode Baris per Baris:
1. `transformers=[...]`:
   - Cabang `"num"`:
     - `SimpleImputer(strategy="median")`: Mengisi missing value fitur numerik dengan nilai median (kebal outlier dibandingkan rata-rata).
     - `StandardScaler()`: Menstandarkan skala fitur numerik agar model KNN dan Regresi Linier tidak bias terhadap fitur berangka besar seperti luas tanah.
   - Cabang `"cat"`:
     - `SimpleImputer(strategy="constant", fill_value="Unknown")`: Mengisi missing value kecamatan dengan kategori "Unknown".
     - `OneHotEncoder(handle_unknown="ignore")`: Mengubah nama-nama kecamatan menjadi kolom biner dummy. `handle_unknown="ignore"` memastikan jika kelak di aplikasi web ada nama kecamatan baru yang belum pernah dilatih, sistem tidak akan error (*crash*).

---

## Sel 20 [Markdown] & Sel 21 [Code]: Model Benchmarking 5 Algoritma Regresi

### Kode Sel 21:
```python
def evaluasi_model(y_true, y_pred):
    mask = np.asarray(y_true) > 0
    mape = mean_absolute_percentage_error(np.asarray(y_true)[mask], np.asarray(y_pred)[mask]) * 100
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {"MAE (Rp)": mae, "MAPE (%)": mape, "R2": r2}

candidate_models = {
    "Ridge Regression": Ridge(alpha=1.0),
    "K-Nearest Neighbors": KNeighborsRegressor(n_neighbors=7),
    "Decision Tree": DecisionTreeRegressor(max_depth=12, random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=20, random_state=42, n_jobs=-1),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42)
}

hasil_benchmark = []
fitted_pipelines = {}

for nama, model in candidate_models.items():
    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", TransformedTargetRegressor(
            regressor=model,
            func=np.log1p,
            inverse_func=np.expm1
        ))
    ])
    
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    
    metrik = evaluasi_model(y_test, y_pred)
    metrik["Model"] = nama
    hasil_benchmark.append(metrik)
    fitted_pipelines[nama] = pipe

df_benchmark = pd.DataFrame(hasil_benchmark)[["Model", "MAE (Rp)", "MAPE (%)", "R2"]].sort_values("R2", ascending=False)
display(df_benchmark)
```

### Bedah Kode Baris per Baris:
1. `TransformedTargetRegressor(regressor=model, func=np.log1p, inverse_func=np.expm1)`:
   - Saat `fit()`, target harga diubah menjadi $\ln(1 + \text{Harga})$.
   - Model mempelajari pola logaritmik harga.
   - Saat `predict()`, hasil logaritmik otomatis dikonversi balik dengan $e^{\hat{y}} - 1$.
2. Pelatihan serentak 5 kandidat model membuktikan bahwa **Random Forest Regressor** adalah pemenang mutlak:
   - Nilai $R^2$ mencapai **0.8409** (mampu menjelaskan 84,1% variansi).
   - MAE berada di kisaran Rp 431 Juta dan MAPE sekitar 20.5%.

---

## Sel 22 [Markdown] & Sel 23 [Code]: Hyperparameter Tuning (RandomizedSearchCV)

### Kode Sel 23:
```python
param_dist = {
    "regressor__regressor__n_estimators": [100, 150, 200],
    "regressor__regressor__max_depth": [15, 20, 25, 30],
    "regressor__regressor__min_samples_split": [2, 5],
    "regressor__regressor__min_samples_leaf": [1, 2]
}

base_rf_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", TransformedTargetRegressor(
        regressor=RandomForestRegressor(random_state=42, n_jobs=-1),
        func=np.log1p,
        inverse_func=np.expm1
    ))
])

cv = KFold(n_splits=5, shuffle=True, random_state=42)

search = RandomizedSearchCV(
    estimator=base_rf_pipeline,
    param_distributions=param_dist,
    n_iter=8,
    scoring="neg_mean_absolute_error",
    cv=cv,
    random_state=42,
    n_jobs=-1,
    verbose=1,
    refit=True
)

search.fit(X_train, y_train)
```

### Bedah Kode Baris per Baris:
1. `param_dist`: Mendefinisikan ruang pencarian kombinasi hyperparameter untuk Random Forest:
   - `n_estimators`: Jumlah pohon keputusan dalam ensemble (100, 150, 200 pohon).
   - `max_depth`: Kedalaman maksimum percabangan tiap pohon.
   - `min_samples_split` & `min_samples_leaf`: Syarat jumlah sampel minimum untuk mencegah pohon menghafal data (*overfitting*).
2. `KFold(n_splits=5, shuffle=True)`: Membagi data latih menjadi 5 lipatan (*folds*) secara acak agar setiap kombinasi hyperparameter diuji pada 5 subset data berbeda.
3. `RandomizedSearchCV(..., scoring="neg_mean_absolute_error", refit=True)`: Menjalankan pencarian acak 8 iterasi terbaik berdasarkan skor MAE terendah, dan otomatis melatih ulang model terbaik pada seluruh data training (`refit=True`).

---

## Sel 24 [Markdown] & Sel 25 [Code]: Permutation Feature Importance

### Kode Sel 25:
```python
model_final = search.best_estimator_

perm = permutation_importance(
    model_final,
    X_test,
    y_test,
    n_repeats=5,
    random_state=42,
    scoring="neg_mean_absolute_error",
    n_jobs=-1
)

df_importance = pd.DataFrame({
    "Fitur": X_test.columns,
    "Importance": perm.importances_mean,
    "Std": perm.importances_std
}).sort_values("Importance", ascending=False)

display(df_importance)

plt.figure(figsize=(9, 5))
plt.barh(df_importance["Fitur"][::-1], df_importance["Importance"][::-1], color="#2563eb")
plt.xlabel("Kenaikan Nilai Neg-MAE (Dampak terhadap Penurunan Error)")
plt.title("Permutation Feature Importance (Data Testing)", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.show()
```

### Bedah Kode Baris per Baris:
1. `permutation_importance`: Metode evaluasi tingkat kontribusi fitur yang paling objektif dan tidak bias. Caranya: nilai pada satu kolom diacak (*shuffled*), lalu dihitung seberapa anjlok akurasi model. Semakin drastis error meningkat ketika suatu fitur diacak, berarti fitur tersebut semakin penting.
2. Hasil grafik bar mendemonstrasikan bahwa:
   - **Luas Tanah ($m^2$)** menduduki peringkat #1 paling krusial.
   - **Luas Bangunan ($m^2$)** menduduki peringkat #2.
   - **Kecamatan** menduduki peringkat #3.
   - Jumlah Kamar Tidur, Kamar Mandi, dan Garasi memberikan pengaruh tambahan.

---

## Sel 26 [Markdown] & Sel 27 [Code]: Evaluasi Model Akhir & Analisis Diagnostik Residual

### Kode Sel 27:
```python
pred_final = model_final.predict(X_test)
hasil_final = evaluasi_model(y_test, pred_final)

print("=== HASIL EVALUASI MODEL AKHIR (TEST SET) ===")
print(f"MAE  : Rp {hasil_final['MAE (Rp)']:,.0f}")
print(f"MAPE : {hasil_final['MAPE (%)']:.2f}%")
print(f"R²   : {hasil_final['R2']:.4f}")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# 1. Scatter Plot Aktual vs Prediksi
axes[0].scatter(y_test / 1e9, pred_final / 1e9, alpha=0.3, color="#2563eb")
batas_min = min(y_test.min(), pred_final.min()) / 1e9
batas_max = max(y_test.max(), pred_final.max()) / 1e9
axes[0].plot([batas_min, batas_max], [batas_min, batas_max], "r--", linewidth=2)
axes[0].set_xlabel("Harga Aktual (Miliar Rp)")
axes[0].set_ylabel("Harga Prediksi (Miliar Rp)")
axes[0].set_title("Harga Aktual vs Prediksi Model (Garis Ideal Merah)")

# 2. Residual Plot
residuals = (y_test - pred_final) / 1e9
axes[1].scatter(pred_final / 1e9, residuals, alpha=0.3, color="#059669")
axes[1].axhline(0, color="r", linestyle="--", linewidth=2)
axes[1].set_xlabel("Harga Prediksi (Miliar Rp)")
axes[1].set_ylabel("Residual (Aktual - Prediksi) (Miliar Rp)")
axes[1].set_title("Plot Residual vs Harga Prediksi")

plt.tight_layout()
plt.show()
```

### Bedah Kode Baris per Baris:
1. `hasil_final`: Menghitung skor final pada data testing murni:
   - **$R^2 = 0.8448$**: Model sangat handal (84,5% akurasi penjelasan variansi).
   - **MAE = Rp 419 Juta**: Error rata-rata turun drastis dari sebelumnya Rp 2,19 Miliar.
   - **MAPE = 20.00%**: Sesuai dengan standar penilaian industri properti.
2. Grafik Kiri (*Scatter Plot Aktual vs Prediksi*): Titik-titik biru menyebar merapat di sekitar garis putus-putus merah $y = x$. Ini menunjukkan prediksi model selaras kuat dengan harga pasar aktual.
3. Grafik Kanan (*Residual Plot*): Menampilkan selisih $y_{\text{aktual}} - \hat{y}_{\text{prediksi}}$. Sebaran titik terpusat simetris di sekitar garis horizontal 0, membuktikan model tidak mengalami bias sistematis (*homoscedasticity terpenuhi*).

---

## Sel 28 [Markdown], Sel 29 [Code], & Sel 30 [Markdown]: Model Persistence & Deployment

### Kode Sel 29:
```python
# Menyimpan model ke disk
model_path = "model_harga_rumah.joblib"
joblib.dump(model_final, model_path)
print(f"Model berhasil disimpan ke berkas: {model_path}")

# Verifikasi simulasi pemuatan model seperti di Streamlit
model_loaded = joblib.load(model_path)

sample_input = pd.DataFrame([{
    "Kamar Tidur_num": 3,
    "Kamar Mandi_num": 2,
    "Garasi_num": 1,
    "Luas Tanah_num": 100.0,
    "Luas Bangunan_num": 80.0,
    "Kecamatan": "Cinere"
}])

estimasi_sample = float(model_loaded.predict(sample_input)[0])
print(f"Contoh Prediksi Properti Cinere (LT 100m², LB 80m², 3KT, 2KM): Rp {estimasi_sample:,.0f}")
```

### Bedah Kode Baris per Baris:
1. `joblib.dump(model_final, model_path)`: Menyimpan pipeline lengkap (termasuk preprocessor, target transformator, dan estimator Random Forest) ke berkas biner disk `model_harga_rumah.joblib`.
2. `model_loaded = joblib.load(model_path)`: Memuat kembali berkas model untuk memverifikasi proses deserialisasi berjalan lancar.
3. `sample_input = pd.DataFrame([{...}])`: Membuat contoh DataFrame input 1 baris yang merepresentasikan karakteristik rumah tinggal di Cinere.
4. `model_loaded.predict(sample_input)`: Menghasilkan estimasi harga pasar wajar (~Rp 1,15 Miliar) secara instan, membuktikan pipeline siap dipakai secara langsung oleh aplikasi web Streamlit (`app.py`).
