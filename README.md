# Analisis Pengelompokan Tingkat Stabilitas Harga Pangan Nasional Menggunakan K-Means Clustering

**Capstone Project UAS Pembelajaran Mesin — Genap 2025/2026**
Fadhlilah Musa Ulil Albab · A11.2024.15975 · Teknik Informatika, Universitas Dian Nuswantoro

## Ringkasan

Proyek ini membangun pipeline machine learning end-to-end yang mengelompokkan 25 provinsi Indonesia ke dalam segmen stabilitas harga pangan menggunakan **K-Means Clustering** (dibandingkan **DBSCAN** sebagai baseline). Fitur clustering: Indeks Koefisien Variasi (KV) harga produsen tahun 2022, 2023, 2024, dan rata-rata stok Cadangan Pangan Pemerintah Daerah (CPPD) 2025. Hasil clustering diterjemahkan menjadi label segmen dan rekomendasi kebijakan, disajikan dalam dashboard Streamlit interaktif 5 tab.

**Model final: K-Means, K=3** — Silhouette 0.406, Davies-Bouldin 0.764 (`random_state=42`). DBSCAN terbaik mencapai silhouette lebih tinggi (~0.64) namun membuang ~44% provinsi sebagai noise, sehingga tidak layak untuk pemetaan kebijakan yang harus mencakup seluruh provinsi.

## Hasil Cluster

| Cluster | Segmen | Provinsi | Profil (rata-rata) |
|---|---|---|---|
| Hub | **Zona Hub Logistik (Surplus CPPD)** | Jawa Barat (1) | KV 5.94% · CPPD 2.152 Ton (outlier faktual) |
| Stabil | **Zona Mandiri & Stabil** | Aceh, Bali, Banten, Jambi, Jawa Tengah, Kalimantan Barat, Kalimantan Selatan, Kalimantan Tengah, Lampung, NTB, NTT, Riau, Sulawesi Selatan, Sulawesi Tenggara, Sumatera Barat, Sumatera Selatan, Sumatera Utara (17) | KV 5.44% · CPPD 155 Ton |
| Rentan | **Zona Rentan Fluktuatif** | Bengkulu, DI Yogyakarta, Gorontalo, Jawa Timur, Sulawesi Barat, Sulawesi Tengah, Sulawesi Utara (7) | KV 8.68% · CPPD 196 Ton |

**Rekomendasi kebijakan:** zona rentan → operasi pasar darurat + percepatan distribusi logistik + subsidi silang; zona stabil → optimalisasi buffer stock sebagai penyokong; hub logistik → distribusi surplus CPPD ke zona rentan terdekat.

## Struktur Repository

```
capstone-project-data-mining/
├── data/
│   ├── raw/                      # 3 CSV asli (KV, CPPD, sertifikat JKP)
│   └── processed/                # dataset_pangan_terintegrasi.csv, hasil_clustering.csv
├── notebooks/
│   ├── 01_eda.ipynb              # EDA + preprocessing (5 insight tervisualisasi)
│   ├── 02_modeling.ipynb         # grid search K-Means vs DBSCAN + evaluasi
│   └── 03_interpretation.ipynb   # profil cluster + insight kebijakan
├── src/
│   ├── data_preprocessing.py     # load + cleansing + integrasi dataset
│   ├── train_model.py            # grid search + model final + labeling
│   ├── evaluate_model.py         # metrik + 4 visualisasi evaluasi
│   ├── inference.py              # prediksi sampel baru dari model tersimpan
│   └── utils.py                  # path, normalisasi nama, parsing KV
├── models/                       # kmeans_model.pkl, scaler.pkl, meta.pkl
├── app/
│   └── app.py                    # dashboard Streamlit 5 tab
├── reports/
│   ├── figures/                  # 11 PNG (EDA + evaluasi + interpretasi)
│   ├── kmeans_grid.csv           # hasil grid search K-Means (42 kombinasi)
│   └── dbscan_grid.csv           # hasil grid search DBSCAN (24 kombinasi)
├── requirements.txt
└── README.md
```

## Cara Menjalankan

```bash
pip install -r requirements.txt

# 1. Preprocessing -> data/processed/dataset_pangan_terintegrasi.csv
python src/data_preprocessing.py

# 2. Training + tuning -> models/*.pkl, reports/*_grid.csv, hasil_clustering.csv
python src/train_model.py

# 3. Evaluasi + figure -> reports/figures/*.png
python src/evaluate_model.py

# 4. Uji inference 1 sampel dummy
python src/inference.py

# 5. Dashboard (jalankan dari root repo; entry point Streamlit Cloud: app/app.py)
streamlit run app/app.py
```

Notebook dijalankan berurutan (01 → 02 → 03); seluruh output cell sudah tersimpan.

## Dataset

Sumber resmi:

- KV Harga Produsen 2022–2024: https://data.go.id/dataset/dataset/koefisien-variasi-harga-pangan-tingkat-produsen-provinsi-angka-2022-2024
- CPPD Provinsi (update Desember 2025): https://data.badanpangan.go.id/datasetpublications/qh3/cppd-prov-update-desember-2025
- Sertifikat Keamanan Pangan (Health Certificate) 2019–2024: https://data.badanpangan.go.id/datasetpublications/3fc/health-certificat-2019-2024

| File | Isi | Catatan parsing |
|---|---|---|
| `Koefisien_Variasi_Harga_Tingkat_Produsen_Tahun_2022-2024.csv` | 11.232 baris; KV bulanan 15 komoditas, 26 provinsi, 2022–2024 | sep `;`, skiprows=3, nilai `"7,09%"`, ~40% missing |
| `CADANGAN_PANGAN.csv` | 418 baris; CPPD bulanan 2025, 38 provinsi | sep `;`, skiprows=1, desimal koma; CPPD 0.00 faktual |
| `sertifikat_keamanan_pangan.csv` | Sertifikat JKP nasional 2018–2024 | typo tahun `202` dikoreksi → 2020 |

## Limitasi & Catatan Metodologis

1. **Deviasi dari proposal:** fitur Sertifikat JKP dibatalkan karena data hanya tersedia level nasional (tidak bisa per provinsi); dipakai sebagai konteks EDA saja.
2. **Hipotesis proposal tidak terbukti:** korelasi CPPD vs KV ≈ −0.04 — besaran stok cadangan tidak berasosiasi dengan stabilitas harga; temuan valid yang dibahas di laporan.
3. **Silhouette 0.41 < target 0.5:** wajar untuk data sosial-ekonomi dengan batas antar kelompok yang tidak tegas; trade-off vs DBSCAN dibahas di notebook 02.
4. **Cakupan 25 provinsi** karena data KV produsen hanya mencakup 26 provinsi (1 drop karena KV kosong seluruhnya).
5. **Tanpa train/test split:** clustering unsupervised untuk pemetaan populasi penuh; evaluasi memakai metrik internal.
6. Interpretasi fitur memakai **profil centroid** (SHAP/LIME tidak berlaku untuk unsupervised learning).
