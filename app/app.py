"""Dashboard Streamlit — Segmentasi Stabilitas Harga Pangan Nasional (K-Means).

Jalankan dari root repo: streamlit run app/app.py
"""
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
FEATURES_FALLBACK = ["KV_2022", "KV_2023", "KV_2024", "CPPD_Ton"]

SEG_COLORS = {
    "Zona Mandiri & Stabil": "#2e8b57",
    "Zona Transisi / Waspada": "#e67e22",
    "Zona Rentan Fluktuatif": "#cd5c5c",
    "Zona Hub Logistik (Surplus CPPD)": "#daa520",
}
REKOMENDASI = {
    "Zona Rentan Fluktuatif": (
        "Operasi pasar darurat, percepatan distribusi logistik dari pusat, "
        "dan subsidi silang antar provinsi."),
    "Zona Transisi / Waspada": (
        "Pemantauan intensif harga komoditas hortikultura dan penguatan "
        "cadangan penyangga sebelum musim rawan."),
    "Zona Mandiri & Stabil": (
        "Optimalisasi buffer stock dan penetapan sebagai penyokong zona rentan."),
    "Zona Hub Logistik (Surplus CPPD)": (
        "Ditetapkan sebagai hub distribusi nasional; surplus CPPD disalurkan "
        "ke zona rentan terdekat."),
}

st.set_page_config(page_title="Stabilitas Harga Pangan — K-Means",
                   page_icon="🌾", layout="wide")


@st.cache_resource
def load_artifacts():
    kmeans = joblib.load(ROOT / "models" / "kmeans_model.pkl")
    scaler = joblib.load(ROOT / "models" / "scaler.pkl")
    meta = joblib.load(ROOT / "models" / "meta.pkl")
    return kmeans, scaler, meta


@st.cache_data
def load_data():
    df = pd.read_csv(ROOT / "data" / "processed" / "hasil_clustering.csv")
    kmeans_grid = pd.read_csv(ROOT / "reports" / "kmeans_grid.csv")
    dbscan_grid = pd.read_csv(ROOT / "reports" / "dbscan_grid.csv")
    return df, kmeans_grid, dbscan_grid


def predict_df(samples: pd.DataFrame, kmeans, scaler, meta) -> pd.DataFrame:
    X = samples[meta["features"]].astype(float).values
    labels = kmeans.predict(scaler.transform(X))
    out = samples.copy()
    out["Cluster"] = labels
    out["Segmen"] = [meta["label_names"][l] for l in labels]
    return out


kmeans, scaler, meta = load_artifacts()
df, kmeans_grid, dbscan_grid = load_data()
FEATURES = meta.get("features", FEATURES_FALLBACK)

st.title("🌾 Segmentasi Stabilitas Harga Pangan Nasional")
st.caption("K-Means Clustering berdasarkan Indeks Koefisien Variasi (2022–2024) "
           "dan Cadangan Pangan Pemerintah Daerah (2025) — "
           "Fadhlilah Musa Ulil Albab · A11.2024.15975 · UDINUS")

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["📊 Dashboard EDA", "🤖 Model Demo", "📈 Evaluasi Model",
     "🧠 Interpretasi", "📚 Dokumentasi"])

# ============ TAB 1 — DASHBOARD EDA ============
with tab1:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Jumlah Provinsi", df["Provinsi"].nunique())
    c2.metric("Rata-rata KV 2022–2024", f"{df['KV_mean'].mean():.2f}%")
    c3.metric("Total CPPD 2025", f"{df['CPPD_Ton'].sum():,.0f} Ton")
    c4.metric("Jumlah Segmen", df["Segmen"].nunique())

    st.subheader("Ranking Provinsi Berdasarkan Volatilitas Harga")
    d = df.sort_values("KV_mean")
    fig = px.bar(d, x="KV_mean", y="Provinsi", color="Segmen", orientation="h",
                 color_discrete_map=SEG_COLORS,
                 labels={"KV_mean": "KV rata-rata 2022–2024 (%)"}, height=650)
    fig.update_layout(legend=dict(orientation="h", y=-0.1))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Cadangan Pangan vs Volatilitas Harga")
    fig2 = px.scatter(df, x="CPPD_Ton", y="KV_mean", color="Segmen",
                      color_discrete_map=SEG_COLORS, log_x=True,
                      hover_name="Provinsi", text="Provinsi",
                      labels={"CPPD_Ton": "CPPD 2025 (Ton, skala log)",
                              "KV_mean": "KV rata-rata (%)"}, height=550)
    fig2.update_traces(textposition="top center", textfont_size=9)
    st.plotly_chart(fig2, use_container_width=True)
    r = df["CPPD_Ton"].corr(df["KV_mean"])
    st.info(f"Korelasi CPPD vs KV = **{r:.3f}** — hipotesis 'cadangan besar ↔ "
            "harga stabil' tidak terbukti; keduanya dimensi independen dalam clustering.")

# ============ TAB 2 — MODEL DEMO ============
with tab2:
    st.subheader("Prediksi Segmen Provinsi")
    st.write("Masukkan nilai fitur secara manual **atau** upload CSV "
             f"dengan kolom: `{'`, `'.join(FEATURES)}`.")

    with st.form("manual_form"):
        cols = st.columns(4)
        vals = {
            "KV_2022": cols[0].number_input("KV 2022 (%)", 0.0, 100.0, 9.4, 0.1),
            "KV_2023": cols[1].number_input("KV 2023 (%)", 0.0, 100.0, 4.1, 0.1),
            "KV_2024": cols[2].number_input("KV 2024 (%)", 0.0, 100.0, 5.6, 0.1),
            "CPPD_Ton": cols[3].number_input("CPPD (Ton)", 0.0, 100000.0, 150.0, 10.0),
        }
        submitted = st.form_submit_button("🔍 Prediksi Segmen")
    if submitted:
        res = predict_df(pd.DataFrame([vals]), kmeans, scaler, meta)
        seg = res["Segmen"].iloc[0]
        st.success(f"**Cluster {res['Cluster'].iloc[0]} — {seg}**")
        st.markdown(f"**Rekomendasi kebijakan:** {REKOMENDASI.get(seg, '-')}")

    st.divider()
    up = st.file_uploader("Upload CSV (batch prediksi)", type="csv")
    if up is not None:
        try:
            batch = pd.read_csv(up)
            missing = [c for c in FEATURES if c not in batch.columns]
            if missing:
                st.error(f"Kolom wajib tidak ditemukan: {missing}. "
                         f"CSV harus memuat kolom: {FEATURES}.")
            elif batch[FEATURES].isna().any().any():
                st.error("Terdapat nilai kosong pada kolom fitur — lengkapi dahulu.")
            else:
                res = predict_df(batch, kmeans, scaler, meta)
                st.dataframe(res, use_container_width=True)
                st.download_button("⬇️ Unduh hasil", res.to_csv(index=False),
                                   "hasil_prediksi.csv", "text/csv")
        except Exception as e:
            st.error(f"Gagal membaca CSV: {e}")

# ============ TAB 3 — EVALUASI MODEL ============
with tab3:
    best = kmeans_grid.sort_values("silhouette", ascending=False).iloc[0]
    c1, c2, c3 = st.columns(3)
    c1.metric("Silhouette Score", f"{best['silhouette']:.4f}")
    c2.metric("Davies-Bouldin Index", f"{best['davies_bouldin']:.4f}")
    c3.metric("K Optimal", int(best["K"]))

    sub = kmeans_grid[(kmeans_grid["init"] == "k-means++") & (kmeans_grid["n_init"] == 30)]
    c1, c2 = st.columns(2)
    with c1:
        fig = px.line(sub, x="K", y="inertia", markers=True,
                      title="Elbow Method (Inertia per K)")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.line(sub, x="K", y=["silhouette", "davies_bouldin"], markers=True,
                      title="Silhouette & Davies-Bouldin per K")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Grid Search DBSCAN (baseline)")
    st.dataframe(dbscan_grid.sort_values("silhouette_non_noise", ascending=False),
                 use_container_width=True, height=280)
    st.caption(
        "**Justifikasi pemilihan K-Means:** DBSCAN terbaik mencapai silhouette "
        "≈0.64, tetapi mengorbankan ~44% provinsi sebagai noise. Kebijakan "
        "stabilisasi pangan harus mencakup SEMUA provinsi, sehingga K-Means "
        "(silhouette 0.41, cakupan 100%) dipilih sebagai model final.")

# ============ TAB 4 — INTERPRETASI ============
with tab4:
    st.subheader("Profil Centroid per Cluster (skala asli)")
    centroids = pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_),
                             columns=FEATURES).round(2)
    centroids.insert(0, "Segmen", [meta["label_names"][i] for i in centroids.index])
    centroids.insert(0, "Cluster", centroids.index)
    st.dataframe(centroids, use_container_width=True, hide_index=True)

    for cid in sorted(df["Cluster"].unique()):
        grp = df[df["Cluster"] == cid]
        seg = grp["Segmen"].iloc[0]
        with st.expander(f"Cluster {cid} — {seg} ({len(grp)} provinsi)"):
            st.markdown("**Anggota:** " + ", ".join(grp["Provinsi"]))
            st.markdown(
                f"**Profil:** KV rata-rata {grp['KV_mean'].mean():.2f}% · "
                f"CPPD rata-rata {grp['CPPD_Ton'].mean():,.1f} Ton")
            st.markdown(f"**Strategi intervensi:** {REKOMENDASI.get(seg, '-')}")

# ============ TAB 5 — DOKUMENTASI ============
with tab5:
    st.markdown(f"""
### Dataset
| Dataset | Cakupan | Peran |
|---|---|---|
| Koefisien Variasi harga produsen (Badan Pangan Nasional) | 26 provinsi × 15 komoditas × 36 bulan (2022–2024) | Fitur `KV_2022`, `KV_2023`, `KV_2024` |
| Cadangan Pangan Pemerintah Daerah / CPPD | 38 provinsi, bulanan 2025 | Fitur `CPPD_Ton` (rata-rata 2025) |
| Sertifikat Jaminan Keamanan Pangan | Nasional, 2018–2024 | Konteks EDA saja (tidak bisa per provinsi) |

### Metodologi
1. **Preprocessing** — parsing format lokal (separator `;`, desimal koma, nilai `%`),
   normalisasi nama provinsi/komoditas, koreksi typo tahun, agregasi per provinsi,
   merge inner (25 provinsi), imputasi median.
2. **Fitur** — `{FEATURES}` dengan StandardScaler. Tanpa train/test split
   (unsupervised, pemetaan populasi penuh).
3. **Modeling** — grid search K-Means (42 kombinasi) vs DBSCAN (24 kombinasi);
   K optimal = 3 via Silhouette & Davies-Bouldin; `random_state=42`.
4. **Labeling** — centroid CPPD > 3× median → Hub Logistik; sisanya diurutkan
   KV rata-rata → Mandiri & Stabil / Rentan Fluktuatif.

### Cara Pakai
- **Dashboard EDA** — eksplorasi ranking provinsi & hubungan CPPD-KV.
- **Model Demo** — isi 4 fitur atau upload CSV untuk prediksi segmen.
- **Evaluasi Model** — metrik, elbow, dan perbandingan dengan DBSCAN.
- **Interpretasi** — profil centroid, anggota, dan strategi per segmen.

### Limitasi
- Cakupan 25 provinsi (dibatasi ketersediaan data KV produsen).
- Fitur sertifikat JKP dibatalkan (data hanya nasional) — deviasi dari proposal.
- Hipotesis CPPD↔stabilitas tidak terbukti (r ≈ −0.04) — dibahas di laporan.
""")

st.sidebar.markdown("### Capstone Project UAS\nPembelajaran Mesin — UDINUS")
st.sidebar.markdown("**Model:** K-Means (K=3)\n\n**Silhouette:** 0.406\n\n**DBI:** 0.764")
