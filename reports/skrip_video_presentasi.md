# Skrip Video Presentasi YouTube (±10 Menit)

**Judul video:** Capstone Project Pembelajaran Mesin — Clustering Stabilitas Harga Pangan Nasional dengan K-Means
**Fadhlilah Musa Ulil Albab · A11.2024.15975 · UDINUS**

> Total narasi ±1.400 kata ≈ 9–10 menit dengan tempo bicara santai.
> Kolom "Tampilan Layar" = apa yang direkam (screen record) saat narasi dibacakan.
> Tips rekam: buka dulu semua yang dibutuhkan (VS Code/browser notebook, aplikasi Streamlit di localhost, folder repo) supaya tidak ada jeda loading.

---

## SEGMEN 1 — Pembukaan (0:00 – 0:45)

**Tampilan layar:** slide judul / cover laporan, lalu wajah (opsional).

> Assalamualaikum warahmatullahi wabarakatuh. Halo semuanya, perkenalkan saya Fadhlilah Musa Ulil Albab, NIM A11.2024.15975, dari program studi Teknik Informatika Universitas Dian Nuswantoro.
>
> Pada video ini saya akan mempresentasikan capstone project Ujian Akhir Semester mata kuliah Pembelajaran Mesin, dengan judul: **Analisis Pengelompokan Tingkat Stabilitas Harga Pangan Nasional Menggunakan Algoritma K-Means Clustering Berdasarkan Indeks Koefisien Variasi**.
>
> Singkatnya: saya mengelompokkan provinsi-provinsi di Indonesia berdasarkan seberapa stabil harga pangannya dan seberapa besar cadangan pangannya, lalu menerjemahkan hasil cluster menjadi rekomendasi kebijakan.

## SEGMEN 2 — Latar Belakang & Problem Statement (0:45 – 2:00)

**Tampilan layar:** BAB I laporan, atau slide berisi 2 poin: KV dan CPPD.

> Kenapa topik ini penting? Harga pangan yang naik-turun tajam itu merugikan dua arah — petani tidak pasti pendapatannya, konsumen terancam inflasi pangan. Pemerintah memantau ini lewat **Indeks Koefisien Variasi** atau KV: rasio simpangan baku terhadap rata-rata harga. Semakin kecil KV, semakin stabil harga.
>
> Di sisi lain, setiap pemerintah daerah wajib punya **Cadangan Pangan Pemerintah Daerah** atau CPPD, sebagai alat intervensi ketika harga bergejolak.
>
> Nah, masalahnya: belum ada pemetaan yang menggabungkan dua dimensi ini. Provinsi mana yang rentan fluktuasi harga? Mana yang stabil? Mana yang bisa jadi penyangga logistik nasional?
>
> Karena tidak ada label kelas — tidak ada daftar resmi "provinsi rentan" — maka ini adalah masalah **unsupervised learning**, dan saya menyelesaikannya dengan **K-Means Clustering**, dibandingkan dengan **DBSCAN** sebagai baseline.

## SEGMEN 3 — Dataset & Preprocessing (2:00 – 3:45)

**Tampilan layar:** buka `notebooks/01_eda.ipynb` — scroll bagian load data, tunjukkan CSV mentah sekilas (separator `;`, nilai `"7,09%"`), lalu tabel dataset terintegrasi.

> Saya menggunakan tiga dataset publik dari Badan Pangan Nasional.
>
> Pertama, data **Koefisien Variasi harga tingkat produsen 2022 sampai 2024** — sebelas ribu dua ratus baris, 15 komoditas pangan strategis, 26 provinsi, granularitas bulanan.
>
> Kedua, data **CPPD tahun 2025** — stok cadangan pangan bulanan untuk 38 provinsi.
>
> Ketiga, data **sertifikat Jaminan Keamanan Pangan** 2018 sampai 2024.
>
> Datasetnya menantang secara teknis. Separatornya titik-koma, desimalnya pakai koma, nilai KV tersimpan sebagai teks seperti "tujuh koma nol sembilan persen", ada typo tahun "202" yang harus dikoreksi jadi 2020, nama provinsi tidak konsisten antar file — "D titik I Yogyakarta" versus "DI Yogyakarta" — dan 40 persen nilai KV-nya kosong.
>
> Alur preprocessing-nya: parsing dan pembersihan semua format tadi, normalisasi nama provinsi dan komoditas, lalu agregasi — rata-rata KV per provinsi per tahun menjadi kolom KV 2022, KV 2023, KV 2024, dan rata-rata CPPD 2025 per provinsi. Setelah merge inner, hasilnya **25 provinsi dengan 4 fitur**, yang kemudian distandardisasi pakai StandardScaler — ini penting, karena tanpa scaling, CPPD yang nilainya ribuan ton akan mendominasi perhitungan jarak dibanding KV yang cuma satuan persen.
>
> Satu catatan jujur: di proposal, sertifikat JKP mau saya jadikan fitur. Ternyata datanya hanya level nasional, tidak per provinsi — jadi tidak bisa dipakai untuk clustering. Ini saya dokumentasikan sebagai deviasi dan limitasi, dan datanya tetap dipakai untuk konteks EDA.

## SEGMEN 4 — EDA: 5 Insight (3:45 – 5:15)

**Tampilan layar:** scroll notebook 01, berhenti di tiap visualisasi sesuai narasi.

> Dari eksplorasi data, ada lima insight utama.
>
> **Satu** — volatilitas harga sangat dinamis antar tahun: rata-rata KV 9,55 persen di 2022, turun drastis ke 3,95 di 2023, naik lagi ke 5 persen di 2024, seiring El Niño.
>
> **Dua** — kesenjangan antar provinsi besar: Kalimantan Tengah paling stabil di angka 4 persen, Sulawesi Utara paling fluktuatif hampir 10 persen.
>
> **Tiga** — dari heatmap korelasi, KV antar tahun berkorelasi positif: provinsi yang fluktuatif cenderung tetap fluktuatif. Artinya masalahnya struktural, bukan kebetulan.
>
> **Empat** — ini temuan paling menarik. Hipotesis awal saya: provinsi dengan cadangan pangan besar harusnya harganya lebih stabil. Ternyata korelasinya minus nol koma nol tiga tujuh — praktis **nol**. Hipotesis **tidak terbukti**. Tapi ini temuan yang valid dan penting: besarnya stok saja tidak menjamin stabilitas harga — yang lebih menentukan kemungkinan adalah distribusinya. Dan justru karena kedua fitur ini independen, keduanya layak dipakai bersama dalam clustering.
>
> **Lima** — komoditas paling fluktuatif adalah cabai rawit merah, cabai merah keriting, dan bawang merah — khas hortikultura yang mudah busuk dan panennya musiman.

## SEGMEN 5 — Modeling: K-Means vs DBSCAN (5:15 – 7:15)

**Tampilan layar:** buka `notebooks/02_modeling.ipynb` — tunjukkan tabel grid search, elbow plot, silhouette plot, tabel perbandingan DBSCAN.

> Masuk ke pemodelan. Saya melakukan **grid search penuh** untuk dua algoritma, semuanya dengan random state 42 supaya reproducible.
>
> Untuk K-Means: K dari 2 sampai 8, dikali dua metode inisialisasi — k-means++ dan random — dikali tiga nilai n_init. Total **42 kombinasi**. Untuk DBSCAN: enam nilai eps dikali empat nilai min_samples, total **24 kombinasi**.
>
> Metrik evaluasinya tiga: **Silhouette Score** — semakin tinggi semakin baik, **Davies-Bouldin Index** — semakin rendah semakin baik, dan **Inertia** untuk analisis elbow.
>
> Hasilnya: elbow plot menunjukkan siku di K sama dengan 3, dan silhouette tertinggi juga di K sama dengan 3, yaitu **0,406**, dengan Davies-Bouldin 0,764.
>
> Di sini juga ada analog diskusi overfitting-underfitting untuk clustering: K=2 itu underfit — terlalu general, cuma misahin outlier dari sisanya. K=5 ke atas itu overfit — silhouette anjlok dan muncul cluster beranggota satu-dua provinsi yang tidak bermakna untuk kebijakan. K=3 paling seimbang.
>
> Sekarang pembandingnya. DBSCAN terbaik mencapai silhouette **0,636** — lebih tinggi dari K-Means! Tapi ada harga yang harus dibayar: DBSCAN membuang **44 persen provinsi sebagai noise**. Sebelas dari dua puluh lima provinsi tidak dapat cluster.
>
> Untuk kebijakan pangan, itu tidak bisa diterima — setiap provinsi wajib dapat segmen, dan provinsi yang dianggap "noise" justru sering yang paling butuh intervensi. Silhouette DBSCAN tinggi karena dia cuma dievaluasi pada data yang gampang. Jadi keputusan final saya: **K-Means, K=3**, cakupan 100 persen.

## SEGMEN 6 — Hasil & Interpretasi (7:15 – 8:30)

**Tampilan layar:** scatter PCA 2D, heatmap profil centroid, lalu notebook 03 bagian anggota segmen.

> Ini hasil clusternya, divisualisasikan dengan PCA dua dimensi.
>
> Karena SHAP dan LIME tidak berlaku untuk unsupervised learning — tidak ada target yang diprediksi — interpretasi fitur saya lakukan lewat **profil centroid**, nilai rata-rata tiap fitur per cluster di skala asli.
>
> Hasilnya tiga segmen. **Zona Mandiri dan Stabil** — 17 provinsi dengan KV rendah dan konsisten. **Zona Rentan Fluktuatif** — 7 provinsi dengan KV tinggi di semua tahun: Bengkulu, DI Yogyakarta, Gorontalo, Jawa Timur, dan tiga provinsi Sulawesi. Dan **Zona Hub Logistik** — satu provinsi, Jawa Barat, dengan CPPD ekstrem dua ribu seratus lima puluh ton, lebih dari sepuluh kali lipat provinsi lain. Cluster beranggota tunggal ini bukan kegagalan model — ini realita outlier yang justru bermakna kebijakan.
>
> Rekomendasinya: zona rentan dapat prioritas operasi pasar dan percepatan logistik, terutama untuk cabai dan bawang merah. Zona stabil jadi penyokong. Dan Jawa Barat ditetapkan sebagai hub distribusi yang menyalurkan surplus cadangannya ke zona rentan terdekat.

## SEGMEN 7 — Demo Aplikasi Streamlit (8:30 – 9:40)

**Tampilan layar:** demo langsung aplikasi (`streamlit run app/app.py`). Klik tiap tab sesuai narasi. Di tab Model Demo, isi form dan klik prediksi; kalau sempat, upload CSV.

> Terakhir, seluruh pipeline ini di-deploy sebagai aplikasi web Streamlit dengan lima tab.
>
> Tab pertama, **Dashboard EDA** — ada metric cards dan ranking provinsi interaktif yang warnanya sesuai segmen.
>
> Tab kedua, **Model Demo** — pengguna bisa memasukkan empat nilai fitur secara manual... saya isi contoh... klik prediksi... dan keluar cluster, label segmen, plus rekomendasi kebijakannya. Bisa juga upload CSV untuk prediksi banyak data sekaligus, lengkap dengan validasi kolom.
>
> Tab ketiga, **Evaluasi Model** — menampilkan metrik, elbow, dan tabel grid DBSCAN beserta justifikasi pemilihan K-Means.
>
> Tab keempat, **Interpretasi** — profil centroid dan daftar anggota tiap segmen dengan strategi intervensinya.
>
> Dan tab kelima, **Dokumentasi** — dataset, metodologi, dan limitasi.

## SEGMEN 8 — Kesimpulan & Penutup (9:40 – 10:45)

**Tampilan layar:** slide kesimpulan / BAB VI laporan, akhiri dengan slide berisi link GitHub + Streamlit.

> Kesimpulannya: pipeline machine learning end-to-end berhasil dibangun — dari tiga dataset mentah yang formatnya menantang, sampai dashboard interaktif. K-Means dengan K=3 menghasilkan tiga segmen yang actionable dengan cakupan 100 persen provinsi.
>
> Dua catatan jujur sebagai limitasi: cakupan baru 25 dari 38 provinsi karena keterbatasan data KV, dan silhouette 0,41 memang di bawah target proposal 0,5 — wajar untuk data sosial-ekonomi yang batas antar kelompoknya kontinu.
>
> Dan sebagai pengembangan ke depan, model ini justru makin relevan dengan kondisi terkini. Pelemahan rupiah di 2026 menekan harga pangan lewat komoditas dan input produksi yang masih impor — seperti kedelai dan pakan ternak — dan provinsi di zona rentan adalah yang paling terekspos. Program Makan Bergizi Gratis juga terbukti menggerakkan harga ayam dan telur di tingkat produsen mengikuti kalender sekolah — itu sumber volatilitas baru yang persis terukur oleh Koefisien Variasi. Jadi begitu data KV 2025–2026 dirilis, model ini tinggal dijalankan ulang untuk melihat apakah keanggotaan segmen bergeser — dan peta ini bisa jadi alat prioritisasi intervensi yang selalu mutakhir.
>
> Seluruh kode, notebook, model, dan laporan tersedia di repository GitHub, dan aplikasinya bisa dicoba langsung lewat link Streamlit di deskripsi video.
>
> Sekian presentasi dari saya. Terima kasih sudah menonton. Wassalamualaikum warahmatullahi wabarakatuh.

---

## Checklist Sebelum Rekam

- [ ] Jalankan `streamlit run app/app.py` dan biarkan terbuka di browser
- [ ] Buka ketiga notebook (sudah ada outputnya, tidak perlu re-run)
- [ ] Siapkan 1 file CSV kecil untuk demo upload (kolom: KV_2022, KV_2023, KV_2024, CPPD_Ton)
- [ ] Isi link GitHub & Streamlit di slide penutup dan deskripsi video
- [ ] Cek durasi tiap segmen saat gladi — kalau lebih dari 10:45, pangkas Segmen 4 (insight 2 dan 5 bisa dipercepat) atau ringkas paragraf rupiah/MBG di Segmen 8 jadi 2 kalimat
- [ ] Kalau dosen tanya "kenapa tidak pakai data 2025–2026?": jawab bahwa KV produsen per provinsi untuk 2025–2026 belum dirilis Badan Pangan Nasional; yang tersedia baru harga bulanan level nasional — makanya dijadikan rencana validasi temporal di rekomendasi pengembangan (BAB VI), bukan dipaksakan masuk model
