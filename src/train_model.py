"""Training & tuning: grid search K-Means vs DBSCAN, model final + labeling.

Jalankan: python src/train_model.py
Output : models/{kmeans_model,scaler,meta}.pkl,
         reports/{kmeans_grid,dbscan_grid}.csv,
         data/processed/hasil_clustering.csv
"""
import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler

try:
    from utils import (DATA_PROCESSED, FEATURES, MODELS_DIR, RANDOM_STATE,
                       REPORTS_DIR, ensure_dirs)
    from data_preprocessing import build_integrated_dataset
except ImportError:
    from src.utils import (DATA_PROCESSED, FEATURES, MODELS_DIR, RANDOM_STATE,
                           REPORTS_DIR, ensure_dirs)
    from src.data_preprocessing import build_integrated_dataset


def load_dataset() -> pd.DataFrame:
    path = DATA_PROCESSED / "dataset_pangan_terintegrasi.csv"
    if path.exists():
        return pd.read_csv(path)
    return build_integrated_dataset()


def kmeans_grid_search(X) -> pd.DataFrame:
    """Grid K in 2..8 x init x n_init. Metrik: silhouette, DBI, inertia."""
    rows = []
    for k in range(2, 9):
        for init in ("k-means++", "random"):
            for n_init in (10, 20, 30):
                km = KMeans(n_clusters=k, init=init, n_init=n_init,
                            max_iter=300, random_state=RANDOM_STATE).fit(X)
                rows.append({
                    "K": k, "init": init, "n_init": n_init,
                    "inertia": km.inertia_,
                    "silhouette": silhouette_score(X, km.labels_),
                    "davies_bouldin": davies_bouldin_score(X, km.labels_),
                })
    return pd.DataFrame(rows)


def dbscan_grid_search(X) -> pd.DataFrame:
    """Grid eps x min_samples. Silhouette dihitung hanya pada non-noise points."""
    rows = []
    for eps in (0.3, 0.5, 0.7, 1.0, 1.3, 1.6):
        for min_samples in (2, 3, 4, 5):
            db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
            labels = db.labels_
            mask = labels != -1
            n_clusters = len(set(labels[mask])) if mask.any() else 0
            sil = np.nan
            dbi = np.nan
            if n_clusters >= 2 and mask.sum() > n_clusters:
                sil = silhouette_score(X[mask], labels[mask])
                dbi = davies_bouldin_score(X[mask], labels[mask])
            rows.append({
                "eps": eps, "min_samples": min_samples,
                "n_clusters": n_clusters,
                "n_noise": int((~mask).sum()),
                "noise_pct": round(100 * (~mask).mean(), 1),
                "silhouette_non_noise": sil,
                "davies_bouldin_non_noise": dbi,
            })
    return pd.DataFrame(rows)


def label_clusters(df: pd.DataFrame, centroids: pd.DataFrame) -> dict:
    """Labeling sesuai PRD 4.4:
    - centroid CPPD_Ton > 3x median centroid -> Zona Hub Logistik (Surplus CPPD)
    - sisanya diurutkan KV_mean centroid ascending -> Mandiri & Stabil,
      (Transisi/Waspada jika ada 3 non-hub), Rentan Fluktuatif.
    """
    med = centroids["CPPD_Ton"].median()
    hub_ids = centroids.index[centroids["CPPD_Ton"] > 3 * med].tolist()
    labels = {cid: "Zona Hub Logistik (Surplus CPPD)" for cid in hub_ids}

    non_hub = centroids.drop(index=hub_ids).copy()
    kv_cols = [c for c in centroids.columns if c.startswith("KV_")]
    non_hub["KV_mean_c"] = non_hub[kv_cols].mean(axis=1)
    ordered = non_hub.sort_values("KV_mean_c").index.tolist()
    if len(ordered) == 2:
        names = ["Zona Mandiri & Stabil", "Zona Rentan Fluktuatif"]
    elif len(ordered) == 3:
        names = ["Zona Mandiri & Stabil", "Zona Transisi / Waspada",
                 "Zona Rentan Fluktuatif"]
    else:
        names = [f"Zona Tingkat {i+1}" for i in range(len(ordered))]
        if ordered:
            names[0] = "Zona Mandiri & Stabil"
            names[-1] = "Zona Rentan Fluktuatif"
    for cid, name in zip(ordered, names):
        labels[cid] = name
    return labels


def main():
    ensure_dirs()
    df = load_dataset()
    X_raw = df[FEATURES].values

    scaler = StandardScaler().fit(X_raw)
    X = scaler.transform(X_raw)

    # --- Grid search ---
    kmeans_grid = kmeans_grid_search(X)
    kmeans_grid.to_csv(REPORTS_DIR / "kmeans_grid.csv", index=False)
    dbscan_grid = dbscan_grid_search(X)
    dbscan_grid.to_csv(REPORTS_DIR / "dbscan_grid.csv", index=False)

    # --- Pilih konfigurasi K-Means terbaik (silhouette tertinggi) ---
    best = kmeans_grid.sort_values("silhouette", ascending=False).iloc[0]
    best_k = int(best["K"])
    print(f"Best K-Means: K={best_k}, init={best['init']}, n_init={int(best['n_init'])}, "
          f"silhouette={best['silhouette']:.4f}, DBI={best['davies_bouldin']:.4f}")

    kmeans = KMeans(n_clusters=best_k, init=best["init"], n_init=int(best["n_init"]),
                    max_iter=300, random_state=RANDOM_STATE).fit(X)
    df["Cluster"] = kmeans.labels_

    # --- Profil centroid dalam skala asli & labeling ---
    centroids = pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_),
                             columns=FEATURES)
    label_names = label_clusters(df, centroids)
    df["Segmen"] = df["Cluster"].map(label_names)

    # --- Simpan artefak ---
    joblib.dump(kmeans, MODELS_DIR / "kmeans_model.pkl")
    joblib.dump(scaler, MODELS_DIR / "scaler.pkl")
    joblib.dump({"features": FEATURES, "label_names": label_names},
                MODELS_DIR / "meta.pkl")
    out = DATA_PROCESSED / "hasil_clustering.csv"
    df.to_csv(out, index=False)

    print(f"Tersimpan: {out}")
    print(df.groupby("Segmen")["Provinsi"].apply(list))
    return df, kmeans, scaler, label_names


if __name__ == "__main__":
    main()
