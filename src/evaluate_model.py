"""Evaluasi & visualisasi model: elbow, silhouette-per-K, PCA 2D, profil centroid.

Jalankan: python src/evaluate_model.py
Output : reports/figures/*.png + ringkasan metrik di stdout.
"""
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import davies_bouldin_score, silhouette_score

try:
    from utils import (DATA_PROCESSED, FEATURES, FIGURES_DIR, MODELS_DIR,
                       REPORTS_DIR, ensure_dirs)
except ImportError:
    from src.utils import (DATA_PROCESSED, FEATURES, FIGURES_DIR, MODELS_DIR,
                           REPORTS_DIR, ensure_dirs)

sns.set_theme(style="whitegrid")


def load_artifacts():
    kmeans = joblib.load(MODELS_DIR / "kmeans_model.pkl")
    scaler = joblib.load(MODELS_DIR / "scaler.pkl")
    meta = joblib.load(MODELS_DIR / "meta.pkl")
    df = pd.read_csv(DATA_PROCESSED / "hasil_clustering.csv")
    return kmeans, scaler, meta, df


def plot_elbow(grid: pd.DataFrame, save: bool = True):
    """Elbow plot dari inertia terbaik per K (init=k-means++, n_init=30)."""
    sub = grid[(grid["init"] == "k-means++") & (grid["n_init"] == 30)]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(sub["K"], sub["inertia"], "o-", color="steelblue")
    ax.set_xlabel("Jumlah Cluster (K)")
    ax.set_ylabel("Inertia (WCSS)")
    ax.set_title("Elbow Method — K-Means")
    if save:
        fig.savefig(FIGURES_DIR / "elbow_plot.png", dpi=150, bbox_inches="tight")
    return fig


def plot_silhouette_per_k(grid: pd.DataFrame, save: bool = True):
    sub = grid[(grid["init"] == "k-means++") & (grid["n_init"] == 30)]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(sub["K"], sub["silhouette"], "o-", color="seagreen", label="Silhouette")
    ax.plot(sub["K"], sub["davies_bouldin"], "s--", color="indianred",
            label="Davies-Bouldin")
    best_k = sub.loc[sub["silhouette"].idxmax(), "K"]
    ax.axvline(best_k, color="gray", ls=":", label=f"K optimal = {int(best_k)}")
    ax.set_xlabel("Jumlah Cluster (K)")
    ax.set_ylabel("Skor")
    ax.set_title("Silhouette & Davies-Bouldin per K")
    ax.legend()
    if save:
        fig.savefig(FIGURES_DIR / "silhouette_per_k.png", dpi=150, bbox_inches="tight")
    return fig


def plot_pca_clusters(df, scaler, save: bool = True):
    """Scatter PCA 2D berwarna cluster dengan anotasi nama provinsi."""
    X = scaler.transform(df[FEATURES].values)
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X)
    fig, ax = plt.subplots(figsize=(11, 8))
    palette = sns.color_palette("Set1", df["Cluster"].nunique())
    for cid, color in zip(sorted(df["Cluster"].unique()), palette):
        m = df["Cluster"] == cid
        seg = df.loc[m, "Segmen"].iloc[0]
        ax.scatter(coords[m, 0], coords[m, 1], s=90, color=color,
                   label=f"Cluster {cid} — {seg}", edgecolor="k", zorder=3)
    for i, prov in enumerate(df["Provinsi"]):
        ax.annotate(prov, (coords[i, 0], coords[i, 1]), fontsize=7,
                    xytext=(4, 4), textcoords="offset points")
    var = pca.explained_variance_ratio_
    ax.set_xlabel(f"PC1 ({var[0]:.1%} varians)")
    ax.set_ylabel(f"PC2 ({var[1]:.1%} varians)")
    ax.set_title("Hasil Clustering K-Means — Proyeksi PCA 2D")
    ax.legend(fontsize=8)
    if save:
        fig.savefig(FIGURES_DIR / "pca_clusters.png", dpi=150, bbox_inches="tight")
    return fig


def plot_centroid_heatmap(kmeans, scaler, meta, save: bool = True):
    """Heatmap profil centroid (skala asli) — interpretasi fitur pengganti
    SHAP/LIME yang tidak berlaku untuk unsupervised learning."""
    centroids = pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_),
                             columns=meta["features"])
    centroids.index = [f"C{i}: {meta['label_names'][i]}" for i in centroids.index]
    # Normalisasi per kolom agar warna sebanding antar fitur
    norm = (centroids - centroids.min()) / (centroids.max() - centroids.min())
    fig, ax = plt.subplots(figsize=(9, 4.5))
    sns.heatmap(norm, annot=centroids.round(2), fmt="", cmap="YlOrRd",
                cbar_kws={"label": "Relatif (0-1 per fitur)"}, ax=ax)
    ax.set_title("Profil Centroid per Cluster (nilai asli, warna dinormalisasi)")
    if save:
        fig.savefig(FIGURES_DIR / "centroid_heatmap.png", dpi=150, bbox_inches="tight")
    return fig


def main():
    ensure_dirs()
    kmeans, scaler, meta, df = load_artifacts()
    X = scaler.transform(df[FEATURES].values)

    sil = silhouette_score(X, df["Cluster"])
    dbi = davies_bouldin_score(X, df["Cluster"])
    print(f"Model final K-Means (K={kmeans.n_clusters}): "
          f"Silhouette={sil:.4f}, Davies-Bouldin={dbi:.4f}, Inertia={kmeans.inertia_:.2f}")

    grid = pd.read_csv(REPORTS_DIR / "kmeans_grid.csv")
    plot_elbow(grid)
    plot_silhouette_per_k(grid)
    plot_pca_clusters(df, scaler)
    plot_centroid_heatmap(kmeans, scaler, meta)
    print(f"Figure tersimpan di {FIGURES_DIR}")
    return sil, dbi


if __name__ == "__main__":
    main()
