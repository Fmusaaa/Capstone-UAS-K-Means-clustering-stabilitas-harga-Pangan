"""Inference: prediksi segmen untuk sampel baru dari model tersimpan.

Jalankan: python src/inference.py  (memprediksi 1 sampel dummy)
"""
import joblib
import numpy as np
import pandas as pd

try:
    from utils import MODELS_DIR
except ImportError:
    from src.utils import MODELS_DIR


def load_artifacts():
    kmeans = joblib.load(MODELS_DIR / "kmeans_model.pkl")
    scaler = joblib.load(MODELS_DIR / "scaler.pkl")
    meta = joblib.load(MODELS_DIR / "meta.pkl")
    return kmeans, scaler, meta


def predict(samples, kmeans=None, scaler=None, meta=None) -> pd.DataFrame:
    """samples: DataFrame/array dengan kolom [KV_2022, KV_2023, KV_2024, CPPD_Ton].

    Return DataFrame dengan kolom Cluster dan Segmen.
    """
    if kmeans is None:
        kmeans, scaler, meta = load_artifacts()
    features = meta["features"]
    if isinstance(samples, pd.DataFrame):
        missing = [c for c in features if c not in samples.columns]
        if missing:
            raise ValueError(f"Kolom wajib tidak ditemukan: {missing}")
        X = samples[features].values
        result = samples.copy()
    else:
        X = np.atleast_2d(np.asarray(samples, dtype=float))
        result = pd.DataFrame(X, columns=features)
    labels = kmeans.predict(scaler.transform(X))
    result["Cluster"] = labels
    result["Segmen"] = [meta["label_names"][l] for l in labels]
    return result


if __name__ == "__main__":
    dummy = pd.DataFrame([{"KV_2022": 9.5, "KV_2023": 4.0,
                           "KV_2024": 5.5, "CPPD_Ton": 120.0}])
    print(predict(dummy).to_string(index=False))
