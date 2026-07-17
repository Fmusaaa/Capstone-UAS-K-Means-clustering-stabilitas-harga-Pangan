"""Preprocessing: load 3 CSV mentah -> dataset terintegrasi per provinsi.

Jalankan: python src/data_preprocessing.py
Output : data/processed/dataset_pangan_terintegrasi.csv
"""
import numpy as np
import pandas as pd

try:
    from utils import (DATA_PROCESSED, DATA_RAW, CPPD_FILE, KV_FILE,
                       SERTIFIKAT_FILE, ensure_dirs, normalize_komoditas,
                       normalize_provinsi, parse_kv)
except ImportError:  # dipanggil sebagai package (src.data_preprocessing)
    from src.utils import (DATA_PROCESSED, DATA_RAW, CPPD_FILE, KV_FILE,
                           SERTIFIKAT_FILE, ensure_dirs, normalize_komoditas,
                           normalize_provinsi, parse_kv)


def load_kv() -> pd.DataFrame:
    """Load KV harga produsen 2022-2024 (sep=';', skiprows=3, nilai '7,09%')."""
    df = pd.read_csv(DATA_RAW / KV_FILE, sep=";", skiprows=3)
    df.columns = [c.strip() for c in df.columns]
    df["Provinsi"] = df["Provinsi"].apply(normalize_provinsi)
    df["Komoditas"] = df["Komoditas"].apply(normalize_komoditas)
    df["KV"] = df["Koefisien Variasi"].apply(parse_kv)
    return df


def load_cppd() -> pd.DataFrame:
    """Load CPPD bulanan 2025 (sep=';', skiprows=1, decimal=',').

    Provinsi dengan CPPD 0.00 adalah data faktual dan TIDAK dihapus.
    """
    df = pd.read_csv(DATA_RAW / CPPD_FILE, sep=";", skiprows=1, decimal=",")
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns={"Wilayah": "Provinsi"})
    df["Provinsi"] = df["Provinsi"].apply(normalize_provinsi)
    df["CPPD_Ton"] = pd.to_numeric(df["CPPD_Ton"], errors="coerce")
    return df


def load_sertifikat() -> pd.DataFrame:
    """Load sertifikat JKP nasional 2018-2024; koreksi typo tahun 202 -> 2020.

    Catatan: data ini hanya level NASIONAL sehingga tidak dipakai sebagai
    fitur clustering per provinsi (deviasi dari proposal, hanya konteks EDA).
    """
    df = pd.read_csv(DATA_RAW / SERTIFIKAT_FILE, sep=";")
    df.columns = ["Jumlah_Sertifikat", "Tahun"]
    df["Tahun"] = df["Tahun"].astype(str).str.strip().replace({"202": "2020"})
    df["Tahun"] = df["Tahun"].astype(int)
    df["Jumlah_Sertifikat"] = pd.to_numeric(df["Jumlah_Sertifikat"], errors="coerce")
    return df.sort_values("Tahun").reset_index(drop=True)


def build_integrated_dataset(save: bool = True) -> pd.DataFrame:
    """Agregasi KV per (Provinsi, Tahun) -> pivot KV_2022..2024, mean CPPD 2025,
    merge inner, imputasi median, feature engineering KV_mean & KV_trend."""
    ensure_dirs()
    kv = load_kv()
    cppd = load_cppd()

    # Mean KV per (Provinsi, Tahun) lintas semua komoditas & bulan
    kv_agg = kv.groupby(["Provinsi", "Tahun"])["KV"].mean().reset_index()
    kv_pivot = kv_agg.pivot(index="Provinsi", columns="Tahun", values="KV")
    kv_pivot.columns = [f"KV_{c}" for c in kv_pivot.columns]
    kv_pivot = kv_pivot.reset_index()

    # Drop provinsi yang KV-nya kosong di semua tahun (tidak bisa diimputasi bermakna)
    kv_cols = [c for c in kv_pivot.columns if c.startswith("KV_")]
    kv_pivot = kv_pivot.dropna(subset=kv_cols, how="all").reset_index(drop=True)

    # Mean CPPD 2025 per provinsi
    cppd_agg = cppd.groupby("Provinsi")["CPPD_Ton"].mean().reset_index()

    # Merge inner: hasil +-25 provinsi (KV hanya mencakup 26 provinsi)
    df = kv_pivot.merge(cppd_agg, on="Provinsi", how="inner")

    # Imputasi missing KV per kolom dengan median
    for c in kv_cols:
        df[c] = df[c].fillna(df[c].median())

    # Feature engineering (untuk EDA saja, bukan fitur clustering)
    df["KV_mean"] = df[kv_cols].mean(axis=1)
    df["KV_trend"] = df["KV_2024"] - df["KV_2022"]

    df = df.sort_values("Provinsi").reset_index(drop=True)
    if save:
        out = DATA_PROCESSED / "dataset_pangan_terintegrasi.csv"
        df.to_csv(out, index=False)
        print(f"Tersimpan: {out} ({df.shape[0]} provinsi x {df.shape[1]} kolom)")
    return df


if __name__ == "__main__":
    df = build_integrated_dataset()
    print(df.head())
