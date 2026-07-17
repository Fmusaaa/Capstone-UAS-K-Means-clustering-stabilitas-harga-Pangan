"""Utilitas bersama: path proyek, normalisasi nama, parsing nilai KV."""
from pathlib import Path

# Root repo = satu level di atas folder src/
ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

RANDOM_STATE = 42
FEATURES = ["KV_2022", "KV_2023", "KV_2024", "CPPD_Ton"]

KV_FILE = "Koefisien_Variasi_Harga_Tingkat_Produsen_Tahun_2022-2024.csv"
CPPD_FILE = "CADANGAN_PANGAN.csv"
SERTIFIKAT_FILE = "sertifikat_keamanan_pangan.csv"


def normalize_provinsi(name: str) -> str:
    """Seragamkan nama provinsi antar dataset (mis. 'D.I Yogyakarta' -> 'DI Yogyakarta')."""
    if not isinstance(name, str):
        return name
    name = " ".join(name.strip().split())
    replacements = {
        "D.I Yogyakarta": "DI Yogyakarta",
        "D.I. Yogyakarta": "DI Yogyakarta",
        "Daerah Istimewa Yogyakarta": "DI Yogyakarta",
    }
    return replacements.get(name, name)


def normalize_komoditas(name: str) -> str:
    """Seragamkan nama komoditas (mis. 'TK. Petani' vs 'Tk. Petani')."""
    if not isinstance(name, str):
        return name
    name = " ".join(name.strip().split())
    return name.replace("TK. ", "Tk. ")


def parse_kv(value) -> float:
    """Ubah string KV '7,09%' menjadi float 7.09. NaN dibiarkan NaN."""
    if value is None or (isinstance(value, float)):
        return value
    s = str(value).strip().replace("%", "").replace(",", ".")
    if s == "" or s.lower() == "nan":
        return float("nan")
    return float(s)


def ensure_dirs():
    for d in (DATA_PROCESSED, MODELS_DIR, FIGURES_DIR):
        d.mkdir(parents=True, exist_ok=True)
