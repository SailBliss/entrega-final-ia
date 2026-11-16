from datetime import date
from functools import lru_cache
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "datos.csv"

NOMBRES_CATEGORIAS = {
    0: "Buena",
    1: "Aceptable",
    2: "Por encima de aceptable",
}


def categorizar_ica(pm25: float) -> int:
    """Agrupación académica con cortes de 12 y 37 µg/m³."""
    if pm25 <= 12.0:
        return 0
    if pm25 <= 37.0:
        return 1
    return 2


@lru_cache(maxsize=8)
def _load(path, modified_ns):
    df = pd.read_csv(path, parse_dates=["fecha"])
    df["categoria"] = df["pm25"].map(categorizar_ica)
    return df.sort_values(["estacion", "fecha"]).reset_index(drop=True)


def load_clean_data(filepath=DATA_PATH) -> pd.DataFrame:
    path = Path(filepath).resolve()
    return _load(path, path.stat().st_mtime_ns).copy()


def get_stations(filepath=DATA_PATH) -> list[str]:
    return sorted(load_clean_data(filepath)["estacion"].unique().tolist())


def parse_date(value):
    try:
        return pd.Timestamp(date.fromisoformat(value))
    except (ValueError, TypeError):
        raise ValueError("Usa una fecha válida en formato AAAA-MM-DD.") from None


def station_data(df, station_id):
    rows = df[df["estacion"] == station_id]
    if rows.empty:
        raise ValueError(f"Estación «{station_id}» no encontrada. Consulta el listado de estaciones.")
    return rows


def get_pm25(station_id, start_date, end_date, filepath=DATA_PATH):
    df = load_clean_data(filepath)
    rows = station_data(df, station_id)
    start, end = parse_date(start_date), parse_date(end_date)
    if start > end:
        raise ValueError("La fecha inicial debe ser anterior o igual a la final.")
    if end < df["fecha"].min() or start > df["fecha"].max():
        raise ValueError("El intervalo queda fuera del período histórico de 2020.")
    return rows.loc[rows["fecha"].between(start, end)].copy()
