import os
import pandas as pd

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "datos.csv")

NOMBRES_CATEGORIAS = {
    0: "Buena",
    1: "Aceptable",
    2: "Dañina (Alerta)",
}

def categorizar_ica(pm25: float) -> int:
    """Clasifica concentración PM2.5 (ug/m3) según Resolución 2254 de 2017."""
    if pm25 <= 12.0:
        return 0
    elif pm25 <= 37.0:
        return 1
    return 2

def load_clean_data(filepath: str = DATA_PATH) -> pd.DataFrame:
    """Carga el dataset limpio y asigna la categoría ICA diaria."""
    df = pd.read_csv(filepath)
    df["fecha"] = pd.to_datetime(df["fecha"])
    df["categoria"] = df["pm25"].apply(categorizar_ica)
    return df.sort_values(["estacion", "fecha"]).reset_index(drop=True)

def get_stations(filepath: str = DATA_PATH) -> list[str]:
    """Devuelve la lista ordenada de estaciones disponibles."""
    df = load_clean_data(filepath)
    return sorted(df["estacion"].unique().tolist())

def get_pm25(
    station_id: str,
    start_date: str,
    end_date: str,
    filepath: str = DATA_PATH,
) -> pd.DataFrame:
    """Devuelve mediciones históricas diarias para una estación y rango de fechas."""
    df = load_clean_data(filepath)
    mask = (
        (df["estacion"] == station_id)
        & (df["fecha"] >= pd.to_datetime(start_date))
        & (df["fecha"] <= pd.to_datetime(end_date))
    )
    cols = ["fecha", "estacion", "municipio", "pm25", "categoria", "latitud", "longitud"]
    return df.loc[mask, [c for c in cols if c in df.columns]].copy()
