import numpy as np
import pandas as pd
from src.data_loader import DATA_PATH, NOMBRES_CATEGORIAS, load_clean_data, parse_date, station_data

FEATURE_COLS = [
    "pm25_hoy",
    "pm25_ayer",
    "pm25_antier",
    "delta_1d",
    "media_movil_3d",
    "mes_sin",
    "mes_cos",
]

def build_features(df_estacion: pd.DataFrame) -> pd.DataFrame:
    """Genera variables rezagadas y estacionales para predecir el día t+1."""
    df = df_estacion.sort_values("fecha").copy()
    df["pm25_hoy"] = df["pm25"]
    df["pm25_ayer"] = df["pm25"].shift(1)
    df["pm25_antier"] = df["pm25"].shift(2)
    df["delta_1d"] = df["pm25_hoy"] - df["pm25_ayer"]
    df["media_movil_3d"] = df["pm25"].rolling(3).mean()

    mes = df["fecha"].dt.month
    df["mes_sin"] = np.sin(2 * np.pi * mes / 12.0)
    df["mes_cos"] = np.cos(2 * np.pi * mes / 12.0)

    consecutive = df["fecha"].diff().eq(pd.Timedelta(days=1))
    df["fecha_objetivo"] = df["fecha"].shift(-1).where(consecutive.shift(-1, fill_value=False))
    df["target_manana"] = df["categoria"].shift(-1).where(df["fecha_objetivo"].notna())
    return df.loc[consecutive & consecutive.shift(1, fill_value=False)].reset_index(drop=True)

class GaussianNaiveBayes:
    """Implementación propia de Gaussian Naive Bayes desde cero en NumPy."""

    def __init__(self, var_smoothing: float = 1e-9):
        self.var_smoothing = var_smoothing
        self.classes_ = None
        self.priors_ = None
        self.means_ = None
        self.vars_ = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        n_samples, n_features = X.shape
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)

        self.priors_ = np.zeros(n_classes, dtype=np.float64)
        self.means_ = np.zeros((n_classes, n_features), dtype=np.float64)
        self.vars_ = np.zeros((n_classes, n_features), dtype=np.float64)

        epsilon = max(self.var_smoothing * np.var(X, axis=0).max(), np.finfo(float).eps)

        for idx, c in enumerate(self.classes_):
            X_c = X[y == c]
            self.priors_[idx] = X_c.shape[0] / float(n_samples)
            self.means_[idx, :] = np.mean(X_c, axis=0)
            self.vars_[idx, :] = np.var(X_c, axis=0) + epsilon
        return self

    def _joint_log_likelihood(self, X: np.ndarray) -> np.ndarray:
        jll = []
        for idx in range(len(self.classes_)):
            log_prior = np.log(self.priors_[idx])
            mean = self.means_[idx]
            var = self.vars_[idx]
            log_likelihood = -0.5 * np.sum(np.log(2.0 * np.pi * var))
            log_likelihood -= 0.5 * np.sum(((X - mean) ** 2) / var, axis=1)
            jll.append(log_prior + log_likelihood)
        return np.array(jll).T

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        jll = self._joint_log_likelihood(X)
        max_jll = np.max(jll, axis=1, keepdims=True)
        weights = np.exp(jll - max_jll)
        return weights / weights.sum(axis=1, keepdims=True)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.classes_[np.argmax(self._joint_log_likelihood(X), axis=1)]

MIN_TRAIN = 30


def valid_reference_dates(station_id, filepath=DATA_PATH):
    df = load_clean_data(filepath)
    features = build_features(station_data(df, station_id))
    labeled = features.dropna(subset=["target_manana"])
    if len(labeled) < MIN_TRAIN or labeled["target_manana"].nunique() < 2:
        return []
    min_rows = labeled["fecha_objetivo"].iloc[MIN_TRAIN - 1]
    min_classes = labeled.loc[labeled["target_manana"] != labeled["target_manana"].iloc[0], "fecha_objetivo"].iloc[0]
    valid = features["fecha"].between(max(min_rows, min_classes), df["fecha"].max() - pd.Timedelta(days=1))
    return features.loc[valid, "fecha"].dt.strftime("%Y-%m-%d").tolist()


def predict_tomorrow(station_id, reference_date=None, filepath=DATA_PATH):
    """Ajusta por estación con etiquetas conocidas al cierre de la referencia."""
    df = load_clean_data(filepath)
    features = build_features(station_data(df, station_id))
    if reference_date is None:
        fechas = valid_reference_dates(station_id, filepath)
        if not fechas:
            raise ValueError("No hay fechas con suficiente historial para esta estación.")
        reference_date = fechas[-1]
    ref = parse_date(reference_date)
    target = ref + pd.Timedelta(days=1)
    if ref < df["fecha"].min() or target > df["fecha"].max():
        raise ValueError("La referencia y el día siguiente deben pertenecer al período histórico de 2020.")
    row = features.loc[features["fecha"] == ref]
    if row.empty:
        raise ValueError("Se requieren mediciones en la fecha elegida y en los dos días calendario previos.")
    train = features.loc[features["fecha_objetivo"] <= ref]
    if len(train) < MIN_TRAIN or train["target_manana"].nunique() < 2:
        raise ValueError("Se requieren al menos 30 ejemplos históricos y dos categorías observadas.")
    model = GaussianNaiveBayes().fit(train[FEATURE_COLS].to_numpy(), train["target_manana"].astype(int).to_numpy())
    probs = model.predict_proba(row[FEATURE_COLS].to_numpy())[0]
    pred_id = int(model.classes_[probs.argmax()])
    by_category = {name: 0.0 for name in NOMBRES_CATEGORIAS.values()}
    for category, probability in zip(model.classes_, probs):
        by_category[NOMBRES_CATEGORIAS[int(category)]] = float(probability)
    return {
        "estacion": station_id,
        "fecha_referencia": ref.date().isoformat(),
        "fecha_prediccion": target.date().isoformat(),
        "pm25_referencia": float(row["pm25_hoy"].iloc[0]),
        "categoria_predicha_id": pred_id,
        "categoria_predicha": NOMBRES_CATEGORIAS[pred_id],
        "probabilidades": by_category,
        "categorias_entrenadas": [NOMBRES_CATEGORIAS[int(c)] for c in model.classes_],
        "n_entrenamiento": len(train),
        "ultima_etiqueta_entrenamiento": train["fecha_objetivo"].max().date().isoformat(),
        "modelo": "Gaussian Naive Bayes (implementación propia)",
        "alcance": "Predicción experimental sobre datos históricos de 2020.",
    }
