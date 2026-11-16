import numpy as np
import pandas as pd
from src.data_loader import DATA_PATH, NOMBRES_CATEGORIAS, load_clean_data

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

    df["target_manana"] = df["categoria"].shift(-1)
    return df.dropna().reset_index(drop=True)

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

        epsilon = self.var_smoothing * np.var(X, axis=0).max()

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
        log_prob_x = max_jll + np.log(
            np.sum(np.exp(jll - max_jll), axis=1, keepdims=True)
        )
        return np.exp(jll - log_prob_x)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.classes_[np.argmax(self._joint_log_likelihood(X), axis=1)]

_MODEL_CACHE = {}

def train_station_model(station_id: str, filepath: str = DATA_PATH):
    df = load_clean_data(filepath)
    df_est = df[df["estacion"] == station_id]
    if df_est.empty:
        raise ValueError(f"Estación '{station_id}' no encontrada.")

    df_feat = build_features(df_est)
    X = df_feat[FEATURE_COLS].values
    y = df_feat["target_manana"].astype(int).values

    split = int(len(X) * 0.8)
    model = GaussianNaiveBayes().fit(X[:split], y[:split])
    _MODEL_CACHE[station_id] = (model, df_feat)
    return model, df_feat

def predict_tomorrow(
    station_id: str,
    reference_date: str = None,
    filepath: str = DATA_PATH,
) -> dict:
    """Devuelve la predicción de categoría ICA y probabilidades para t+1."""
    if station_id not in _MODEL_CACHE:
        train_station_model(station_id, filepath)

    model, df_feat = _MODEL_CACHE[station_id]

    if reference_date is None:
        fila = df_feat.iloc[[-1]]
    else:
        ref_dt = pd.to_datetime(reference_date)
        anteriores = df_feat[df_feat["fecha"] <= ref_dt]
        if anteriores.empty:
            raise ValueError(f"Sin datos previos a {reference_date} para {station_id}.")
        fila = anteriores.iloc[[-1]]

    X_in = fila[FEATURE_COLS].values
    probs = model.predict_proba(X_in)[0]
    pred_id = int(model.predict(X_in)[0])

    prob_dict = {nombre: 0.0 for nombre in NOMBRES_CATEGORIAS.values()}
    for c, p in zip(model.classes_, probs):
        prob_dict[NOMBRES_CATEGORIAS[int(c)]] = round(float(p), 4)

    return {
        "estacion": station_id,
        "fecha_referencia": fila["fecha"].iloc[0].strftime("%Y-%m-%d"),
        "fecha_prediccion": (fila["fecha"].iloc[0] + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
        "pm25_referencia": round(float(fila["pm25_hoy"].iloc[0]), 2),
        "categoria_predicha_id": pred_id,
        "categoria_predicha": NOMBRES_CATEGORIAS[pred_id],
        "probabilidades": prob_dict,
        "modelo": "Gaussian Naive Bayes (Implementación propia)",
    }
