import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from src.data_loader import DATA_PATH, get_stations, load_clean_data
from src.modelo_clasico import FEATURE_COLS, GaussianNaiveBayes, build_features

def evaluar_todas_estaciones(filepath: str = DATA_PATH) -> pd.DataFrame:
    df_clean = load_clean_data(filepath)
    resultados = []

    for est in get_stations(filepath):
        df_feat = build_features(df_clean[df_clean["estacion"] == est])
        X = df_feat[FEATURE_COLS].values
        y = df_feat["target_manana"].astype(int).values
        y_hoy = df_feat["categoria"].astype(int).values

        split = int(len(X) * 0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        y_base = y_hoy[split:]

        gnb = GaussianNaiveBayes().fit(X_train, y_train)
        y_pred = gnb.predict(X_test)

        resultados.append({
            "Estación": est,
            "Días Test": len(y_test),
            "Acc Base": round(accuracy_score(y_test, y_base), 4),
            "Acc GNB": round(accuracy_score(y_test, y_pred), 4),
            "F1-Macro Base": round(f1_score(y_test, y_base, average="macro", zero_division=0), 4),
            "F1-Macro GNB": round(f1_score(y_test, y_pred, average="macro", zero_division=0), 4),
        })

    return pd.DataFrame(resultados)

if __name__ == "__main__":
    tabla = evaluar_todas_estaciones()
    print(tabla.to_string(index=False))
    print("\nPromedio global:")
    print(tabla[["Acc Base", "Acc GNB", "F1-Macro Base", "F1-Macro GNB"]].mean().round(4))
