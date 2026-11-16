"""Evalúa GNB y persistencia en la misma prueba cronológica."""
import json

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from src.data_loader import DATA_PATH, NOMBRES_CATEGORIAS, ROOT, load_clean_data
from src.modelo_clasico import FEATURE_COLS, MIN_TRAIN, GaussianNaiveBayes, build_features


def evaluar_todas_estaciones(filepath=DATA_PATH):
    resultados = []
    for estacion, muestras in load_clean_data(filepath).groupby("estacion"):
        casos = build_features(muestras).dropna(subset=["target_manana"])
        split = int(len(casos) * 0.8)
        train, test = casos.iloc[:split], casos.iloc[split:]
        if len(train) < MIN_TRAIN or test.empty or train["target_manana"].nunique() < 2:
            resultados.append({"estacion": estacion, "estado": "datos insuficientes"})
            continue
        gnb = GaussianNaiveBayes().fit(train[FEATURE_COLS].to_numpy(), train["target_manana"].astype(int).to_numpy())
        y = test["target_manana"].astype(int)
        pred = gnb.predict(test[FEATURE_COLS].to_numpy())
        base = test["categoria"]
        labels = list(NOMBRES_CATEGORIAS)
        resultados.append({
            "estacion": estacion,
            "estado": "evaluada",
            "n_train": len(train),
            "n_test": len(test),
            "inicio_prueba": test["fecha"].iloc[0].date().isoformat(),
            "fin_prueba": test["fecha"].iloc[-1].date().isoformat(),
            "accuracy_gnb": accuracy_score(y, pred),
            "accuracy_base": accuracy_score(y, base),
            "f1_macro_gnb": f1_score(y, pred, labels=labels, average="macro", zero_division=0),
            "f1_macro_base": f1_score(y, base, labels=labels, average="macro", zero_division=0),
        })
    return pd.DataFrame(resultados)


if __name__ == "__main__":
    tabla = evaluar_todas_estaciones()
    validas = tabla[tabla["estado"] == "evaluada"]
    mean = validas[["accuracy_gnb", "accuracy_base", "f1_macro_gnb", "f1_macro_base"]].mean()
    resultado = {
        "estaciones_evaluadas": len(validas),
        "n_prueba": int(validas["n_test"].sum()),
        "promedio_entre_estaciones": mean.to_dict(),
        "por_estacion": json.loads(tabla.to_json(orient="records", force_ascii=False)),
    }
    (ROOT / "tests/resultados_modelo.json").write_text(json.dumps(resultado, indent=2, ensure_ascii=False), encoding="utf-8")
    print(tabla.to_string(index=False))
    print(mean.to_string())
