import json

import httpx
import numpy as np
import pytest
from openai import RateLimitError
from streamlit.testing.v1 import AppTest

from src import agent, tools
from src.data_loader import ROOT, get_pm25, load_clean_data
from src.modelo_clasico import build_features, predict_tomorrow, valid_reference_dates
from src.rag import obtener_coleccion
from tests.evaluacion_agente import valid_page

STATION = "BARBOSA - TORRE SOCIAL"


def test_csv_adaptador():
    df = load_clean_data()
    assert len(df) == 7689
    assert df["estacion"].nunique() == 22
    assert not df.duplicated(["estacion", "fecha"]).any()
    assert not df[["fecha", "pm25", "estacion"]].isna().any().any()
    args = (STATION, "2020-01-01", "2020-01-05")
    raw = get_pm25(*args)
    adapted = tools.get_pm25(*args)
    assert adapted["cantidad"] == len(raw) == 5
    assert np.allclose([r["pm25"] for r in adapted["registros"]], raw["pm25"])
    json.dumps(adapted, allow_nan=False)


@pytest.mark.parametrize("args", [
    ("XYZ", "2020-01-01", "2020-01-05"),
    (STATION, "2020-02-31", "2020-03-05"),
    (STATION, "2020-01-05", "2020-01-01"),
    (STATION, "2020-01-06", "2020-01-06"),
    (STATION, "2026-11-01", "2026-11-06"),
])
def test_filtros_invalidos(args):
    output = tools.get_pm25(*args)
    assert output["estado"] == "error"
    assert output["mensaje"]


def test_prediccion_temporal(tmp_path):
    ref = valid_reference_dates(STATION)[-10]
    result = predict_tomorrow(STATION, ref)
    assert sum(result["probabilidades"].values()) == pytest.approx(1.0)
    assert result["ultima_etiqueta_entrenamiento"] <= ref
    df = load_clean_data()
    future = (df["estacion"] == STATION) & (df["fecha"] > ref)
    df.loc[future, "pm25"] = 500.0
    changed = tmp_path / "changed.csv"
    df.drop(columns="categoria").to_csv(changed, index=False)
    revised = predict_tomorrow(STATION, ref, changed)
    assert revised["probabilidades"] == pytest.approx(result["probabilidades"])
    assert build_features(df[df["estacion"] == STATION]).loc[lambda x: x["fecha"] == "2020-01-07"].empty
    assert tools.predict_tomorrow(STATION, "2026-11-10")["estado"] == "error"


def test_recuperacion_documental(monkeypatch):
    monkeypatch.setenv("RAG_SEMANTICO", "0")
    assert obtener_coleccion().count() == 1095
    resultado = tools.search_docs("límites máximos de PM2.5 en la Resolución 2254 de 2017")
    assert resultado["estado"] == "ok"
    assert all(valid_page(s) for s in resultado["fuentes"])
    assert any("resolucion_2254" in s["documento"] for s in resultado["fuentes"])


def test_429_activa_respaldo(monkeypatch):
    class Provider429:
        def create(self, **kwargs):
            response = httpx.Response(429, request=httpx.Request("POST", "https://example.org/"))
            raise RateLimitError("Límite", response=response, body=None)

    from types import SimpleNamespace
    monkeypatch.setattr(agent, "cliente_groq", SimpleNamespace(chat=SimpleNamespace(completions=Provider429())))
    result = agent.ask_agent("¿Qué estaciones están disponibles?")
    assert result["modo"] == "respaldo"
    assert "429" in result["error_modelo"]
    assert STATION in result["respuesta"]


def test_dashboard(monkeypatch):
    monkeypatch.setattr(agent, "cliente_groq", None)
    monkeypatch.setenv("RAG_SEMANTICO", "0")
    app = AppTest.from_file(str(ROOT / "app.py")).run(timeout=30)
    assert not app.exception
    assert len(app.get("arrow_vega_lite_chart")) == 2
    assert app.dataframe
    next(b for b in app.button if b.label == "Ejecutar predicción").click().run(timeout=30)
    assert not app.exception
    assert len(app.get("arrow_vega_lite_chart")) == 3
    app.button(key="suggest-Mediciones").click().run(timeout=30)
    app.button(key="suggest-POECA").click().run(timeout=30)
    assert not app.exception
    assert len(app.session_state["chat_history"]) == 2
    data, doc = [item[1] for item in app.session_state["chat_history"]]
    assert data["modo"] == "respaldo" and data["herramientas_usadas"] == ["get_pm25"]
    assert doc["modo"] == "respaldo" and doc["fuentes"]
