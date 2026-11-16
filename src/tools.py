"""Adaptadores JSON de los datos, el modelo histórico y el corpus."""
import json
import os

from src import data_loader, modelo_clasico


def get_stations():
    df = data_loader.load_clean_data()
    catalogo = (
        df.groupby(["estacion", "municipio"], as_index=False)["fecha"]
        .agg(desde="min", hasta="max")
        .rename(columns={"estacion": "station_id"})
    )
    return {"estado": "ok", "estaciones": json.loads(catalogo.to_json(orient="records", date_format="iso"))}


def get_pm25(station_id, start_date, end_date):
    try:
        df = data_loader.get_pm25(station_id, start_date, end_date)
    except ValueError as error:
        return {"estado": "error", "mensaje": str(error)}
    if df.empty:
        return {"estado": "error", "mensaje": "No hay mediciones para esta estación en el intervalo elegido."}
    rows = df[["fecha", "pm25"]].copy()
    rows["fecha"] = rows["fecha"].dt.strftime("%Y-%m-%d")
    return {
        "estado": "ok",
        "estacion": station_id,
        "fecha_inicial": start_date,
        "fecha_final": end_date,
        "unidad": "µg/m³",
        "cantidad": len(df),
        "promedio": round(float(df["pm25"].mean()), 2),
        "minimo": float(df["pm25"].min()),
        "maximo": float(df["pm25"].max()),
        "registros": rows.to_dict(orient="records"),
    }


def predict_tomorrow(station_id, reference_date):
    try:
        return {"estado": "ok", **modelo_clasico.predict_tomorrow(station_id, reference_date)}
    except ValueError as error:
        return {"estado": "error", "mensaje": str(error)}


def search_docs(question):
    from src.rag import buscar_documentos, cargar_modelo

    if not question.strip():
        return {"estado": "error", "mensaje": "Escribe una pregunta para buscar en el corpus."}
    semantic = os.getenv("RAG_SEMANTICO") == "1"
    model = cargar_modelo() if semantic else None
    documentos = buscar_documentos(question, model, cantidad=4)
    if not documentos:
        return {"estado": "error", "mensaje": "No se encontraron fragmentos relacionados en el corpus."}
    return {
        "estado": "ok",
        "recuperacion": "híbrida" if semantic else "léxica",
        "fuentes": [
            {"documento": d["fuente"], "pagina": d["pagina"], "contenido": d["texto"]}
            for d in documentos
        ],
    }
