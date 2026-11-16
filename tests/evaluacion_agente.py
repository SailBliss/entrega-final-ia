"""Evalúa las mismas preguntas en modo local o contra Groq."""
import argparse
import json
import re

import pymupdf

from src import agent
from src.data_loader import ROOT
from src.rag import CARPETA_PDF

PREGUNTAS = ROOT / "tests" / "preguntas_agente.json"


def valid_page(source):
    name = source["documento"]
    path = CARPETA_PDF / name
    if path.name != name or not path.is_file():
        return False
    with pymupdf.open(path) as pdf:
        return 1 <= source["pagina"] <= len(pdf)


def evaluar(use_llm=False):
    casos = json.loads(PREGUNTAS.read_text(encoding="utf-8"))
    resultados = []
    bloqueo = "Falta configurar GROQ_API_KEY." if use_llm and agent.cliente_groq is None else None
    for caso in casos:
        esperado = caso["herramienta_esperada"]
        fila = {"id": caso["id"], "pregunta": caso["pregunta"], "herramienta_esperada": esperado}
        if bloqueo:
            resultados.append({**fila, "estado": "no ejecutada", "error": bloqueo})
            continue
        result = agent.ask_agent(
            caso["pregunta"], caso.get("station_id"), caso.get("reference_date"), use_llm=use_llm,
        )
        sources = result["fuentes"]
        citas = [{"documento": s["documento"], "pagina": s["pagina"], "pagina_valida": valid_page(s)} for s in sources]
        if use_llm:
            referencias = re.findall(agent.CITATION_PATTERN, result["respuesta"])
            permitidas = {(s["documento"], s["pagina"]) for s in citas if s["pagina_valida"]}
            referencias_ok = bool(referencias) and all((nombre, int(pagina)) in permitidas for nombre, pagina in referencias)
            documentos_citados = {nombre for nombre, _ in referencias}
        else:
            referencias_ok = all(c["pagina_valida"] for c in citas)
            documentos_citados = {s["documento"] for s in citas}
        cita_correcta = None
        if esperado == "search_docs":
            cita_correcta = bool(citas) and referencias_ok and bool(documentos_citados.intersection(caso["documentos_esperados"]))
        esperado_modo = "llm" if use_llm else "respaldo"
        ejecuciones = [e["resultado"] for e in result["resultados_herramientas"] if e["herramienta"] == esperado]
        correcto = bool(ejecuciones) and result["modo"] == esperado_modo and cita_correcta is not False
        if esperado == "get_pm25" and ejecuciones:
            correcto = correcto and all(
                e.get("estacion") == caso["station_id"] and e.get("fecha_inicial") == caso["start_date"]
                and e.get("fecha_final") == caso["end_date"] and e.get("cantidad", 0) > 0 for e in ejecuciones
            )
        if esperado == "predict_tomorrow" and ejecuciones:
            correcto = correcto and all(
                e.get("estacion") == caso["station_id"] and e.get("fecha_referencia") == caso["reference_date"]
                for e in ejecuciones
            )
        error = result["error_modelo"] if use_llm else None
        estado = "correcta" if correcto else "incorrecta"
        if error and error.startswith("Groq") and result["modo"] != "llm":
            estado = "no ejecutada"
            if any(f"HTTP {n}" in error for n in (401, 403, 429)):
                bloqueo = error
        resultados.append({
            **fila, "estado": estado, "modo_respuesta": result["modo"],
            "herramientas_usadas": result["herramientas_usadas"], "respuesta": result["respuesta"],
            "citas_recuperadas": citas, "cita_correcta": cita_correcta,
            "error": error, "errores_herramientas": result["errores_herramientas"],
        })
    return resultados


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--llm", action="store_true", help="Realiza llamadas a Groq con la clave local.")
    args = parser.parse_args()
    resultados = evaluar(args.llm)
    modo = "groq" if args.llm else "local"
    path = ROOT / "tests" / f"resultados_agente_{modo}.json"
    path.write_text(json.dumps(resultados, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    for estado in ("correcta", "incorrecta", "no ejecutada"):
        print(f"{modo} / {estado}: {sum(r['estado'] == estado for r in resultados)}")
    print(path)


if __name__ == "__main__":
    main()
