
import json
import re
import sys
import unicodedata

from pathlib import Path


# ============================================================
# 1. CONFIGURACION DE RUTAS
# ============================================================

RUTA_PROYECTO = Path(__file__).resolve().parents[1]
RUTA_SRC = RUTA_PROYECTO / "src"

sys.path.insert(0, str(RUTA_SRC))

import agent

from agent import ask_agent


# Desactivar Groq para evaluar las herramientas localmente.
# Esto evita consumir la cuota de tokens durante las pruebas.
agent.cliente_groq = None


RUTA_PREGUNTAS = (
    RUTA_PROYECTO / "tests" / "preguntas_agente.json"
)

RUTA_RESULTADOS = (
    RUTA_PROYECTO / "tests" / "resultados_evaluacion.json"
)


# ============================================================
# 2. FUNCIONES DE NORMALIZACION
# ============================================================

def normalizar(texto):

    texto = str(texto).lower()

    texto = unicodedata.normalize("NFD", texto)

    texto = "".join(
        caracter
        for caracter in texto
        if unicodedata.category(caracter) != "Mn"
    )

    texto = texto.replace("_", " ")
    texto = texto.replace("-", " ")

    texto = re.sub(r"\s+", " ", texto)

    return texto.strip()


# ============================================================
# 3. DETECCION DE CITAS DOCUMENTALES
# ============================================================

def revisar_cita(respuesta, fuentes):
    """
    Comprueba que el nombre del documento y su pagina
    aparezcan juntos en un mismo fragmento de respuesta.
    """

    if not respuesta or not fuentes:
        return False

    fragmentos = re.split(
        r"fragmento\s+\d+",
        respuesta,
        flags=re.IGNORECASE
    )

    # Tambien permite respuestas que no tengan encabezados
    # FRAGMENTO 1, FRAGMENTO 2, etc.
    if len(fragmentos) == 1:
        fragmentos = [respuesta]

    for fuente in fuentes:

        documento = fuente.get("documento")
        pagina = fuente.get("pagina")

        if not documento or pagina is None:
            continue

        nombre_documento = normalizar(
            Path(str(documento)).name
        )

        # Eliminar la extension .pdf, incluso si aparece dos veces.
        nombre_documento = re.sub(
            r"(?:\.pdf)+$",
            "",
            nombre_documento
        )

        for fragmento in fragmentos:

            fragmento_normalizado = normalizar(fragmento)

            documento_encontrado = (
                nombre_documento in fragmento_normalizado
            )

            patron_pagina = (
                r"\bp[aá]gina\s*:\s*"
                + re.escape(str(pagina))
                + r"\b"
            )

            pagina_encontrada = re.search(
                patron_pagina,
                fragmento,
                flags=re.IGNORECASE
            )

            if documento_encontrado and pagina_encontrada:
                return True

    return False


# ============================================================
# 4. CARGA DE PREGUNTAS
# ============================================================

def cargar_preguntas():

    if not RUTA_PREGUNTAS.exists():

        raise FileNotFoundError(
            f"No se encontro el archivo: {RUTA_PREGUNTAS}"
        )

    with open(
        RUTA_PREGUNTAS,
        "r",
        encoding="utf-8"
    ) as archivo:

        preguntas = json.load(archivo)

    if isinstance(preguntas, dict):

        preguntas = preguntas.get(
            "preguntas",
            preguntas.get("questions", [])
        )

    if not isinstance(preguntas, list):

        raise ValueError(
            "El archivo JSON debe contener una lista de preguntas."
        )

    return preguntas


# ============================================================
# 5. OBTENER DATOS DE CADA PREGUNTA
# ============================================================

def obtener_dato(pregunta, alternativas, valor_default=None):

    for clave in alternativas:

        if clave in pregunta:
            return pregunta[clave]

    return valor_default


# ============================================================
# 6. EVALUACION DE UNA PREGUNTA
# ============================================================

def evaluar_pregunta(numero, pregunta):

    texto_pregunta = obtener_dato(
        pregunta,
        ["pregunta", "question", "consulta"],
        ""
    )

    herramienta_esperada = obtener_dato(
        pregunta,
        [
            "herramienta_esperada",
            "herramienta",
            "expected_tool"
        ],
        None
    )

    station_id = obtener_dato(
        pregunta,
        ["station_id", "estacion", "estacion_id"],
        "estacion_demo_1"
    )

    reference_date = obtener_dato(
        pregunta,
        ["reference_date", "fecha_referencia"],
        "2026-11-15"
    )

    print("\n" + "=" * 70)
    print(f"PREGUNTA {numero}: {texto_pregunta}")
    print("=" * 70)

    if not texto_pregunta:

        return {
            "numero": numero,
            "pregunta": "",
            "estado": "ERROR_CONFIGURACION",
            "error": "La pregunta esta vacia.",
            "herramienta_esperada": herramienta_esperada
        }

    try:

        resultado = ask_agent(
            question=texto_pregunta,
            station_id=station_id,
            reference_date=reference_date
        )

        respuesta = resultado.get("respuesta", "")

        herramientas_usadas = resultado.get(
            "herramientas_usadas",
            []
        )

        fuentes = resultado.get("fuentes", [])

        proveedor = resultado.get(
            "proveedores_usados",
            []
        )

        respuesta_alternativa = resultado.get(
            "respuesta_alternativa",
            False
        )

        # --------------------------------------------
        # COMPROBAR HERRAMIENTA
        # --------------------------------------------

        herramienta_correcta = (
            herramienta_esperada in herramientas_usadas
            if herramienta_esperada
            else bool(herramientas_usadas)
        )

        # --------------------------------------------
        # COMPROBAR CITAS
        # --------------------------------------------

        cita_detectada = revisar_cita(
            respuesta,
            fuentes
        )

        # --------------------------------------------
        # DETERMINAR ESTADO
        # --------------------------------------------

        if herramienta_correcta:

            if respuesta_alternativa:
                estado = "CORRECTA_RESPALDO_LOCAL"
            else:
                estado = "CORRECTA"

        elif herramientas_usadas:

            estado = "HERRAMIENTA_INCORRECTA"

        else:

            estado = "SIN_HERRAMIENTA"

        print("\nHerramienta esperada:")
        print(herramienta_esperada)

        print("\nHerramientas utilizadas:")
        print(herramientas_usadas)

        print("\nEstado:")
        print(estado)

        print("\nFuentes recuperadas:")
        print(len(fuentes))

        print("\nCita documental detectada:")
        print("SI" if cita_detectada else "NO")

        print("\nRespuesta alternativa:")
        print(respuesta_alternativa)

        print("\nProveedor:")
        print(proveedor)

        print("\nRespuesta:")
        print(respuesta)

        return {
            "numero": numero,
            "pregunta": texto_pregunta,
            "herramienta_esperada": herramienta_esperada,
            "herramientas_usadas": herramientas_usadas,
            "herramienta_correcta": herramienta_correcta,
            "estado": estado,
            "fuentes_recuperadas": len(fuentes),
            "fuentes": fuentes,
            "cita_detectada": cita_detectada,
            "respuesta_alternativa": respuesta_alternativa,
            "proveedores": proveedor,
            "respuesta": respuesta,
            "error_modelo": resultado.get("error_modelo")
        }

    except Exception as error:

        print("\nERROR AL EJECUTAR LA PREGUNTA:")
        print(str(error))

        return {
            "numero": numero,
            "pregunta": texto_pregunta,
            "herramienta_esperada": herramienta_esperada,
            "estado": "ERROR_EJECUCION",
            "error": str(error),
            "herramienta_correcta": False,
            "cita_detectada": False
        }


# ============================================================
# 7. FUNCION PRINCIPAL DE EVALUACION
# ============================================================

def ejecutar_evaluacion():

    print("\n" + "=" * 70)
    print("EVALUACION DEL AGENTE DE CALIDAD DEL AIRE")
    print("=" * 70)

    print("\nProveedor LLM:")
    print("Groq desactivado para esta evaluacion.")
    print("Se utilizara el respaldo local.")

    preguntas = cargar_preguntas()

    print(f"\nPreguntas cargadas: {len(preguntas)}")

    resultados = []

    for numero, pregunta in enumerate(preguntas, start=1):

        resultado = evaluar_pregunta(
            numero,
            pregunta
        )

        resultados.append(resultado)

    # ========================================================
    # 8. CALCULO DE INDICADORES
    # ========================================================

    total = len(resultados)

    intentos = sum(
        1
        for resultado in resultados
        if resultado.get("herramientas_usadas")
    )

    correctas = sum(
        1
        for resultado in resultados
        if resultado.get("herramienta_correcta", False)
    )

    errores = sum(
        1
        for resultado in resultados
        if resultado.get("estado") in [
            "ERROR_EJECUCION",
            "HERRAMIENTA_INCORRECTA",
            "SIN_HERRAMIENTA",
            "ERROR_CONFIGURACION"
        ]
    )

    respuestas_locales = sum(
        1
        for resultado in resultados
        if resultado.get("respuesta_alternativa", False)
    )

    preguntas_con_fuentes = sum(
        1
        for resultado in resultados
        if resultado.get("fuentes_recuperadas", 0) > 0
    )

    citas_detectadas = sum(
        1
        for resultado in resultados
        if resultado.get("cita_detectada", False)
    )

    # Porcentajes
    porcentaje_herramientas = (
        correctas / total * 100
        if total else 0
    )

    porcentaje_citas = (
        citas_detectadas / preguntas_con_fuentes * 100
        if preguntas_con_fuentes else 0
    )

    porcentaje_exito = (
        correctas / total * 100
        if total else 0
    )

    # ========================================================
    # 9. RESUMEN FINAL
    # ========================================================

    resumen = {

        "preguntas_totales": total,

        "preguntas_con_intento_herramienta": intentos,

        "herramientas_seleccionadas_correctamente": correctas,

        "porcentaje_seleccion_correcta": round(
            porcentaje_herramientas,
            2
        ),

        "porcentaje_preguntas_correctas": round(
            porcentaje_exito,
            2
        ),

        "errores_ejecucion_o_seleccion": errores,

        "respuestas_mediante_respaldo_local": respuestas_locales,

        "preguntas_con_fuentes_recuperadas": preguntas_con_fuentes,

        "preguntas_con_cita_detectada": citas_detectadas,

        "porcentaje_deteccion_citas": round(
            porcentaje_citas,
            2
        ),

        "proveedor_llm": "Desactivado durante la evaluacion",

        "observaciones": [
            (
                "La evaluacion se realizo mediante respaldo local "
                "sin utilizar el proveedor Groq."
            ),
            (
                "Los resultados miden la seleccion y ejecucion "
                "de herramientas, no la capacidad de razonamiento "
                "del LLM."
            ),
            (
                "La deteccion automatica de citas comprueba "
                "la presencia del documento y la pagina en "
                "un mismo fragmento de respuesta."
            ),
            (
                "Las citas deben verificarse manualmente "
                "contra los documentos PDF originales."
            ),
            (
                "Las mediciones y predicciones de demostracion "
                "no representan datos reales."
            )
        ]
    }

    # ========================================================
    # 10. GUARDAR RESULTADOS EN JSON
    # ========================================================

    informe = {

        "resumen": resumen,

        "resultados_individuales": resultados

    }

    with open(
        RUTA_RESULTADOS,
        "w",
        encoding="utf-8"
    ) as archivo:

        json.dump(
            informe,
            archivo,
            ensure_ascii=False,
            indent=4
        )

    # ========================================================
    # 11. MOSTRAR RESULTADOS
    # ========================================================

    print("\n" + "=" * 70)
    print("RESULTADOS FINALES DE LA EVALUACION")
    print("=" * 70)

    print(f"Preguntas totales: {total}")

    print(f"Preguntas con intento de herramienta: {intentos}")

    print(
        "Herramientas seleccionadas correctamente: "
        f"{correctas}"
    )

    print(
        "Seleccion correcta: "
        f"{porcentaje_herramientas:.2f}%"
    )

    print(
        "Preguntas correctas sobre el total: "
        f"{porcentaje_exito:.2f}%"
    )

    print(
        f"Errores de ejecucion o seleccion: {errores}"
    )

    print(
        "Respuestas mediante respaldo local: "
        f"{respuestas_locales}"
    )

    print(
        "Preguntas con fuentes recuperadas: "
        f"{preguntas_con_fuentes}"
    )

    print(
        f"Citas documentales detectadas: {citas_detectadas}"
    )

    print(
        f"Porcentaje de deteccion de citas: "
        f"{porcentaje_citas:.2f}%"
    )

    print("\nOBSERVACIONES:")

    for observacion in resumen["observaciones"]:
        print("- " + observacion)

    print("\nResultados guardados en:")
    print(RUTA_RESULTADOS)

    print("\nEVALUACION FINALIZADA.")


# ============================================================
# 12. EJECUCION
# ============================================================

if __name__ == "__main__":

    try:

        ejecutar_evaluacion()

    except Exception as error:

        print("\nNo fue posible completar la evaluacion.")
        print(f"Error: {error}")
