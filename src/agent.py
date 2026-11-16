
import os
import json
import re
import unicodedata

from dotenv import load_dotenv
from openai import OpenAI

from tools import (
    get_stations,
    get_pm25,
    predict_tomorrow,
    search_docs
)


# ==================================================
# 1. CONFIGURACION DE GROQ
# ==================================================

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODELO_GROQ = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")

MAX_ITERACIONES = 3
TIMEOUT_LLM = 45.0

cliente_groq = None

if GROQ_API_KEY:
    cliente_groq = OpenAI(
        api_key=GROQ_API_KEY,
        base_url="https://api.groq.com/openai/v1",
        timeout=TIMEOUT_LLM,
        max_retries=0
    )


# ==================================================
# 2. FUNCION PARA UTILIZAR GROQ
# ==================================================

def solicitar_al_llm(**argumentos):
    """
    Envia solicitudes al modelo LLM mediante Groq.
    """

    if cliente_groq is None:
        mensaje = (
            "No hay un proveedor LLM configurado. "
            "Revisa GROQ_API_KEY en el archivo .env."
        )

        print(mensaje, flush=True)
        return None, None, [mensaje]

    print("\nIntentando conectar con Groq...", flush=True)

    try:
        respuesta = cliente_groq.chat.completions.create(
            model=MODELO_GROQ,
            **argumentos
        )

        print("Conexion exitosa con Groq.", flush=True)

        return respuesta, "Groq", []

    except Exception as error:
        mensaje_error = f"Groq: {str(error)}"

        print(f"\nFallo con Groq: {str(error)}", flush=True)

        return None, None, [mensaje_error]


# ==================================================
# 3. INSTRUCCIONES DEL AGENTE
# ==================================================

INSTRUCCIONES = """
Eres un agente inteligente especializado en calidad
del aire del Valle de Aburra, Colombia.

Tu funcion es consultar documentos ambientales,
mediciones de PM2.5 y predicciones de calidad del aire.

REGLAS:

1. Utiliza search_docs para preguntas sobre normativa,
   protocolos ambientales y recomendaciones de salud.

2. Utiliza get_pm25 para consultar mediciones de PM2.5.

3. Utiliza predict_tomorrow para consultar predicciones
   de calidad del aire.

4. Utiliza get_stations para consultar las estaciones disponibles.

5. No inventes cifras, fechas, documentos ni paginas.

6. Utiliza exclusivamente los resultados de las herramientas
   para responder preguntas sobre mediciones y predicciones.

7. Identifica siempre los resultados simulados.
   Nunca presentes datos simulados como datos reales.

8. Para respuestas documentales, cita el nombre del documento
   y la pagina que respalda la informacion.

9. Si los documentos recuperados no contienen informacion
   suficiente, dilo claramente.

10. No presentes el sistema como una herramienta de diagnostico
    ni como sustituto de recomendaciones medicas.

11. Responde en espanol, de manera clara y organizada.
"""


# ==================================================
# 4. DEFINICION DE HERRAMIENTAS
# ==================================================

herramientas = [

    {
        "type": "function",
        "function": {
            "name": "get_stations",
            "description": "Consulta las estaciones disponibles.",
            "parameters": {
                "type": "object",
                "properties": {},
                "additionalProperties": False
            }
        }
    },

    {
        "type": "function",
        "function": {
            "name": "search_docs",
            "description": (
                "Busca informacion en documentos ambientales "
                "mediante recuperacion RAG."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "Pregunta documental."
                    }
                },
                "required": ["question"],
                "additionalProperties": False
            }
        }
    },

    {
        "type": "function",
        "function": {
            "name": "get_pm25",
            "description": (
                "Consulta mediciones de PM2.5 por estacion y periodo."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "station_id": {
                        "type": "string"
                    },
                    "start_date": {
                        "type": "string",
                        "description": "Fecha inicial YYYY-MM-DD."
                    },
                    "end_date": {
                        "type": "string",
                        "description": "Fecha final YYYY-MM-DD."
                    }
                },
                "required": [
                    "station_id",
                    "start_date",
                    "end_date"
                ],
                "additionalProperties": False
            }
        }
    },

    {
        "type": "function",
        "function": {
            "name": "predict_tomorrow",
            "description": (
                "Consulta la prediccion de calidad del aire "
                "para el dia siguiente."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "station_id": {
                        "type": "string"
                    },
                    "reference_date": {
                        "type": "string",
                        "description": "Fecha de referencia YYYY-MM-DD."
                    }
                },
                "required": [
                    "station_id",
                    "reference_date"
                ],
                "additionalProperties": False
            }
        }
    }

]


# ==================================================
# 5. CLASIFICACION DE PREGUNTAS
# ==================================================

def normalizar_texto(texto):

    texto = texto.lower()

    texto = unicodedata.normalize("NFD", texto)

    return "".join(
        caracter
        for caracter in texto
        if unicodedata.category(caracter) != "Mn"
    )


def seleccionar_herramienta(question):

    pregunta = normalizar_texto(question)

    palabras_estaciones = [
        "estaciones disponibles",
        "que estaciones",
        "estaciones de monitoreo",
        "lista de estaciones"
    ]

    if any(p in pregunta for p in palabras_estaciones):
        return "get_stations"

    palabras_prediccion = [
        "prediccion",
        "pronostico",
        "para manana",
        "dia siguiente"
    ]

    if any(p in pregunta for p in palabras_prediccion):
        return "predict_tomorrow"

    palabras_mediciones = [
        "mediciones",
        "medidos",
        "valores registrados",
        "datos registrados",
        "registros de pm",
        "consulta de pm",
        "historico de pm",
        "consulta las mediciones",
        "historico de pm2.5"
    ]

    if any(p in pregunta for p in palabras_mediciones):
        return "get_pm25"

    palabras_documentales = [
        "resolucion",
        "normativa",
        "norma",
        "pigeca",
        "poeca",
        "protocolo",
        "circular",
        "recomendaciones",
        "salud",
        "humo",
        "enfermedades",
        "asma",
        "epoc",
        "cardiovasculares",
        "efectos",
        "exposicion",
        "grupos vulnerables",
        "contaminacion atmosferica",
        "calidad del aire",
        "limites maximos permisibles",
        "contaminacion del aire"
    ]

    if any(p in pregunta for p in palabras_documentales):
        return "search_docs"

    return None


# ==================================================
# 6. EJECUCION DE HERRAMIENTAS
# ==================================================

def ejecutar_herramienta(
    nombre,
    argumentos,
    station_id=None,
    reference_date=None
):

    if nombre == "get_stations":

        return get_stations()

    elif nombre == "search_docs":

        return search_docs(argumentos["question"])

    elif nombre == "get_pm25":

        estacion = argumentos.get("station_id") or station_id

        if not estacion:
            return {
                "estado": "error",
                "mensaje": "No se proporciono una estacion."
            }

        return get_pm25(
            estacion,
            argumentos["start_date"],
            argumentos["end_date"]
        )

    elif nombre == "predict_tomorrow":

        estacion = argumentos.get("station_id") or station_id

        fecha = (
            argumentos.get("reference_date")
            or reference_date
        )

        if not estacion or not fecha:
            return {
                "estado": "error",
                "mensaje": "Falta la estacion o la fecha."
            }

        return predict_tomorrow(
            estacion,
            fecha
        )

    return {
        "estado": "error",
        "mensaje": f"Herramienta desconocida: {nombre}"
    }


# ==================================================
# 7. RESPALDO LOCAL SIN MODELO LLM
# ==================================================

def ejecutar_respaldo_local(
    question,
    herramienta,
    station_id=None,
    reference_date=None
):
    """
    Ejecuta herramientas sin depender de Groq.
    Reconoce fechas ISO y fechas escritas en español.
    """

    if herramienta == "get_stations":

        argumentos = {}

    elif herramienta == "search_docs":

        argumentos = {
            "question": question
        }

    elif herramienta == "get_pm25":

        texto = normalizar_texto(question)

        # Buscar fechas en formato YYYY-MM-DD
        fechas = re.findall(
            r"\d{4}-\d{2}-\d{2}",
            texto
        )

        # Si no hay dos fechas ISO, buscar fechas en español
        if len(fechas) < 2:

            meses = {
                "enero": "01",
                "febrero": "02",
                "marzo": "03",
                "abril": "04",
                "mayo": "05",
                "junio": "06",
                "julio": "07",
                "agosto": "08",
                "septiembre": "09",
                "setiembre": "09",
                "octubre": "10",
                "noviembre": "11",
                "diciembre": "12"
            }

            patron = (
                r"(?:entre\s+el\s+)?(\d{1,2})"
                r"\s+(?:y|al)\s+(?:el\s+)?(\d{1,2})"
                r"\s+de\s+([a-z]+)\s+de\s+(\d{4})"
            )

            coincidencia = re.search(patron, texto)

            if coincidencia:

                dia_inicio = int(coincidencia.group(1))
                dia_fin = int(coincidencia.group(2))
                mes = meses.get(coincidencia.group(3))
                anio = coincidencia.group(4)

                if mes:

                    fechas = [
                        f"{anio}-{mes}-{dia_inicio:02d}",
                        f"{anio}-{mes}-{dia_fin:02d}"
                    ]

        if len(fechas) < 2:

            raise ValueError(
                "No se encontraron las dos fechas necesarias "
                "para consultar PM2.5."
            )

        if not station_id:

            raise ValueError(
                "No se proporciono una estacion."
            )

        argumentos = {
            "station_id": station_id,
            "start_date": fechas[0],
            "end_date": fechas[1]
        }

    elif herramienta == "predict_tomorrow":

        if not station_id or not reference_date:

            raise ValueError(
                "Falta la estacion o la fecha de referencia."
            )

        argumentos = {
            "station_id": station_id,
            "reference_date": reference_date
        }

    else:

        raise ValueError(
            f"No existe respaldo para la herramienta: {herramienta}"
        )

    return ejecutar_herramienta(
        herramienta,
        argumentos,
        station_id,
        reference_date
    )


# ==================================================
# 8. RESPUESTA ALTERNATIVA SIN MODELO LLM
# ==================================================

def crear_respuesta_respaldo(resultados_herramientas):

    respuesta = ""
    se_encontraron_resultados = False

    for elemento in resultados_herramientas:

        nombre = elemento["herramienta"]
        resultado = elemento["resultado"]

        if nombre == "search_docs":

            fuentes = resultado.get("fuentes", [])

            if fuentes:

                se_encontraron_resultados = True

                respuesta += (
                    "INFORMACION RECUPERADA MEDIANTE RAG\n\n"
                )

                respuesta += (
                    "No fue posible generar una interpretacion "
                    "con el modelo LLM. Se recuperaron estos "
                    "fragmentos documentales:\n\n"
                )

                for i, fuente in enumerate(fuentes, start=1):

                    respuesta += (
                        f"FRAGMENTO {i}\n"
                        f"Documento: {fuente.get('documento', 'No identificado')}\n"
                        f"Pagina: {fuente.get('pagina', 'No identificada')}\n\n"
                        f"Contenido:\n"
                        f"{fuente.get('contenido', '')}\n\n"
                        f"{'-' * 50}\n\n"
                    )

            else:

                respuesta += (
                    "La busqueda RAG no recupero fragmentos "
                    "documentales suficientes.\n"
                )

        else:

            se_encontraron_resultados = True

            respuesta += (
                f"RESULTADO DE {nombre.upper()}\n\n"
                + json.dumps(
                    resultado,
                    ensure_ascii=False,
                    indent=2,
                    default=str
                )
                + "\n\n"
            )

    if not se_encontraron_resultados:

        return (
            "No fue posible obtener resultados de las herramientas "
            "ni generar una respuesta con el modelo."
        )

    respuesta += (
        "\nNOTA: Esta respuesta alternativa se construyo "
        "directamente a partir de los resultados de las herramientas. "
        "No fue generada ni interpretada por el modelo LLM. "
        "Los fragmentos documentales deben verificarse "
        "en sus documentos originales. "
        "Los datos simulados no representan mediciones reales."
    )

    return respuesta


# ==================================================
# 9. FUNCION PRINCIPAL DEL AGENTE
# ==================================================

def ask_agent(
    question,
    station_id=None,
    reference_date=None
):

    herramienta_obligatoria = seleccionar_herramienta(question)

    contexto = (
        f"Estacion seleccionada: {station_id}\n"
        f"Fecha de referencia: {reference_date}\n"
        "Utiliza estos valores cuando sean necesarios."
    )

    mensajes = [
        {
            "role": "system",
            "content": INSTRUCCIONES + "\n" + contexto
        },
        {
            "role": "user",
            "content": question
        }
    ]

    fuentes = []
    herramientas_usadas = []
    resultados_herramientas = []
    proveedores_usados = []

    respuesta = ""
    error_modelo = None
    respaldo_sin_llm = False

    # --------------------------------------------------
    # CICLO DEL AGENTE
    # --------------------------------------------------

    for iteracion in range(MAX_ITERACIONES):

        if iteracion == 0 and herramienta_obligatoria:

            seleccion = {
                "type": "function",
                "function": {
                    "name": herramienta_obligatoria
                }
            }

        else:

            seleccion = "auto"

        respuesta_modelo, proveedor, errores = solicitar_al_llm(
            messages=mensajes,
            tools=herramientas,
            tool_choice=seleccion
        )

        if proveedor and proveedor not in proveedores_usados:
            proveedores_usados.append(proveedor)

        # ----------------------------------------------
        # RESPALDO CUANDO GROQ FALLA
        # ----------------------------------------------

        if respuesta_modelo is None:

            error_modelo = " | ".join(errores)

            if iteracion == 0 and herramienta_obligatoria:

                print(
                    "\nGroq no esta disponible. "
                    "Activando respaldo local...",
                    flush=True
                )

                try:

                    resultado = ejecutar_respaldo_local(
                        question,
                        herramienta_obligatoria,
                        station_id,
                        reference_date
                    )

                    herramientas_usadas.append(
                        herramienta_obligatoria
                    )

                    resultados_herramientas.append({
                        "herramienta": herramienta_obligatoria,
                        "resultado": resultado
                    })

                    if herramienta_obligatoria == "search_docs":

                        fuentes = resultado.get("fuentes", [])

                    respaldo_sin_llm = True

                    print(
                        "Herramienta ejecutada mediante respaldo local.",
                        flush=True
                    )

                except Exception as error:

                    print(
                        f"Error en el respaldo local: {error}",
                        flush=True
                    )

            break

        mensaje = respuesta_modelo.choices[0].message

        # El modelo puede responder directamente

        if not mensaje.tool_calls:

            respuesta = mensaje.content or ""
            break

        # Registrar llamada del modelo

        mensajes.append({
            "role": "assistant",
            "content": mensaje.content,
            "tool_calls": [
                llamada.model_dump(exclude_none=True)
                for llamada in mensaje.tool_calls
            ]
        })

        # Ejecutar cada herramienta solicitada

        for llamada in mensaje.tool_calls:

            nombre = llamada.function.name

            try:

                argumentos = json.loads(
                    llamada.function.arguments
                )

                resultado = ejecutar_herramienta(
                    nombre,
                    argumentos,
                    station_id,
                    reference_date
                )

                if nombre not in herramientas_usadas:
                    herramientas_usadas.append(nombre)

                if nombre == "search_docs":

                    nuevas_fuentes = resultado.get("fuentes", [])

                    for fuente in nuevas_fuentes:

                        if fuente not in fuentes:
                            fuentes.append(fuente)

            except Exception as error:

                resultado = {
                    "estado": "error",
                    "mensaje": str(error)
                }

            resultados_herramientas.append({
                "herramienta": nombre,
                "resultado": resultado
            })

            mensajes.append({
                "role": "tool",
                "tool_call_id": llamada.id,
                "content": json.dumps(
                    resultado,
                    ensure_ascii=False,
                    default=str
                )
            })

    # ==================================================
    # 10. GENERACION DE RESPUESTA FINAL
    # ==================================================

    if (
        not respuesta
        and resultados_herramientas
        and not respaldo_sin_llm
    ):

        print(
            "\nGenerando respuesta final con los resultados...",
            flush=True
        )

        mensajes_finales = [
            {
                "role": "system",
                "content": (
                    INSTRUCCIONES
                    + "\nGenera una respuesta final utilizando "
                    "los resultados de las herramientas. "
                    "No inventes informacion. "
                    "Cita los documentos y paginas recuperados. "
                    "Identifica los resultados simulados."
                )
            },
            {
                "role": "user",
                "content": question
            },
            {
                "role": "user",
                "content": (
                    "Resultados de las herramientas:\n"
                    + json.dumps(
                        resultados_herramientas,
                        ensure_ascii=False,
                        default=str
                    )
                )
            }
        ]

        respuesta_final, proveedor, errores = solicitar_al_llm(
            messages=mensajes_finales
        )

        if proveedor and proveedor not in proveedores_usados:
            proveedores_usados.append(proveedor)

        if respuesta_final is not None:

            respuesta = (
                respuesta_final.choices[0].message.content or ""
            )

        else:

            error_modelo = " | ".join(errores)

    # ==================================================
    # 11. RESPUESTA ALTERNATIVA
    # ==================================================

    if not respuesta and resultados_herramientas:

        print(
            "\nUtilizando respuesta alternativa sin LLM...",
            flush=True
        )

        respuesta = crear_respuesta_respaldo(
            resultados_herramientas
        )

    elif not respuesta and error_modelo:

        respuesta = (
            "No fue posible obtener una respuesta de Groq. "
            "Error: "
            + error_modelo
        )

    elif not respuesta:

        respuesta = (
            "No se obtuvieron resultados para responder la pregunta."
        )

    # ==================================================
    # 12. RESULTADO FINAL
    # ==================================================

    return {
        "respuesta": respuesta,
        "fuentes": fuentes,
        "herramientas_usadas": herramientas_usadas,
        "error_modelo": error_modelo,
        "proveedores_usados": proveedores_usados,
        "respuesta_alternativa": (
            error_modelo is not None
            and len(resultados_herramientas) > 0
        )
    }


# ==================================================
# 13. PRUEBA DEL AGENTE
# ==================================================

if __name__ == "__main__":

    print("=" * 55)
    print("AGENTE INTELIGENTE DE CALIDAD DEL AIRE")
    print("=" * 55)

    print("\nProveedor configurado:")

    if cliente_groq:
        print(
            "- Groq | Modelo:",
            MODELO_GROQ
        )
    else:
        print("No hay claves API configuradas.")

    resultado = ask_agent(
        question=(
            "Segun la Circular 0031 de 2026, "
            "que recomendaciones de salud se deben seguir "
            "durante episodios de humo y que grupos "
            "poblacionales requieren especial atencion?"
        ),
        station_id="estacion_demo_1",
        reference_date="2026-11-15"
    )

    print("\n" + "=" * 55)
    print("RESPUESTA DEL AGENTE")
    print("=" * 55)

    print(resultado["respuesta"])

    print("\nHERRAMIENTAS UTILIZADAS:")
    print(resultado["herramientas_usadas"])

    print("\nFUENTES CONSULTADAS:")

    for fuente in resultado["fuentes"]:

        print(
            f"\nDocumento: {fuente.get('documento')}"
            f"\nPagina: {fuente.get('pagina')}"
        )

    print("\nPROVEEDOR UTILIZADO:")
    print(resultado["proveedores_usados"])

    print("\n¿Se utilizo respuesta alternativa?")
    print(resultado["respuesta_alternativa"])

    if resultado["error_modelo"]:

        print("\nERROR DEL MODELO:")
        print(resultado["error_modelo"])
