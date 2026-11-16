
import sys
import json
from pathlib import Path


# ==================================================
# 1. CONFIGURACION DE RUTAS
# ==================================================

# Obtener la carpeta principal del proyecto
RUTA_PROYECTO = Path(__file__).resolve().parents[1]

# Agregar src para permitir importar agent y sus modulos
RUTA_SRC = RUTA_PROYECTO / "src"
sys.path.insert(0, str(RUTA_SRC))

from agent import ask_agent
import agent


# ==================================================
# 2. CONFIGURACION DE PRUEBAS
# ==================================================

ESTACION = "estacion_demo_1"
FECHA_REFERENCIA = "2026-11-15"

# Desactivar Groq para probar exclusivamente
# el respaldo local de las herramientas
agent.cliente_groq = None


# ==================================================
# 3. PREGUNTAS DE PRUEBA
# ==================================================

pruebas = [
    {
        "nombre": "CONSULTA DE ESTACIONES",
        "pregunta": "¿Qué estaciones están disponibles?",
        "herramienta_esperada": "get_stations"
    },
    {
        "nombre": "MEDICIONES DE PM2.5",
        "pregunta": (
            "Consulta las mediciones de PM2.5 "
            "entre 2026-11-10 y 2026-11-15."
        ),
        "herramienta_esperada": "get_pm25"
    },
    {
        "nombre": "PREDICCION DE CALIDAD DEL AIRE",
        "pregunta": (
            "¿Cuál es la predicción de calidad del aire "
            "para mañana?"
        ),
        "herramienta_esperada": "predict_tomorrow"
    }
]


# ==================================================
# 4. CONTADORES
# ==================================================

resultados = []

total = len(pruebas)
correctas = 0
incorrectas = 0


# ==================================================
# 5. EJECUCION DE LAS PRUEBAS
# ==================================================

print("=" * 65)
print("PRUEBAS DE HERRAMIENTAS DEL AGENTE")
print("SISTEMA INTELIGENTE DE CALIDAD DEL AIRE")
print("=" * 65)

print("\nProveedor LLM desactivado: se utilizara el respaldo local.")
print("Estacion:", ESTACION)
print("Fecha de referencia:", FECHA_REFERENCIA)


for prueba in pruebas:

    print("\n" + "-" * 65)
    print(prueba["nombre"])
    print("-" * 65)

    try:

        resultado = ask_agent(
            question=prueba["pregunta"],
            station_id=ESTACION,
            reference_date=FECHA_REFERENCIA
        )

        herramientas = resultado.get(
            "herramientas_usadas", []
        )

        respuesta = resultado.get(
            "respuesta", ""
        )

        esperada = prueba["herramienta_esperada"]

        # Comprobar si se utilizo la herramienta correcta
        seleccion_correcta = esperada in herramientas

        if seleccion_correcta:
            estado = "CORRECTA"
            correctas += 1
        else:
            estado = "INCORRECTA"
            incorrectas += 1

        print("\nHerramienta esperada:")
        print(esperada)

        print("\nHerramientas utilizadas:")
        print(herramientas)

        print("\nEstado:")
        print(estado)

        print("\nRespuesta del agente:")
        print(respuesta)

        print("\n¿Se utilizo respuesta alternativa?")
        print(resultado.get("respuesta_alternativa", False))

        resultados.append({
            "prueba": prueba["nombre"],
            "pregunta": prueba["pregunta"],
            "herramienta_esperada": esperada,
            "herramientas_utilizadas": herramientas,
            "seleccion_correcta": seleccion_correcta,
            "respuesta": respuesta,
            "respuesta_alternativa": resultado.get(
                "respuesta_alternativa", False
            ),
            "error_modelo": resultado.get("error_modelo"),
            "estado": estado
        })

    except Exception as error:

        incorrectas += 1

        print("\nERROR DE EJECUCION:")
        print(str(error))

        resultados.append({
            "prueba": prueba["nombre"],
            "pregunta": prueba["pregunta"],
            "herramienta_esperada": prueba["herramienta_esperada"],
            "herramientas_utilizadas": [],
            "seleccion_correcta": False,
            "respuesta": "",
            "respuesta_alternativa": False,
            "error": str(error),
            "estado": "ERROR_EJECUCION"
        })


# ==================================================
# 6. RESULTADOS FINALES
# ==================================================

porcentaje = (
    correctas / total * 100
    if total > 0
    else 0
)

print("\n" + "=" * 65)
print("RESULTADOS FINALES")
print("=" * 65)

print("Total de pruebas:", total)
print("Pruebas correctas:", correctas)
print("Pruebas incorrectas:", incorrectas)

print(f"Porcentaje de exito: {porcentaje:.2f}%")


# ==================================================
# 7. GUARDAR RESULTADOS
# ==================================================

ruta_resultados = (
    RUTA_PROYECTO / "tests" /
    "resultados_pruebas_herramientas.json"
)

with open(
    ruta_resultados,
    "w",
    encoding="utf-8"
) as archivo:

    json.dump(
        resultados,
        archivo,
        ensure_ascii=False,
        indent=4
    )


# ==================================================
# 8. FINALIZACION
# ==================================================

print("\nResultados guardados en:")
print(ruta_resultados)

print("\nPRUEBAS FINALIZADAS.")
