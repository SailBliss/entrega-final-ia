
import sys
from pathlib import Path

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1] / "src")
)

from agent import ask_agent


# ==========================================
# PRUEBAS DEL AGENTE COMPLETO
# ==========================================

pruebas = [
    {
        "nombre": "Consulta documental",
        "pregunta": (
            "Segun la Circular 0031 de 2026, "
            "que recomendaciones de salud existen "
            "durante episodios de humo?"
        )
    },
    {
        "nombre": "Consulta de mediciones",
        "pregunta": (
            "Consulta las mediciones de PM2.5 "
            "de los ultimos dias de la estacion seleccionada."
        )
    },
    {
        "nombre": "Consulta de prediccion",
        "pregunta": (
            "Cual es la prediccion de calidad del aire "
            "para manana en la estacion seleccionada?"
        )
    }
]


ESTACION = "estacion_demo_1"
FECHA = "2026-11-15"


for prueba in pruebas:

    print("\n" + "=" * 60)
    print(prueba["nombre"])
    print("=" * 60)

    resultado = ask_agent(
        question=prueba["pregunta"],
        station_id=ESTACION,
        reference_date=FECHA
    )

    print("\nRESPUESTA:")
    print(resultado["respuesta"])

    print("\nHERRAMIENTAS UTILIZADAS:")
    print(resultado["herramientas_usadas"])

    print("\nFUENTES:")
    for fuente in resultado["fuentes"]:
        print(
            fuente["documento"],
            "- pagina",
            fuente["pagina"]
        )
