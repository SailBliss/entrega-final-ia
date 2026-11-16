
from datetime import datetime, timedelta
import hashlib
import math

from rag import cargar_modelo, buscar_documentos


# ==========================================
# 1. CONFIGURACION DE DATOS DE DEMOSTRACION
# ==========================================

# Estas estaciones son ejemplos para probar la integracion.
# Se reemplazaran por las estaciones reales de Persona 1.

ESTACIONES_DEMO = {
    "estacion_demo_1": "Estacion de demostracion 1",
    "estacion_demo_2": "Estacion de demostracion 2",
    "estacion_demo_3": "Estacion de demostracion 3"
}


# ==========================================
# 2. CARGA DEL MODELO DE EMBEDDINGS
# ==========================================

modelo = None


def obtener_modelo():

    global modelo

    if modelo is None:
        modelo = cargar_modelo()

    return modelo


# ==========================================
# 3. CONSULTA DE ESTACIONES
# ==========================================

def get_stations():

    return [
        {
            "station_id": station_id,
            "nombre": nombre,
            "simulado": True
        }
        for station_id, nombre in ESTACIONES_DEMO.items()
    ]


# ==========================================
# 4. VALIDACION DE FECHAS
# ==========================================

def validar_fecha(fecha):

    try:
        return datetime.strptime(fecha, "%Y-%m-%d").date()

    except (ValueError, TypeError):
        raise ValueError(
            "La fecha debe tener el formato YYYY-MM-DD."
        )


# ==========================================
# 5. GENERACION DE MEDICIONES SIMULADAS
# ==========================================

def generar_pm25(station_id, fecha):

    # Se genera un valor reproducible para cada estacion y fecha.

    identificador = f"{station_id}_{fecha.isoformat()}"

    semilla = int(
        hashlib.md5(
            identificador.encode()
        ).hexdigest()[:8],
        16
    )

    variacion = (semilla % 1000) / 1000

    valor = (
        12
        + 5 * math.sin(fecha.timetuple().tm_yday / 15)
        + variacion * 4
    )

    return round(max(valor, 1), 2)


# ==========================================
# 6. CONSULTA DE MEDICIONES PM2.5
# ==========================================

def get_pm25(station_id, start_date, end_date):

    if station_id not in ESTACIONES_DEMO:
        return {
            "estado": "error",
            "mensaje": "La estacion solicitada no existe.",
            "station_id": station_id,
            "registros": []
        }

    fecha_inicio = validar_fecha(start_date)
    fecha_fin = validar_fecha(end_date)

    if fecha_inicio > fecha_fin:
        return {
            "estado": "error",
            "mensaje": "La fecha inicial no puede ser posterior a la final.",
            "registros": []
        }

    if (fecha_fin - fecha_inicio).days > 90:
        return {
            "estado": "error",
            "mensaje": "La consulta de demostracion permite maximo 90 dias.",
            "registros": []
        }

    registros = []
    fecha_actual = fecha_inicio

    while fecha_actual <= fecha_fin:

        valor = generar_pm25(station_id, fecha_actual)

        registros.append({
            "station_id": station_id,
            "estacion": ESTACIONES_DEMO[station_id],
            "fecha": fecha_actual.isoformat(),
            "pm25": valor,
            "unidad": "ug/m3",
            "simulado": True
        })

        fecha_actual += timedelta(days=1)

    return {
        "estado": "exitoso",
        "station_id": station_id,
        "registros": registros,
        "simulado": True
    }


# ==========================================
# 7. PREDICCION DE CALIDAD DEL AIRE
# ==========================================

def predict_tomorrow(station_id, reference_date):

    if station_id not in ESTACIONES_DEMO:
        return {
            "estado": "error",
            "mensaje": "La estacion solicitada no existe."
        }

    fecha_referencia = validar_fecha(reference_date)
    fecha_predicha = fecha_referencia + timedelta(days=1)

    # Prediccion fija de demostracion.
    # No corresponde al modelo Gaussian Naive Bayes de Persona 1.

    probabilidades = {
        "Baja": 0.20,
        "Moderada": 0.65,
        "Alta": 0.15
    }

    return {
        "estado": "exitoso",
        "station_id": station_id,
        "estacion": ESTACIONES_DEMO[station_id],
        "fecha_referencia": fecha_referencia.isoformat(),
        "fecha_predicha": fecha_predicha.isoformat(),
        "categoria": "Moderada",
        "probabilidades": probabilidades,
        "simulado": True,
        "advertencia": (
            "Resultado de demostracion. "
            "No corresponde a una prediccion real ni oficial."
        )
    }


# ==========================================
# 8. BUSQUEDA DOCUMENTAL RAG
# ==========================================

def search_docs(question):

    if not isinstance(question, str) or not question.strip():
        return {
            "estado": "error",
            "mensaje": "La pregunta no puede estar vacia.",
            "fuentes": []
        }

    modelo_embeddings = obtener_modelo()
    pregunta_minuscula = question.lower()

    menciona_circular_0031 = (
        "circular 0031" in pregunta_minuscula
        or "circular 031" in pregunta_minuscula
    )

    resultados_totales = []

    # Busqueda especializada para la Circular 0031

    if menciona_circular_0031:

        print("Busqueda especializada: Circular 0031 de 2026.")

        consultas = [
            (
                "Circular 0031 de 2026 seccion 5.4 recomendaciones "
                "ante incendios forestales y exposicion al humo: "
                "reducir exposicion directa, permanecer en espacios "
                "interiores y evitar actividad fisica intensa"
            ),
            (
                "Circular 0031 de 2026 poblaciones que requieren "
                "especial atencion: niños, personas mayores, "
                "mujeres gestantes, enfermedades respiratorias, "
                "cardiovasculares, discapacidad y trabajadores expuestos"
            ),
            (
                "Circular 0031 de 2026 medidas de proteccion de la salud "
                "durante episodios de humo y deterioro de calidad del aire"
            )
        ]

        for consulta in consultas:

            resultados = buscar_documentos(
                consulta,
                modelo_embeddings,
                cantidad=5
            )

            resultados_totales.extend(resultados)

    else:

        print("Busqueda documental general.")

        resultados_totales = buscar_documentos(
            question,
            modelo_embeddings,
            cantidad=5
        )

    # Eliminar duplicados

    resultados_unicos = []
    identificadores = set()

    for resultado in resultados_totales:

        identificador = (
            resultado["fuente"],
            resultado["pagina"],
            resultado["texto"]
        )

        if identificador not in identificadores:

            identificadores.add(identificador)
            resultados_unicos.append(resultado)

    # Organizar fuentes

    if not resultados_unicos:

        return {
            "estado": "sin_resultados",
            "mensaje": "No se encontro informacion documental.",
            "fuentes": []
        }

    fuentes = []

    for resultado in resultados_unicos:

        fuentes.append({
            "contenido": resultado["texto"],
            "documento": resultado["fuente"],
            "pagina": resultado["pagina"]
        })

    return {
        "estado": "exitoso",
        "fuentes": fuentes
    }
