"""Genera el informe académico a partir de los resultados versionados."""
import json
from collections import Counter
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


DOCS = Path(__file__).resolve().parent
ROOT = DOCS.parent
NAVY = colors.HexColor("#143536")
TEAL = colors.HexColor("#00665E")
SOFT = colors.HexColor("#EAF3F1")
GRAY = colors.HexColor("#496160")


def data(filename):
    return json.loads((ROOT / "tests" / filename).read_text(encoding="utf-8"))


def num(value, decimals=3):
    return f"{value:.{decimals}f}".replace(".", ",")


def footer(canvas, doc):
    width, _ = doc.pagesize
    canvas.setStrokeColor(TEAL)
    canvas.line(doc.leftMargin, 48, width - doc.rightMargin, 48)
    canvas.setFillColor(GRAY)
    canvas.setFont("Helvetica", 8)
    canvas.drawString(doc.leftMargin, 32, "Universidad EAFIT  /  SI3003 Inteligencia Artificial")
    canvas.drawRightString(width - doc.rightMargin, 32, str(canvas.getPageNumber()))


def generar():
    model = data("resultados_modelo.json")
    score = model["promedio_entre_estaciones"]
    local_results = data("resultados_agente_local.json")
    groq_results = data("resultados_agente_groq.json")
    local = Counter(r["estado"] for r in local_results)
    groq = Counter(r["estado"] for r in groq_results)
    issues = sorted({r["error"] for r in groq_results if r.get("error")})
    base = ParagraphStyle("base", fontName="Helvetica", fontSize=10, leading=14.8, textColor=NAVY, spaceAfter=8)
    heading = ParagraphStyle("heading", parent=base, fontName="Helvetica-Bold", fontSize=13, leading=16,
                             textColor=TEAL, spaceBefore=12, spaceAfter=6, keepWithNext=True)
    title = ParagraphStyle("title", parent=base, fontName="Helvetica-Bold", fontSize=27, leading=30,
                           alignment=TA_LEFT, spaceAfter=14)
    caption = ParagraphStyle("caption", parent=base, textColor=GRAY, fontSize=8.5, leading=12)
    tag = ParagraphStyle("tag", parent=caption, fontName="Helvetica-Bold", textColor=TEAL, spaceAfter=10)
    story = []

    def paragraph(value, style=base):
        story.append(Paragraph(value, style))

    def table(cells, widths):
        t = Table(cells, colWidths=widths)
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), TEAL),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
            ("TEXTCOLOR", (0, 1), (-1, -1), NAVY),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [SOFT, colors.white]),
            ("ALIGN", (1, 0), (-1, -1), "CENTER"),
            ("FONTSIZE", (0, 0), (-1, -1), 10),
            ("TOPPADDING", (0, 0), (-1, -1), 8),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ("LINEBELOW", (0, -1), (-1, -1), 0.4, TEAL),
        ]))
        story.extend([t, Spacer(1, 9)])

    paragraph("PROYECTO FINAL  /  UNIVERSIDAD EAFIT", tag)
    paragraph("Sistema inteligente de calidad del aire del Valle de Aburrá", title)
    paragraph("Problema y objetivo", heading)
    paragraph("El PM2.5 suele presentarse como una serie de concentraciones aislada de su contexto ambiental. El sistema permite explorar concentraciones históricas, estimar experimentalmente la categoría para el día calendario siguiente y localizar documentos sobre calidad del aire.")
    paragraph("Datos y procedencia", heading)
    paragraph("El CSV de Persona 1 incluye 7.689 registros en 22 estaciones de 10 municipios, del 1 de enero al 31 de diciembre de 2020. Se verificó la ausencia de duplicados por estación y fecha y de valores faltantes en fecha, estación y PM2.5. El catálogo AMVA/SIATA documenta la red, pero faltan la descarga exacta y el procedimiento que produjo este CSV.")
    paragraph("Clasificación clásica y validación temporal", heading)
    paragraph("Gaussian Naive Bayes se implementó en NumPy. Combina tres mediciones consecutivas, la diferencia diaria, una media móvil de tres días y el mes cíclico. Usa cortes de 12 y 37 µg/m³, inspirados en la tabla 6 de la Resolución 2254 de 2017. La tercera clase agrupa todas las concentraciones mayores de 37 µg/m³. La clasificación es académica y simplifica el ICA oficial.")
    paragraph("La evaluación separó cronológicamente el 80 % inicial y el 20 % final por estación. Solo compara ventanas de cuatro días consecutivos con objetivo observado. La línea base repite la categoría del día de referencia. Con la app, el modelo se ajusta por estación al cierre de la fecha elegida, con un mínimo de 30 ejemplos y dos categorías.")
    paragraph("Resultados cuantitativos", heading)
    table([
        ["Método", "Accuracy", "F1 macro"],
        ["Gaussian Naive Bayes", num(score["accuracy_gnb"] * 100, 2) + " %", num(score["f1_macro_gnb"])],
        ["Persistencia", num(score["accuracy_base"] * 100, 2) + " %", num(score["f1_macro_base"])],
    ], [235, 120, 110])
    pruebas = format(model["n_prueba"], ",").replace(",", ".")
    paragraph(f"Promedio sin ponderar entre {model['estaciones_evaluadas']} estaciones, sobre {pruebas} casos de prueba. El F1 macro incluye las tres clases y asigna cero cuando la métrica de una clase no puede calcularse.", caption)
    paragraph("La persistencia superó al clasificador en ambas métricas. La clase superior a 37 µg/m³ concentra 4.415 registros del CSV. El desbalance, la correlación entre variables, los huecos y la cobertura de un solo año limitan la interpretación. Las probabilidades carecen de calibración.")

    story.append(PageBreak())
    paragraph("INTEGRACIÓN  /  EVALUACIÓN EXPERIMENTAL", tag)
    paragraph("Agente, dashboard y verificación", title)
    paragraph("Arquitectura e interacción", heading)
    paragraph("Streamlit muestra la serie de PM2.5, la comparación de promedios por estación, las probabilidades del clasificador y el chat. El agente usa un ciclo de hasta tres solicitudes a Groq y cuatro herramientas: listar estaciones, consultar mediciones, predecir y recuperar documentos. Los adaptadores serializan mediciones y resultados del modelo en JSON. La interfaz muestra el modo de respuesta, los errores, las herramientas y las fuentes.")
    paragraph("Recuperación documental", heading)
    paragraph("El corpus incluye nueve PDFs sobre normativa, planes ambientales y salud. La colección ChromaDB incluida tiene 1.095 fragmentos con nombre de archivo y página. La búsqueda base por coincidencias de texto funciona localmente. El modo híbrido añade similitud semántica con MiniLM multilingüe. El pipeline conserva extracción por página, OCR para escaneos y fragmentación con solapamiento.")
    paragraph("Evaluación del agente", heading)
    table([
        ["Modalidad", "Correctas", "Incorrectas", "Sin ejecución"],
        ["Respaldo local", local["correcta"], local["incorrecta"], local["no ejecutada"]],
        ["Groq", groq["correcta"], groq["incorrecta"], groq["no ejecutada"]],
    ], [183, 94, 94, 94])
    paragraph("Las diez preguntas incluyen seis consultas documentales, dos de PM2.5, una de predicción y una de estaciones. Se verifica la herramienta, los parámetros de fechas y estación cuando aplican, el documento esperado y la página. El criterio de citas de Groq exige referencias explícitas a los fragmentos recuperados. Esta prueba no califica exhaustivamente el respaldo de cada afirmación.")
    if issues:
        paragraph("Incidencias registradas en Groq. " + escape(" ".join(issues)))
    paragraph("Conclusiones y límites", heading)
    paragraph("La aplicación une consultas históricas, un modelo reproducible y el corpus ambiental en un mismo flujo. Para esta partición, la persistencia es la referencia de mayor desempeño. El experimento requiere verificar la procedencia exacta de los datos y ampliar la validación antes de extrapolar a otros años. El inventario de fuentes y el repositorio conservan los documentos, parámetros y resultados verificables.")
    paragraph('Fuentes de referencia. <link href="https://datosabiertos.metropol.gov.co/node/99" color="#00665E">Catálogo de mediciones AMVA/SIATA</link> y <link href="https://www.minambiente.gov.co/wp-content/uploads/2021/10/Resolucion-2254-de-2017.pdf" color="#00665E">Resolución 2254 de 2017</link>. Materiales del equipo: entrega de Persona 1 y descripción técnica del agente. Uso de IA declarado en docs/declaracion_ia.md.', caption)

    output = DOCS / "informe_final.pdf"
    doc = SimpleDocTemplate(str(output), leftMargin=62, rightMargin=62, topMargin=53, bottomMargin=65,
                            title="Sistema inteligente de calidad del aire del Valle de Aburrá",
                            author="Proyecto académico SI3003 · Universidad EAFIT")
    doc.build(story, onFirstPage=footer, onLaterPages=footer)


if __name__ == "__main__":
    generar()
