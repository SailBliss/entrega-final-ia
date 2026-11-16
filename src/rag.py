
import re
import hashlib
import io
from pathlib import Path

import pymupdf
import chromadb
import pytesseract

from PIL import Image, ImageOps, ImageEnhance, ImageFilter
from sentence_transformers import SentenceTransformer


# ==========================================
# CONFIGURACIÓN
# ==========================================

CARPETA_PDF = Path("data/documentos_ambientales")
RUTA_BD = "data/chroma_db"
NOMBRE_COLECCION = "documentos_oficiales_v2"

MODELO_EMBEDDINGS = "paraphrase-multilingual-MiniLM-L12-v2"

TAMANO_FRAGMENTO = 900
SOLAPAMIENTO = 150

# Ubicación de Tesseract OCR en Windows
pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)


# ==========================================
# CARGAR MODELO DE EMBEDDINGS
# ==========================================

# Cargar el modelo multilingüe para representar textos
def cargar_modelo():

    print("Cargando modelo multilingüe...")

    return SentenceTransformer(MODELO_EMBEDDINGS)


# ==========================================
# CONEXIÓN CON CHROMADB
# ==========================================

# Crear o recuperar la colección de documentos
def obtener_coleccion():

    cliente = chromadb.PersistentClient(path=RUTA_BD)

    return cliente.get_or_create_collection(
        name=NOMBRE_COLECCION,
        metadata={"hnsw:space": "cosine"}
    )


# ==========================================
# EXTRAER TEXTO DE LOS PDF
# ==========================================

# Extraer texto por página y aplicar OCR cuando sea necesario
def leer_pdf(ruta_pdf):

    paginas = []

    with pymupdf.open(ruta_pdf) as documento:

        for numero, pagina in enumerate(documento, start=1):

            texto = pagina.get_text("text").strip()

            # Aplicar OCR si la página no contiene texto extraíble
            if not texto:

                print(
                    f"  Aplicando OCR mejorado a la página {numero}..."
                )

                pixmap = pagina.get_pixmap(
                    matrix=pymupdf.Matrix(3, 3),
                    colorspace=pymupdf.csGRAY,
                    alpha=False
                )

                imagen = Image.open(
                    io.BytesIO(pixmap.tobytes("png"))
                )

                # Mejorar la imagen para facilitar el reconocimiento
                imagen = ImageOps.autocontrast(imagen)
                imagen = ImageEnhance.Contrast(imagen).enhance(1.5)
                imagen = imagen.filter(ImageFilter.SHARPEN)

                texto = pytesseract.image_to_string(
                    imagen,
                    lang="spa+eng",
                    config="--psm 3"
                ).strip()

            # Guardar el texto y el número de página
            if texto:

                paginas.append({
                    "pagina": numero,
                    "texto": texto
                })

            else:

                print(
                    f"  AVISO: Página {numero} sin texto reconocido."
                )

    return paginas


# ==========================================
# DIVIDIR TEXTO EN ORACIONES
# ==========================================

# Separar el texto en oraciones para crear fragmentos coherentes
def dividir_oraciones(texto):

    texto = re.sub(r"\s+", " ", texto.strip())

    oraciones = re.split(
        r'(?<=[.!?])\s+',
        texto
    )

    return [
        oracion.strip()
        for oracion in oraciones
        if oracion.strip()
    ]


# ==========================================
# CREAR FRAGMENTOS
# ==========================================

# Dividir el texto conservando parte del contexto anterior
def dividir_texto(texto):

    oraciones = dividir_oraciones(texto)

    fragmentos = []
    actual = []

    # Calcular la longitud de un grupo de oraciones
    def longitud(lista):

        return len(" ".join(lista))

    # Guardar un fragmento de texto
    def agregar_fragmento(lista):

        contenido = " ".join(lista).strip()

        if contenido:
            fragmentos.append(contenido)

    for oracion in oraciones:

        # Dividir oraciones demasiado largas
        if len(oracion) > TAMANO_FRAGMENTO:

            if actual:

                agregar_fragmento(actual)
                actual = []

            inicio = 0

            while inicio < len(oracion):

                fin = inicio + TAMANO_FRAGMENTO

                fragmentos.append(
                    oracion[inicio:fin].strip()
                )

                inicio = fin - SOLAPAMIENTO

            continue

        propuesta = actual + [oracion]

        if longitud(propuesta) <= TAMANO_FRAGMENTO:

            actual = propuesta

        else:

            anterior = actual.copy()

            agregar_fragmento(actual)

            # Recuperar oraciones completas como contexto
            actual = []

            for frase in reversed(anterior):

                propuesta_contexto = [frase] + actual

                if longitud(propuesta_contexto) <= SOLAPAMIENTO:

                    actual = propuesta_contexto

                else:

                    break

            # Evitar superar el tamaño máximo
            while actual and (
                longitud(actual + [oracion]) > TAMANO_FRAGMENTO
            ):

                actual.pop(0)

            actual.append(oracion)

    if actual:

        agregar_fragmento(actual)

    return fragmentos


# ==========================================
# IDENTIFICADORES DE FRAGMENTOS
# ==========================================

# Crear identificadores únicos para los fragmentos
def crear_id(fuente, pagina, numero, texto):

    contenido = f"{fuente}_{pagina}_{numero}_{texto}"

    hash_texto = hashlib.md5(
        contenido.encode("utf-8")
    ).hexdigest()

    return hash_texto


# ==========================================
# GUARDAR INFORMACIÓN EN CHROMADB
# ==========================================

# Almacenar fragmentos, embeddings y referencias documentales
def guardar_en_chroma(fragmentos, embeddings, fuente, paginas):

    coleccion = obtener_coleccion()

    ids = []
    metadatos = []

    for i, texto in enumerate(fragmentos):

        ids.append(
            crear_id(fuente, paginas[i], i, texto)
        )

        metadatos.append({
            "fuente": Path(fuente).name,
            "pagina": paginas[i]
        })

    coleccion.upsert(
        ids=ids,
        documents=fragmentos,
        embeddings=embeddings.tolist(),
        metadatas=metadatos
    )


# ==========================================
# PROCESAR DOCUMENTOS PDF
# ==========================================

# Extraer, fragmentar e indexar todos los documentos
def cargar_documentos(modelo):

    coleccion = obtener_coleccion()

    # Limpiar la colección antes de crear el índice
    coleccion.delete(
        where={"fuente": {"$ne": ""}}
    )

    total = 0

    archivos = sorted(
        CARPETA_PDF.glob("*.pdf")
    )

    if not archivos:

        print("No se encontraron documentos PDF.")

        return 0

    for archivo in archivos:

        # Ignorar archivos de prueba
        if archivo.name.startswith("prueba"):

            continue

        print("\nProcesando:", archivo.name)

        paginas = leer_pdf(archivo)

        if not paginas:

            print("AVISO: No se pudo extraer texto.")

            continue

        fragmentos = []
        numeros_paginas = []

        for pagina in paginas:

            partes = dividir_texto(
                pagina["texto"]
            )

            fragmentos.extend(partes)

            numeros_paginas.extend(
                [pagina["pagina"]] * len(partes)
            )

        if not fragmentos:

            print("AVISO: No se generaron fragmentos.")

            continue

        # Convertir los fragmentos en embeddings
        embeddings = modelo.encode(
            fragmentos,
            show_progress_bar=False,
            normalize_embeddings=True
        )

        # Guardar los fragmentos en ChromaDB
        guardar_en_chroma(
            fragmentos,
            embeddings,
            archivo.name,
            numeros_paginas
        )

        total += len(fragmentos)

        print("Páginas con texto:", len(paginas))
        print("Fragmentos:", len(fragmentos))

    return total


# ==========================================
# BÚSQUEDA HÍBRIDA
# ==========================================

# Palabras que no aportan mucho a la búsqueda
PALABRAS_VACIAS = {
    "que", "qué", "como", "cómo", "cual", "cuál",
    "los", "las", "una", "uno", "unos", "unas",
    "del", "con", "para", "por", "durante", "entre",
    "sobre", "frente", "ante", "este", "esta",
    "deben", "pueden", "personas", "de", "la", "el",
    "y", "o", "en", "a", "un", "al", "se", "su"
}


# Identificar las palabras importantes de una pregunta
def extraer_palabras(texto):

    palabras = re.findall(
        r"\b[a-záéíóúüñ0-9]{3,}\b",
        texto.lower()
    )

    return {
        palabra
        for palabra in palabras
        if palabra not in PALABRAS_VACIAS
    }


# Calcular coincidencias entre la pregunta y un fragmento
def calcular_coincidencias(pregunta, texto):

    palabras_pregunta = extraer_palabras(pregunta)
    palabras_texto = extraer_palabras(texto)

    if not palabras_pregunta:

        return 0

    coincidencias = palabras_pregunta.intersection(
        palabras_texto
    )

    return len(coincidencias) / len(palabras_pregunta)


# Combinar búsqueda semántica y búsqueda por palabras clave
def buscar_documentos(pregunta, modelo, cantidad=5):

    coleccion = obtener_coleccion()

    total = coleccion.count()

    if total == 0:

        return []

    # Buscar fragmentos mediante significado
    embedding = modelo.encode(
        [pregunta],
        normalize_embeddings=True
    )

    candidatos = min(40, total)

    resultados = coleccion.query(
        query_embeddings=embedding.tolist(),
        n_results=candidatos,
        include=[
            "documents",
            "metadatas",
            "distances"
        ]
    )

    # Organizar los resultados semánticos
    documentos_semanticos = {}

    for posicion, (identificador, texto, metadata, distancia) in enumerate(
        zip(
            resultados["ids"][0],
            resultados["documents"][0],
            resultados["metadatas"][0],
            resultados["distances"][0]
        ),
        start=1
    ):

        documentos_semanticos[identificador] = {
            "texto": texto,
            "fuente": metadata["fuente"],
            "pagina": metadata["pagina"],
            "distancia": float(distancia),
            "posicion": posicion
        }

    # Recuperar los fragmentos para buscar palabras clave
    todos = coleccion.get(
        include=["documents", "metadatas"]
    )

    documentos_clave = []

    for identificador, texto, metadata in zip(
        todos["ids"],
        todos["documents"],
        todos["metadatas"]
    ):

        puntuacion = calcular_coincidencias(
            pregunta,
            texto
        )

        if puntuacion > 0:

            documentos_clave.append({
                "id": identificador,
                "texto": texto,
                "fuente": metadata["fuente"],
                "pagina": metadata["pagina"],
                "puntuacion": puntuacion
            })

    # Ordenar los resultados por coincidencia de palabras
    documentos_clave.sort(
        key=lambda x: x["puntuacion"],
        reverse=True
    )

    # Combinar las posiciones de ambas búsquedas con RRF
    puntuaciones = {}

    for posicion, identificador in enumerate(
        documentos_semanticos,
        start=1
    ):

        puntuaciones[identificador] = (
            puntuaciones.get(identificador, 0)
            + 1 / (60 + posicion)
        )

    for posicion, documento in enumerate(
        documentos_clave,
        start=1
    ):

        identificador = documento["id"]

        puntuaciones[identificador] = (
            puntuaciones.get(identificador, 0)
            + 1 / (60 + posicion)
        )

    # Seleccionar los fragmentos con mejor puntuación combinada
    mejores_ids = sorted(
        puntuaciones,
        key=lambda x: puntuaciones[x],
        reverse=True
    )

    documentos_finales = []

    for identificador in mejores_ids[:cantidad]:

        if identificador in documentos_semanticos:

            documento = documentos_semanticos[identificador]

        else:

            documento_clave = next(
                d for d in documentos_clave
                if d["id"] == identificador
            )

            documento = {
                "texto": documento_clave["texto"],
                "fuente": documento_clave["fuente"],
                "pagina": documento_clave["pagina"],
                "distancia": 1.0
            }

        documento["puntuacion_hibrida"] = (
            puntuaciones[identificador]
        )

        documentos_finales.append(documento)

    return documentos_finales


# ==========================================
# PREGUNTAS DE EVALUACIÓN INICIAL
# ==========================================

PREGUNTAS = [

    "¿Qué enfermedades y eventos de salud están priorizados en la vigilancia de los efectos de la contaminación del aire?",

    "¿Qué grupos de edad son considerados vulnerables frente a la contaminación atmosférica en Medellín?",

    "¿Qué medidas de promoción, prevención y participación comunitaria establece el protocolo de vigilancia ambiental?",

    "¿Qué significa POECA, cuál es su objetivo y qué medidas establece para enfrentar episodios críticos de contaminación atmosférica?",

    "¿Qué contaminantes atmosféricos analiza el Informe Anual de Calidad del Aire 2025?",

    "¿Cómo se comportaron las concentraciones de PM2.5 en las estaciones de monitoreo del Valle de Aburrá durante 2025?",

    "¿Qué efectos respiratorios y cardiovasculares se relacionan con la exposición a contaminantes atmosféricos?",

    "¿Qué estrategias de gestión de calidad del aire y de ciudad inteligente se plantean para el Valle de Aburrá?"

]


# ==========================================
# EJECUTAR INDEXACIÓN Y EVALUACIÓN
# ==========================================

if __name__ == "__main__":

    modelo = cargar_modelo()

    total = cargar_documentos(modelo)

    print("\n" + "=" * 65)

    print("TOTAL DE FRAGMENTOS INDEXADOS:", total)

    coleccion = obtener_coleccion()

    print("TOTAL EN CHROMADB:", coleccion.count())

    # Probar la búsqueda con las preguntas de evaluación
    for numero, pregunta in enumerate(PREGUNTAS, start=1):

        resultados = buscar_documentos(
            pregunta,
            modelo,
            cantidad=5
        )

        print("\n" + "=" * 65)

        print("PREGUNTA", numero)

        print(pregunta)

        print("\nRESULTADOS:")

        if not resultados:

            print("No se encontraron documentos.")

            continue

        for resultado in resultados:

            print("\nFUENTE:", resultado["fuente"])

            print("PÁGINA:", resultado["pagina"])

            print(
                "DISTANCIA:",
                round(resultado["distancia"], 4)
            )

            print(
                "PUNTUACIÓN HÍBRIDA:",
                round(resultado["puntuacion_hibrida"], 4)
            )

            print("TEXTO:", resultado["texto"])
