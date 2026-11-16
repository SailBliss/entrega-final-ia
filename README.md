# Sistema inteligente de calidad del aire del Valle de Aburrá

Proyecto final de Inteligencia Artificial, SI3003, Universidad EAFIT. Explora concentraciones históricas de PM2.5, estima la categoría del día calendario siguiente a una fecha de 2020 y consulta documentos ambientales mediante un agente con herramientas.

**Alcance histórico.** El período del CSV es 2020. Las predicciones son experimentales. Las respuestas documentales orientan hacia las fuentes del corpus y requieren revisar la vigencia y el contexto de cada documento.

## Puesta en marcha

Se probó con Python 3.12 en Windows. Desde la raíz del repositorio, en PowerShell:

```powershell
git clone https://github.com/SailBliss/sarah.git
cd sarah
git switch feature/integracion-final-ia
py -3.12 -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python -m streamlit run app.py
```

En macOS o Linux, crea el entorno con `python3.12 -m venv .venv` y usa `.venv/bin/python` en los comandos. Abre la URL que imprime Streamlit, normalmente [localhost:8501](http://localhost:8501).

Para habilitar el LLM, copia `.env.example` a `.env` y define `GROQ_API_KEY`. Reinicia el proceso después de cambiarlo. El modelo predeterminado es `openai/gpt-oss-20b`; se puede configurar con `GROQ_MODEL`. Ante ausencia de clave, 429 u otra falla del servicio, el chat muestra un respaldo local etiquetado. Las preguntas con fechas explícitas funcionan mejor en formato AAAA-MM-DD. El chat conserva los turnos durante la sesión y utiliza la estación, el período y la fecha de referencia actuales para cada consulta.

## Demostración

1. Abre **Explorador**, elige **BARBOSA - TORRE SOCIAL** y selecciona del 1 al 5 de enero de 2020. Verifica la serie y la tabla de PM2.5.
2. Abre **Comparación** y cambia el intervalo. Las barras muestran medias de los registros disponibles para cada estación y municipio.
3. En **Predicción**, selecciona el 15 de noviembre de 2020 y pulsa **Ejecutar predicción**. Revisa las probabilidades para el 16 de noviembre.
4. En **Chat**, prueba **Mediciones**, **Predicción** y **POECA**. La interfaz indica el modo de respuesta y las herramientas ejecutadas. Abre **Fuentes documentales** para verificar archivo, página y fragmento.

## Arquitectura y método

- **Streamlit** presenta las series, los promedios, la predicción, las fuentes y el historial de sesión.
- **IA clásica** en `src/modelo_clasico.py`. Gaussian Naive Bayes en NumPy estima una clase diaria por estación. Usa tres días consecutivos de PM2.5, variación diaria, media móvil y mes cíclico. Cada predicción ajusta el modelo con etiquetas conocidas hasta la fecha de referencia. Exige 30 ejemplos y dos categorías como mínimo. La prueba usa un corte cronológico de 80/20 por estación y compara con persistencia de la categoría del día de referencia.
- **Agente** en `src/agent.py`. Hasta tres solicitudes por pregunta. El adaptador `src/tools.py` expone `get_stations`, `get_pm25`, `predict_tomorrow` y `search_docs`. Las consultas amplias envían al proveedor estadísticas del intervalo y una muestra de diez fechas. El explorador presenta la serie completa.
- **Recuperación** en `src/rag.py`. Reutiliza la colección ChromaDB incluida. La instalación base usa coincidencias de texto y conserva los metadatos de archivo y página. El modo híbrido combina coincidencias y similitud semántica de Sentence Transformers.

## Datos y corpus

El ZIP de Persona 1 aporta `data/datos.csv`, con 7.689 filas y 22 estaciones, sin duplicados estación-fecha. El [catálogo oficial AMVA/SIATA](https://datosabiertos.metropol.gov.co/node/99) describe mediciones de su red. El ZIP carece de URL de descarga, versión y script de agregación que permitan atribuirle el CSV exacto o determinar la cobertura horaria de los valores diarios.

Las tres clases académicas usan cortes continuos de 12 y 37 µg/m³, derivados de los dos primeros tramos de la [Resolución 2254 de 2017, tabla 6](https://www.minambiente.gov.co/wp-content/uploads/2021/10/Resolucion-2254-de-2017.pdf). Se agrupan los valores superiores a 37 µg/m³ en una sola clase. El ICA oficial tiene más niveles y su cálculo requiere los procedimientos de la autoridad ambiental. El modelo no establece declaraciones regulatorias.

Los PDFs y la colección de recuperación se incluyen en `data/`. El [inventario documental](docs/fuentes_rag.md) describe los nueve documentos. La página reportada es la posición de la página en el PDF, empezando en uno.

### Recuperación híbrida y reconstrucción

```powershell
.\.venv\Scripts\python -m pip install -r requirements-semantic.txt
$env:RAG_SEMANTICO = "1"
.\.venv\Scripts\python -m streamlit run app.py
```

La primera carga requiere acceso a Hugging Face para descargar `paraphrase-multilingual-MiniLM-L12-v2`. Para reconstruir el índice, instala Tesseract con `spa` y `eng` para las páginas escaneadas. Añádelo al PATH o define `TESSERACT_CMD` como variable de entorno antes de ejecutar:

```powershell
.\.venv\Scripts\python -m src.rag --output data/chroma_rebuilt
$env:RAG_INDEX_DIR = "data/chroma_rebuilt"
.\.venv\Scripts\python -m streamlit run app.py
```

La salida debe ser un directorio nuevo. El índice versionado se conserva y la aplicación no lo reconstruye al arrancar.

## Evaluaciones

```powershell
.\.venv\Scripts\python -m pip install -r requirements-dev.txt
.\.venv\Scripts\python -m compileall -q app.py src tests
.\.venv\Scripts\python -m src.evaluacion_modelo
.\.venv\Scripts\python -m tests.evaluacion_agente
.\.venv\Scripts\python -m tests.evaluacion_agente --llm
.\.venv\Scripts\python -m pytest -q
.\.venv\Scripts\python docs/generar_informe.py
```

Los resultados en `tests/` distinguen el respaldo local de Groq. El archivo `resultados_agente_groq_inicial.json` conserva la primera ejecución, previa al ajuste de coincidencias por identificador normativo. La prueba de 429 inyecta un error de API para verificar el manejo. La evaluación documental comprueba la herramienta, la correspondencia con un documento esperado y la existencia de las páginas citadas. No mide exactitud semántica de cada afirmación. El experimento clásico cubre un año, usa variables correlacionadas y probabilidades sin calibrar. Los cortes de emisión y los huecos reducen las ventanas válidas.

## Estructura y créditos

| Ruta | Contenido |
|---|---|
| `app.py` | Interfaz de exploración, comparación, predicción y chat |
| `src/` | Datos, modelo clásico, agente, herramientas y recuperación |
| `data/` | CSV histórico, PDFs e índice ChromaDB |
| `tests/` | Casos, evaluaciones, pruebas de integración y evidencia |
| `docs/` | Informe, corpus y declaración de IA |

Persona 1 aportó los datos y el clasificador clásico. Persona 2 aportó el agente, el corpus, el índice y las preguntas de evaluación. La integración añadió la aplicación y las verificaciones sobre ambos aportes. El historial inicial identifica al autor del commit base como `SailBliss`. Consulta la [declaración de uso de IA](docs/declaracion_ia.md) y el [informe académico](docs/informe_final.pdf).
