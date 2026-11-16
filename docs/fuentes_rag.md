# Fuentes documentales del sistema RAG

## 1. Descripción general

El sistema de generación aumentada por recuperación (RAG) del agente de calidad del aire del Valle de Aburrá utiliza un conjunto de documentos ambientales, normativos y de salud pública.

Estos documentos constituyen la base documental que permite al agente recuperar información relacionada con la normativa de calidad del aire, los protocolos para enfrentar episodios críticos de contaminación atmosférica y las recomendaciones de protección de la salud.

## 2. Documentos utilizados

| Documento | Contenido y utilidad dentro del RAG |
|---|---|
| Resolución 2254 de 2017 | Norma nacional de calidad del aire. Se utiliza para consultar los niveles máximos permisibles de contaminantes atmosféricos. |
| Acuerdo 16 de 2017 – PIGECA | Documento relacionado con el Plan Integral de Gestión de la Calidad del Aire del Valle de Aburrá. Aporta información sobre estrategias y objetivos de gestión. |
| Acuerdo 04 de 2018 – POECA | Contiene el protocolo metropolitano para enfrentar episodios de contaminación atmosférica. Permite consultar niveles de actuación y medidas de respuesta. |
| Circular 0031 de 2026 | Contiene instrucciones y recomendaciones del sector salud frente a incendios forestales, exposición al humo y deterioro de la calidad del aire. |
| Protocolo de salud ambiental de 2019 | Aporta información sobre vigilancia y seguimiento de eventos relacionados con la salud ambiental y la contaminación atmosférica. |
| Informe de calidad del aire 2025 | Proporciona información técnica sobre el estado de la calidad del aire y el monitoreo atmosférico. |
| Efectos de la contaminación atmosférica en la salud – AMVA 2022 | Aporta información sobre los efectos de la contaminación atmosférica y las poblaciones vulnerables. |
| Calidad del aire, Valle de Aburrá y ciudad inteligente – 2021 | Documento de contexto sobre calidad del aire y gestión ambiental urbana. |
| Plan de acción POECA de CORANTIOQUIA 2026 | Material complementario relacionado con acciones y gestión de episodios de contaminación. |

## 3. Procesamiento documental

Los documentos se almacenan en formato PDF dentro de:

`data/documentos_ambientales/`

El procesamiento documental se realiza mediante las siguientes etapas:

1. Lectura de los archivos PDF utilizando PyMuPDF.
2. Extracción del texto de cada página.
3. Aplicación de reconocimiento óptico de caracteres (OCR) cuando los documentos no contienen texto extraíble.
4. División del contenido en fragmentos (*chunks*) de aproximadamente 900 caracteres, con un solapamiento de 150 caracteres.
5. Generación de representaciones vectoriales mediante el modelo Sentence Transformers `paraphrase-multilingual-MiniLM-L12-v2`.
6. Almacenamiento de los fragmentos y sus metadatos en ChromaDB.

Los metadatos permiten conservar información sobre el documento de origen y el número de página correspondiente.

## 4. Recuperación de información

La función `search_docs()` realiza las consultas documentales del agente.

El sistema combina la búsqueda semántica mediante embeddings con una búsqueda por coincidencia de palabras clave. Posteriormente, integra los resultados para recuperar los fragmentos más relevantes.

La información recuperada incluye:

- Contenido del fragmento.
- Nombre del documento.
- Número de página.

Estos elementos permiten identificar las fuentes utilizadas para elaborar las respuestas del agente.

## 5. Relación con los requisitos del proyecto

El corpus documental se organiza para cubrir los tres ejes solicitados para el agente de calidad del aire:

- **Normativa:** Resolución 2254 de 2017 y documentos de gestión de calidad del aire.
- **Protocolos de episodios críticos:** Acuerdo 04 de 2018 y documentación relacionada con el POECA.
- **Recomendaciones de salud:** Circular 0031 de 2026 y documentos sobre efectos de la contaminación atmosférica.

## 6. Limitaciones

- La calidad de la recuperación depende de la calidad del texto extraído de los PDF.
- Los documentos escaneados requieren OCR, que puede introducir errores de reconocimiento.
- Algunos fragmentos recuperados pueden contener información repetida o poco relacionada con la pregunta.
- La recuperación de un fragmento no garantiza por sí sola que la respuesta final sea correcta.
- Los documentos normativos deben revisarse para verificar su vigencia y posibles modificaciones posteriores.
- Las fuentes deben consultarse en sus documentos originales antes de utilizar la información como referencia oficial.