# Declaración de uso de IA generativa

- **OpenAI Codex** asistió en la inspección de los materiales, la integración de los módulos existentes, la corrección del modelo temporal, la implementación de Streamlit, la adaptación del agente, la escritura y ejecución de pruebas y la preparación de la documentación y el informe. Las cifras reportadas provienen de los comandos registrados y los resultados versionados.
- **Groq con `openai/gpt-oss-20b`** es el servicio de inferencia del agente. Cuando está disponible, interpreta la pregunta, solicita herramientas y sintetiza una respuesta con los resultados obtenidos. La app identifica el modo de respuesta y deja visibles los documentos recuperados para su contraste.
- **Sentence Transformers** proporciona representaciones vectoriales del corpus y de las preguntas en modo híbrido. El respaldo local usa reglas de selección y resultados estructurados de las herramientas.

Los aportes originales se conservan en el historial de Git y se acreditan por rol. La revisión académica y la comprensión de los métodos corresponden al equipo. Los resultados de ejemplo se limitan al material incluido y la aplicación mantiene las credenciales en variables de entorno.
