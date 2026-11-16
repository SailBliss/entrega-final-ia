
import os
from dotenv import load_dotenv
from openai import OpenAI

# Leer la clave API del archivo .env
load_dotenv()

# Conectar con Groq
cliente = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

# Enviar una pregunta al modelo
respuesta = cliente.chat.completions.create(
    model="openai/gpt-oss-20b",
    messages=[
        {
            "role": "system",
            "content": "Eres un asistente especializado en calidad del aire."
        },
        {
            "role": "user",
            "content": "¿Qué es el material particulado PM2.5?"
        }
    ]
)

# Mostrar la respuesta
print("\nRESPUESTA DEL MODELO:\n")
print(respuesta.choices[0].message.content)
