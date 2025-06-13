from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from dotenv import load_dotenv
import google.generativeai as genai
from docx import Document
import traceback 

# Cargar archivo .env con las credenciales de la API
load_dotenv()

# Configurar FastAPI
app = FastAPI()

# Permitir que React se comunique con FastAPI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permitir cualquier origen, puedes poner tu dominio específico
    allow_credentials=True,
    allow_methods=["*"],  # Permitir todos los métodos HTTP
    allow_headers=["*"],  # Permitir todos los encabezados
)

# Configurar Google Generative AI (Gemini)
GOOGLE_API_KEY = "AIzaSyCyk08TLRfsamf8ePvW0fElhFfy8z2NNIU"
genai.configure(api_key=GOOGLE_API_KEY)
modelo = genai.GenerativeModel("gemini-2.0-flash-exp")
print("GOOGLE_API_KEY:", GOOGLE_API_KEY)

# Ruta al documento Word
ruta_docx = "./preguntas_frecuentes.docx"

def leer_docx(ruta):
    doc = Document(ruta)
    texto = "\n".join([p.text for p in doc.paragraphs])
    return texto

contenido_docx = leer_docx(ruta_docx)

# Modelo de solicitud para recibir preguntas
class ChatRequest(BaseModel):
    question: str

# Modelo para calificación de respuesta
class RatingRequest(BaseModel):
    message_index: int
    rating: str
    previous_response: str

# Ruta para la página principal (puedes crear un HTML si lo deseas)
@app.get("/")
async def home():
    return {"message": "Bienvenido a la API del chatbot"}

# Endpoint para el chatbot que utiliza el contenido del documento Word
@app.post("/chatbot")
async def chatbot_con_docx(request: ChatRequest):
    try:
        pregunta = request.question

        if not pregunta:
            raise HTTPException(status_code=400, detail="No se recibió ninguna pregunta")

        prompt = f"""
        Basándote únicamente en el siguiente contenido del documento Word, responde de forma clara y útil:

        \"\"\"{contenido_docx}\"\"\" 

        Pregunta: {pregunta}
        Respuesta:
        """
        respuesta = modelo.generate_content(prompt).text

        return {"respuesta": respuesta}

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")

# Endpoint para manejar las calificaciones y ajustar la respuesta
@app.post("/rate_response")
async def rate_response(request: RatingRequest):
    try:
        message_index = request.message_index  # Índice del mensaje
        rating = request.rating  # Calificación ("Alto", "Medio", "Bajo")
        previous_response = request.previous_response  # Respuesta anterior que el usuario calificó

        if rating not in ["Alto", "Medio", "Bajo"]:
            raise HTTPException(status_code=400, detail="Calificación inválida. Las opciones son 'Alto', 'Medio' o 'Bajo'.")

        # Generar la nueva respuesta basada en la calificación
        if rating == "Alto":
            respuesta_ajustada = "Respuesta perfecta. No hay cambios necesarios."
        elif rating == "Medio":
            respuesta_ajustada = "La respuesta es buena, pero necesita algunos ajustes para ser más precisa."
        else:
            respuesta_ajustada = "La respuesta no es útil. Necesitamos mejorar la propuesta."

        # Crear un prompt para mejorar la respuesta según la calificación y la respuesta anterior
        prompt_nueva_respuesta = f"""
        Basándote únicamente en el siguiente contenido del documento Word, ajusta la siguiente respuesta en función de la calificación dada:
        
        Calificación: {rating}
        Respuesta anterior: {previous_response}
        
        \"\"\"{contenido_docx}\"\"\" 
        
        Mejora la respuesta anterior según la calificación recibida:
        Respuesta ajustada:
        """

        nueva_respuesta = modelo.generate_content(prompt_nueva_respuesta).text

        # Devolver la respuesta ajustada junto con el mensaje original
        return {"respuesta": nueva_respuesta, "calificacion": respuesta_ajustada}

    except Exception as e:
        raise HTTPException(status_code=500, detail="Ocurrió un error al procesar la calificación.")