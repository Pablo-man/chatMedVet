from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from pypdf import PdfReader
import os
import os
os.environ["TRANSFORMERS_CACHE"] = "/tmp/transformers"
os.environ["SENTENCE_TRANSFORMERS_HOME"] = "/tmp/transformers"
from dotenv import load_dotenv
import google.generativeai as genai

# === Cargar configuración de entorno ===
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

# Validación de claves
if not GOOGLE_API_KEY or not QDRANT_API_KEY or not QDRANT_URL:
    raise Exception("Faltan variables de entorno requeridas.")

# === Configurar modelo de Gemini ===
genai.configure(api_key=GOOGLE_API_KEY)
modelo = genai.GenerativeModel("gemini-1.5-flash")

# === Inicializar FastAPI ===
app = FastAPI()

# === Habilitar CORS para React (localhost:3000) ===
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Cambiar si React corre en otro origen
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Constantes ===
UPLOAD_FOLDER = "docs_upload"
COLLECTION_NAME = "documentos_qdrant"
CHUNK_SIZE = 500
MODEL_DIM = 384

# Crear carpeta si no existe
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# === Inicializar modelo de embeddings y Qdrant ===
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

qdrant_client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY
)

def init_qdrant_collection():
    qdrant_client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=MODEL_DIM, distance=Distance.COSINE)
    )

def process_pdf_to_chunks(file_path: str, chunk_size: int = CHUNK_SIZE):
    reader = PdfReader(file_path)
    full_text = ""
    for i in range (len(reader.pages)):
        page = reader.pages[i]
        if page.extract_text():
            full_text += page.extract_text()
    chunks = [full_text[i:i + chunk_size] for i in range(0, len(full_text), chunk_size)]
    vectors = model.encode(chunks)
    return chunks, vectors

@app.get("/")
def read_root():
    return {"status": "ok"}


@app.post("/upload-doc/")
async def upload_doc(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="El archivo debe ser un .pdf")

    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    with open(file_path, "wb") as f:
        f.write(await file.read())

    chunks, vectors = process_pdf_to_chunks(file_path)
    init_qdrant_collection()

    points = [
        PointStruct(id=i, vector=vectors[i].tolist(), payload={"text": chunks[i]})
        for i in range(len(chunks))
    ]

    qdrant_client.upsert(collection_name=COLLECTION_NAME, points=points)

    return {
        "message": f"{len(points)} fragmentos insertados en Qdrant.",
        "documento": file.filename
    }

class ChatRequest(BaseModel):
    query: str

@app.post("/chatbot/")
async def chatbot(request: ChatRequest):
    try:
        query = request.query
        query_vector = model.encode([query])[0]
        results = qdrant_client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vector,
            limit=4
        )
        context = "\n\n".join([r.payload["text"] for r in results])
        
        # Detectar saludos y preguntas casuales
        casual_greetings = ["hola", "hello", "hi", "buenas", "saludos", "que tal", "como estas"]
        is_casual = any(greeting in query.lower() for greeting in casual_greetings)
        
        prompt = f"""
Eres un asistente virtual especializado en veterinaria que responde preguntas usando SOLO la información del contexto proporcionado.

INSTRUCCIONES IMPORTANTES:

1. PARA SALUDOS O PREGUNTAS CASUALES (como "hola", "¿cómo estás?", etc.):
   - Responde de manera amigable y cordial
   - Preséntate como asistente de la veterinaria
   - Sugiere temas específicos que pueden preguntar como:
     * Servicios veterinarios disponibles
     * Horarios de atención
     * Cuidado de mascotas
     * Procedimientos médicos
     * Precios y consultas
   - Invita a hacer preguntas específicas

2. PARA PREGUNTAS RELACIONADAS CON VETERINARIA PERO SIN INFORMACIÓN SUFICIENTE:
   - Reconoce que la pregunta es relevante
   - Explica que necesitas más información especializada
   - Proporciona este enlace de WhatsApp para contactar un especialista:
     "Para obtener información más detallada sobre este tema, te recomiendo contactar directamente a nuestros especialistas: https://wa.me/5959749898?text=Hola%2C%20tengo%20una%20consulta%20veterinaria"

3. PARA PREGUNTAS CON INFORMACIÓN DISPONIBLE:
   - Responde con lenguaje natural, claro y profesional
   - Organiza la información de manera fácil de entender
   - Usa listas con viñetas para horarios u información estructurada
   - Evita saltos de línea innecesarios

4. PARA PREGUNTAS COMPLETAMENTE FUERA DEL CONTEXTO VETERINARIO:
   - Responde de manera amigable y comprensiva
   - Explica que te especializas en temas veterinarios
   - Sugiere ejemplos de preguntas relevantes que pueden hacer
   - Mantén un tono positivo y servicial

Contexto disponible:
\"\"\"{context}\"\"\"

Pregunta del usuario: {query}

Respuesta completa y bien formateada:
"""
        
        response = modelo.generate_content(prompt)
        return {"respuesta": response.text.strip()}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
