from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from pypdf import PdfReader
import os
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

        prompt = f"""
Eres un asistente que responde preguntas usando SOLO la información del contexto proporcionado.

Responde con un lenguaje natural, claro, profesional y bien organizado para que el usuario entienda fácilmente.

Usa formato limpio, evitando saltos de línea innecesarios o fragmentos incompletos. 

Si vas a dar horarios u otra información por días, organízala en listas con viñetas claras.

Si la información está incompleta o falta, responde: "No tengo suficiente información para responder a esa pregunta."

Si la pregunta no está relacionada con el contenido, responde cordialmente:
"La pregunta no está relacionada con el contenido de la organización. Por favor, formula una consulta relacionada con la organización."

Contexto:
\"\"\"{context}\"\"\"

Pregunta: {query}

Respuesta completa y bien formateada:
"""

        response = modelo.generate_content(prompt)
        return {"respuesta": response.text.strip()}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
