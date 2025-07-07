from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from pypdf import PdfReader
import os
from dotenv import load_dotenv
import openai
from openai import OpenAI

# === Cargar configuración de entorno ===
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

# Validación de claves
if not OPENAI_API_KEY or not QDRANT_API_KEY or not QDRANT_URL:
    raise Exception("Faltan variables de entorno requeridas: OPENAI_API_KEY, QDRANT_API_KEY, QDRANT_URL")

# === Configurar cliente de OpenAI ===
client = OpenAI(api_key=OPENAI_API_KEY)

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
# OpenAI text-embedding-3-small tiene 1536 dimensiones
MODEL_DIM = 1536
EMBEDDING_MODEL = "text-embedding-3-small"

# Crear carpeta si no existe
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# === Inicializar cliente de Qdrant ===
qdrant_client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY
)

def init_qdrant_collection():
    """Recrear la colección de Qdrant con la configuración correcta"""
    qdrant_client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=MODEL_DIM, distance=Distance.COSINE)
    )

def get_openai_embeddings(texts):
    """Obtener embeddings usando la API de OpenAI"""
    try:
        # OpenAI permite hasta 2048 textos por request
        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=texts
        )
        return [embedding.embedding for embedding in response.data]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al generar embeddings: {str(e)}")

def process_pdf_to_chunks(file_path: str, chunk_size: int = CHUNK_SIZE):
    """Procesar PDF y generar chunks con sus embeddings"""
    reader = PdfReader(file_path)
    full_text = ""
    
    # Extraer texto de todas las páginas
    for i in range(len(reader.pages)):
        page = reader.pages[i]
        if page.extract_text():
            full_text += page.extract_text()
    
    # Dividir en chunks
    chunks = [full_text[i:i + chunk_size] for i in range(0, len(full_text), chunk_size)]
    
    # Filtrar chunks vacíos
    chunks = [chunk.strip() for chunk in chunks if chunk.strip()]
    
    if not chunks:
        raise HTTPException(status_code=400, detail="No se pudo extraer texto del PDF")
    
    # Generar embeddings usando OpenAI
    vectors = get_openai_embeddings(chunks)
    
    return chunks, vectors

@app.get("/")
def read_root():
    return {"status": "ok", "message": "API de veterinaria con OpenAI embeddings"}

@app.post("/upload-doc/")
async def upload_doc(file: UploadFile = File(...)):
    """Subir y procesar documento PDF"""
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="El archivo debe ser un .pdf")

    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    
    # Guardar archivo
    with open(file_path, "wb") as f:
        f.write(await file.read())

    try:
        # Procesar PDF y generar embeddings
        chunks, vectors = process_pdf_to_chunks(file_path)
        
        # Inicializar colección
        init_qdrant_collection()

        # Crear puntos para Qdrant
        points = [
            PointStruct(
                id=i, 
                vector=vectors[i], 
                payload={
                    "text": chunks[i],
                    "document": file.filename
                }
            )
            for i in range(len(chunks))
        ]

        # Insertar en Qdrant
        qdrant_client.upsert(collection_name=COLLECTION_NAME, points=points)

        # Limpiar archivo temporal
        os.remove(file_path)

        return {
            "message": f"{len(points)} fragmentos insertados en Qdrant.",
            "documento": file.filename,
            "chunks_procesados": len(chunks)
        }
        
    except Exception as e:
        # Limpiar archivo en caso de error
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=f"Error procesando documento: {str(e)}")

class ChatRequest(BaseModel):
    query: str

@app.post("/chatbot/")
async def chatbot(request: ChatRequest):
    """Endpoint del chatbot con búsqueda semántica"""
    try:
        query = request.query
        
        # Verificar si la colección existe y tiene las dimensiones correctas
        try:
            collection_info = qdrant_client.get_collection(COLLECTION_NAME)
            if collection_info.config.params.vectors.size != MODEL_DIM:
                raise HTTPException(
                    status_code=400, 
                    detail=f"La colección tiene dimensiones incorrectas ({collection_info.config.params.vectors.size}). "
                           f"Se esperaban {MODEL_DIM}. Por favor, sube un documento nuevamente para recrear la colección."
                )
        except Exception as e:
            if "doesn't exist" in str(e).lower():
                raise HTTPException(
                    status_code=404, 
                    detail="No hay documentos cargados. Por favor, sube un documento PDF primero."
                )
            raise e
        
        # Generar embedding de la consulta usando OpenAI
        query_embeddings = get_openai_embeddings([query])
        query_vector = query_embeddings[0]
        
        # Buscar en Qdrant
        results = qdrant_client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vector,
            limit=4
        )
        
        # Construir contexto
        context = "\n\n".join([r.payload["text"] for r in results])
        
        # Detectar saludos y preguntas casuales
        casual_greetings = ["hola", "hello", "hi", "buenas", "saludos", "que tal", "como estas"]
        is_casual = any(greeting in query.lower() for greeting in casual_greetings)
        
        # Generar respuesta usando OpenAI
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
        
        # Usar OpenAI para generar la respuesta
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Eres un asistente veterinario especializado y profesional."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1000,
            temperature=0.7
        )
        
        return {"respuesta": response.choices[0].message.content.strip()}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en chatbot: {str(e)}")

@app.get("/health")
def health_check():
    """Endpoint de verificación de salud"""
    return {
        "status": "healthy",
        "embedding_model": EMBEDDING_MODEL,
        "model_dimensions": MODEL_DIM
    }

# Opcional: Endpoint para verificar la colección
@app.get("/collection-info")
def get_collection_info():
    """Obtener información sobre la colección de Qdrant"""
    try:
        info = qdrant_client.get_collection(COLLECTION_NAME)
        return {
            "collection_name": COLLECTION_NAME,
            "vectors_count": info.vectors_count,
            "vector_size": info.config.params.vectors.size,
            "expected_size": MODEL_DIM,
            "status": info.status,
            "dimension_match": info.config.params.vectors.size == MODEL_DIM
        }
    except Exception as e:
        return {"error": f"No se pudo obtener información de la colección: {str(e)}"}

@app.delete("/reset-collection")
def reset_collection():
    """Reiniciar la colección de Qdrant (útil para cambios de modelo)"""
    try:
        init_qdrant_collection()
        return {
            "message": "Colección reiniciada exitosamente",
            "collection_name": COLLECTION_NAME,
            "vector_dimensions": MODEL_DIM
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al reiniciar colección: {str(e)}")