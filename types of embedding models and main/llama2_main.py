# main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from llama_cpp import Llama  # local GGUF model

# ---------------------------
# FastAPI setup
# ---------------------------
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# ---------------------------
# Request / Response schemas
# ---------------------------
class Query(BaseModel):
    question: str

class Answer(BaseModel):
    answer: str

# ---------------------------
# Paths
# ---------------------------
MODEL_FOLDER = "models"
INDEX_PATH = os.path.join(MODEL_FOLDER, "index.faiss")
META_PATH = os.path.join(MODEL_FOLDER, "metadata.pkl")
LLAMA_PATH = os.path.join(MODEL_FOLDER, "mistral-7b-v0.2-iq3_s-imat.gguf")  # local GGUF model

# ---------------------------
# Check files
# ---------------------------
if not os.path.exists(INDEX_PATH) or not os.path.exists(META_PATH):
    raise FileNotFoundError("FAISS index or metadata not found. Run create_embeddings.py first.")
if not os.path.exists(LLAMA_PATH):
    raise FileNotFoundError("Mistral 7B GGUF model not found. Place it in the models folder.")

# ---------------------------
# Load FAISS index and metadata
# ---------------------------
index = faiss.read_index(INDEX_PATH)
with open(META_PATH, "rb") as f:
    docs = pickle.load(f)

# ---------------------------
# Load embedding model
# ---------------------------
embedding_model = SentenceTransformer("all-mpnet-base-v2")

# ---------------------------
# Retriever function
# ---------------------------
def retrieve(query, k=3):
    q_emb = embedding_model.encode([query], convert_to_numpy=True)
    q_emb = q_emb / np.linalg.norm(q_emb, axis=1, keepdims=True)
    q_emb = q_emb.astype("float32")
    D, I = index.search(q_emb, k)
    results = [docs[i] for i in I[0]]
    return results

# ---------------------------
# Load local Mistral 7B model
# ---------------------------
llm = Llama(model_path=LLAMA_PATH, n_ctx=1024, n_threads=4, temperature=0.0)

# ---------------------------
# Generate answer function
# ---------------------------
def generate_answer(question, context):
    prompt = f"Answer the question based on the context below.\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer:"
    response = llm(prompt, max_tokens=512)
    return response['choices'][0]['text'].strip()

# ---------------------------
# API endpoints
# ---------------------------
@app.post("/ask", response_model=Answer)
def ask_question(query: Query):
    try:
        results = retrieve(query.question)
        if not results:
            return Answer(answer="No relevant context found for this question.")
        context = "\n\n".join(results)
        answer_text = generate_answer(query.question, context)
        return Answer(answer=answer_text)
    except Exception as e:
        return Answer(answer=f"[ERROR] {e}")

@app.post("/chat", response_model=Answer)
def chat_endpoint(query: Query):
    return ask_question(query)

@app.get("/health")
def health_check():
    return {"status": "ok"}
