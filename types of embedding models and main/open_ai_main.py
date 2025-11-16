# main.py

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain.chat_models import ChatOpenAI

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

if not os.path.exists(INDEX_PATH) or not os.path.exists(META_PATH):
    raise FileNotFoundError("FAISS index or metadata not found. Run create_embeddings.py first.")

# ---------------------------
# Load FAISS index and metadata
# ---------------------------
index = faiss.read_index(INDEX_PATH)
with open(META_PATH, "rb") as f:
    docs = pickle.load(f)

# ---------------------------
# Load embedding model (MPNet)
# ---------------------------
embedding_model = SentenceTransformer("all-mpnet-base-v2")

# ---------------------------
# Simple retriever function
# ---------------------------
def retrieve(query, k=3):
    q_emb = embedding_model.encode([query], convert_to_numpy=True)
    q_emb = q_emb / np.linalg.norm(q_emb, axis=1, keepdims=True)
    q_emb = q_emb.astype("float32")
    D, I = index.search(q_emb, k)
    results = [docs[i] for i in I[0]]
    return results

# ---------------------------
# Load LLM
# ---------------------------
chat_model = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

# ---------------------------
# API endpoints
# ---------------------------
@app.post("/ask", response_model=Answer)
def ask_question(query: Query):
    try:
        # Retrieve top-k documents
        results = retrieve(query.question)
        context = "\n\n".join(results)

        # Generate answer using LLM
        rag_answer = chat_model.call_as_llm(
            f"Answer the question based on the context below.\n\nContext:\n{context}\n\nQuestion: {query.question}"
        )

        return Answer(answer=rag_answer)
    except Exception as e:
        return Answer(answer=f"[ERROR] {e}")

@app.post("/chat", response_model=Answer)
def chat_endpoint(query: Query):
    return ask_question(query)

@app.get("/health")
def health_check():
    return {"status": "ok"}
