# main.py

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import numpy as np
import os
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

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
MODEL_FOLDER = os.path.join("models")
INDEX_PATH = os.path.join(MODEL_FOLDER, "index.faiss")
METADATA_PATH = os.path.join(MODEL_FOLDER, "metadata.pkl")

# ---------------------------
# Load FAISS index & metadata
# ---------------------------
try:
    index = faiss.read_index(INDEX_PATH)
    with open(METADATA_PATH, "rb") as f:
        docs = pickle.load(f)
    print(f"[INFO] Loaded {len(docs)} documents, FAISS index has {index.ntotal} vectors")
except Exception as e:
    print("[ERROR] Failed to load FAISS/index:", e)
    index = None
    docs = []

# ---------------------------
# Load embedding model
# ---------------------------
embedding_model = SentenceTransformer('all-mpnet-base-v2')

def get_query_embedding(query: str):
    q_emb = embedding_model.encode([query], convert_to_numpy=True).astype('float32')
    q_emb = q_emb / np.linalg.norm(q_emb, axis=1, keepdims=True)  # normalize
    return q_emb

# ---------------------------
# FLAN-T5 generator
# ---------------------------
generator_model = None
tokenizer = None

def get_generator_model():
    global generator_model, tokenizer
    if generator_model is None:
        model_name = "google/flan-t5-base"
        print("[INFO] Loading FLAN-T5 base model...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        generator_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        print("[INFO] FLAN-T5 loaded")
    return generator_model, tokenizer

# ---------------------------
# Simple RAG retrieval with relevance check
# ---------------------------
def retrieve_chunks(query: str, top_k: int = 5, threshold: float = 0.4):
    """
    Retrieve top-k relevant chunks above similarity threshold.
    """
    if not index or not docs:
        return []

    q_emb = get_query_embedding(query)
    D, I = index.search(q_emb, top_k)  # D = similarity scores

    retrieved_chunks = []
    for score, idx in zip(D[0], I[0]):
        if score >= threshold:
            retrieved_chunks.append(docs[idx])

    return retrieved_chunks

# ---------------------------
# Generate answer from retrieved chunks
# ---------------------------
def generate_answer(query: str, context_chunks: list):
    if not context_chunks:
        return "I can only answer questions related to the courses."

    generator_model, tokenizer = get_generator_model()
    context_text = "\n".join(context_chunks)
    prompt = f"Answer the question based on the context below.\nContext: {context_text}\nQuestion: {query}\nAnswer:"

    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = generator_model.generate(**inputs, max_length=300)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer

# ---------------------------
# API endpoints
# ---------------------------
@app.post("/ask", response_model=Answer)
def ask_question(query: Query):
    retrieved_chunks = retrieve_chunks(query.question, top_k=5)
    answer_text = generate_answer(query.question, retrieved_chunks)
    return Answer(answer=answer_text)

@app.post("/chat", response_model=Answer)
def chat_endpoint(query: Query):
    # Reuse the same logic as /ask
    return ask_question(query)

@app.get("/health")
def health_check():
    return {"status": "ok"}
