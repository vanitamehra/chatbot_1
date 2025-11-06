# backend/main.py

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os, pickle, faiss, numpy as np
from sentence_transformers import SentenceTransformer

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
# Request model
# ---------------------------
class Query(BaseModel):
    question: str

# ---------------------------
# Paths
# ---------------------------
MODEL_FOLDER = "D:/institute/models"
INDEX_PATH = os.path.join(MODEL_FOLDER, "index.faiss")
METADATA_PATH = os.path.join(MODEL_FOLDER, "metadata.pkl")

print(MODEL_FOLDER, INDEX_PATH , METADATA_PATH )

# ---------------------------
# Load FAISS and chunked docs
# ---------------------------
try:
    index = faiss.read_index(INDEX_PATH)
    with open(METADATA_PATH, "rb") as f:
        data = pickle.load(f)
    chunks = data.get("chunks", [])
    chunk_metadata = data.get("chunk_metadata", [])
    print(f"[INFO] Loaded {len(chunks)} chunks from {len(set(chunk_metadata))} documents, FAISS index has {index.ntotal} vectors")
except Exception as e:
    print("[ERROR] Failed to load FAISS/index:", e)
    index = None
    chunks = []
    chunk_metadata = []

# ---------------------------
# Embedding model
# ---------------------------
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# ---------------------------
# FLAN-T5 generator
# ---------------------------
generator_model = None
tokenizer = None

def get_generator_model():
    global generator_model, tokenizer
    if generator_model is None:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        model_name = "google/flan-t5-base"
        print("[INFO] Loading FLAN-T5 base...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        generator_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        print("[INFO] FLAN-T5 loaded")
    return generator_model, tokenizer

def generate_answer(user_question, retrieved_chunks):
    model, tok = get_generator_model()
    context = "\n".join(retrieved_chunks)
    prompt = f"""
You are a helpful assistant. Use ONLY the information below to answer.
If the answer is not in the context, say "Ask about courses or admissions."

Context:
{context}

Question: {user_question}
Answer:
"""
    inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=512)
    outputs = model.generate(**inputs, max_new_tokens=150)
    answer = tok.decode(outputs[0], skip_special_tokens=True)
    return answer

def fallback_answer(user_question):
    model, tok = get_generator_model()
    prompt = f"Answer the following question as a helpful assistant:\nQuestion: {user_question}\nAnswer:"
    inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=512)
    outputs = model.generate(**inputs, max_new_tokens=150)
    return tok.decode(outputs[0], skip_special_tokens=True)

# ---------------------------
# Chat endpoint
# ---------------------------
@app.post("/chat")
async def chat(query: Query):
    user_question = query.question.strip()
    if user_question == "":
        return {"answer": "Please type something!"}

    if index is None or len(chunks) == 0:
        return {"answer": "Backend not ready. Index/chunks missing."}

    # Compute embedding and retrieve top-k chunks
    q_emb = embedding_model.encode([user_question]).astype(np.float32)
    D, I = index.search(q_emb, 7)  # top 7 closest chunks
    retrieved_chunks = [chunks[i] for i in I[0] if i < len(chunks)]

    # Use retrieved chunks if available, else fallback
    if len(retrieved_chunks) == 0:
        answer = fallback_answer(user_question)
    else:
        answer = generate_answer(user_question, retrieved_chunks)

    return {"answer": answer}

# ---------------------------
# Health check
# ---------------------------
@app.get("/health")
def health():
    return {"status": "ok"}
