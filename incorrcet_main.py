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
LLAMA_PATH = os.path.join(MODEL_FOLDER, "minilama-gguf.gguf")  # MiniLLaMA GGUF

# ---------------------------
# Check required files
# ---------------------------
if not os.path.exists(INDEX_PATH) or not os.path.exists(META_PATH):
    raise FileNotFoundError("FAISS index or metadata not found. Run create_embeddings.py first.")
if not os.path.exists(LLAMA_PATH):
    raise FileNotFoundError("MiniLLaMA GGUF model not found. Place it in the models folder.")

# ---------------------------
# Load FAISS index and documents
# ---------------------------
index = faiss.read_index(INDEX_PATH)
with open(META_PATH, "rb") as f:
    docs = pickle.load(f)
print(f"[INFO] Loaded {len(docs)} documents.")

# ---------------------------
# Load embedding model
# ---------------------------
embedding_model = SentenceTransformer("all-mpnet-base-v2")
print("[INFO] SentenceTransformer model loaded.")

# ---------------------------
# Retriever function
# ---------------------------
def retrieve(query, k=3):
    q_emb = embedding_model.encode([query], convert_to_numpy=True)
    q_emb = q_emb / np.linalg.norm(q_emb, axis=1, keepdims=True)
    q_emb = q_emb.astype("float32")
    D, I = index.search(q_emb, k)
    results = [docs[i] for i in I[0]]
    print(f"[DEBUG] Retrieved {len(results)} context chunks.")
    return results

# ---------------------------
# Load local MiniLLaMA model
# ---------------------------
llm = Llama(model_path=LLAMA_PATH, n_ctx=1024, n_threads=4, temperature=0.0)
print("[INFO] MiniLLaMA model loaded.")

# ---------------------------
# Generate answer function
# ---------------------------
def generate_answer(question, context):
    prompt = f"""
<s>[INST] You are a helpful assistant. Use ONLY the context to answer.

Context:
{context}

Question: {question}

Answer concisely. [/INST]
"""
    output = llm(prompt, max_tokens=512)
    answer = output.get("choices", [{}])[0].get("text", "").strip()
    if not answer:
        print("[WARN] LLM returned empty response.")
        return "[ERROR] Model did not generate an answer."
    print(f"[DEBUG] Generated answer: {answer}")
    return answer

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
        print(f"[ERROR] Exception in /ask: {e}")
        return Answer(answer=f"[ERROR] {e}")

@app.post("/chat", response_model=Answer)
def chat_endpoint(query: Query):
    return ask_question(query)

@app.get("/health")
def health_check():
    return {"status": "ok"}
