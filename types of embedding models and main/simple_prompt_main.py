# main.py (LangChain 1.0.3 compatible, CPU-only, manual RAG + PDF generator)

from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import faiss
import numpy as np
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import os

# Import dynamic PDF generator from backend folder
from generate_pdf import generate_pdf_from_text


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
# Paths (root models folder)
# ---------------------------
MODEL_FOLDER = os.path.join("models")  # models folder in project root
index_path = os.path.join(MODEL_FOLDER, "index.faiss")
metadata_path = os.path.join(MODEL_FOLDER, "metadata.pkl")

if not os.path.exists(index_path) or not os.path.exists(metadata_path):
    raise FileNotFoundError(
        "FAISS index or metadata not found. Run create_embeddings.py first."
    )

# Load FAISS index + metadata
index = faiss.read_index(index_path)
with open(metadata_path, "rb") as f:
    docs = pickle.load(f)

# ---------------------------
# Embedding model
# ---------------------------
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# ---------------------------
# Helper: retrieve top k docs
# ---------------------------
def retrieve_docs(query_text, k=5):
    query_vector = embedding_model.embed_query(query_text)
    query_vector = np.array(query_vector).astype("float32")
    query_vector = query_vector / np.linalg.norm(query_vector)  # normalize
    D, I = index.search(np.array([query_vector]), k)
    return [docs[i] for i in I[0]]

# ---------------------------
# Load Flan-T5 LLM via HuggingFace pipeline
# ---------------------------
model_name = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
flan_pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer)

# ---------------------------
# Manual RAG function
# ---------------------------
def run_rag(query_text):
    retrieved_texts = retrieve_docs(query_text)
    context = "\n".join(retrieved_texts)
    prompt = f"Answer the question based on the context below:\n\nContext:\n{context}\n\nQuestion: {query_text}\nAnswer:"
    return flan_pipe(prompt, max_length=256, do_sample=False)[0]['generated_text']

# ---------------------------
# API endpoints
# ---------------------------

@app.post("/ask", response_model=Answer)
def ask_question(query: Query):
    """Regular text-based answer."""
    try:
        return Answer(answer=run_rag(query.question))
    except Exception as e:
        return Answer(answer=f"[ERROR] {e}")

@app.post("/chat", response_model=Answer)
def chat_endpoint(query: Query):
    """Alias endpoint."""
    return ask_question(query)

@app.post("/generate_pdf")
def generate_pdf(query: Query):
    """
    Generates a PDF version of the chatbot answer dynamically.
    """
    try:
        answer_text = run_rag(query.question)
        pdf_buffer = generate_pdf_from_text(answer_text, title=query.question)
        headers = {"Content-Disposition": f"attachment; filename=answer.pdf"}
        return Response(
            content=pdf_buffer.getvalue(),
            media_type="application/pdf",
            headers=headers
        )
    except Exception as e:
        return {"error": str(e)}

@app.get("/health")
def health_check():
    return {"status": "ok"}
