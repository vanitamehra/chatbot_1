# main.py (CPU-only, manual RAG + PDF generator, no LangChain)

from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import faiss
import numpy as np

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import os

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
MODEL_FOLDER = os.path.join("models")
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
# Embedding model using Sentence-Transformers
# ---------------------------
embedding_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

# ---------------------------
# Helper: retrieve top k docs
# ---------------------------
MAX_EMBED_TOKENS = 512  

def retrieve_docs(query_text, k=10):
    query_text = query_text[:1000]  # truncate to avoid long text
    query_vector = embedding_model.encode(query_text, convert_to_numpy=True)
    query_vector = query_vector.astype("float32")
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
    if not retrieved_texts or all(len(t.strip()) == 0 for t in retrieved_texts):
        course_keywords = ["course", "curriculum", "module", "syllabus"]
        fee_keywords = ["fee", "enrollment", "tuition"]
        q_lower = query_text.lower()

        if any(word in q_lower for word in course_keywords):
            return "For further details please contact the institute."
        elif any(word in q_lower for word in fee_keywords):
            return "For enrollment or fee details, please contact the institute directly."
        else:
            return "I can only answer questions related to courses."

    context = "\n".join(retrieved_texts)
    
    prompt = f"""Answer the question ONLY using the context below. 

IMPORTANT: If the context does NOT contain the answer, DO NOT guess. 
Only return the fallback messages as instructed.

Follow these rules based on the question type:

1. **Course-related questions:** Answer only using the context. 
   If the context does not contain the answer, respond: 'For details about courses, please contact the institute.'
2. **Enrollment or fee questions:** Respond: 'For enrollment or fee details, please contact the institute directly.'
3. **Other questions:** Respond: 'I can only answer questions related to courses.'

Context:
{context}

Question: {query_text}
Answer:"""

    result = flan_pipe(prompt, max_new_tokens=256, do_sample=False)
    if not result or 'generated_text' not in result[0]:
        return "Sorry, I could not generate an answer."
    return result[0]['generated_text']

# ---------------------------
# API endpoints
# ---------------------------
@app.get("/")
def root():
    return {"message": "RAG + Chat API running", "status": "ok"}

@app.post("/ask", response_model=Answer)
def ask_question(query: Query):
    try:
        answer_text = run_rag(query.question)
        if not answer_text:
            answer_text = "Sorry, I could not generate an answer."
        return Answer(answer=answer_text)
    except Exception as e:
        return Answer(answer=f"[ERROR] {e}")

@app.post("/chat", response_model=Answer)
def chat_endpoint(query: Query):
    return ask_question(query)

@app.get("/health")
def health_check():
    return {"status": "ok"}
