# backend/create_embeddings.py

from sentence_transformers import SentenceTransformer
import faiss
import pickle
import os

# ---------------------------
# Folders
# ---------------------------
DOCS_FOLDER = "D:/institute_project/backend/data/docs"  # change if different
MODEL_FOLDER = "D:\institute_project\models"

# ---------------------------
# Check folder exists
# ---------------------------
if not os.path.exists(DOCS_FOLDER):
    raise FileNotFoundError(f"Folder not found: {DOCS_FOLDER}")

# ---------------------------
# Read all .txt files
# ---------------------------
documents = []
file_names = []

for filename in os.listdir(DOCS_FOLDER):
    if filename.endswith(".txt"):
        path = os.path.join(DOCS_FOLDER, filename)
        with open(path, "r", encoding="utf-8") as f:
            text = f.read().strip()
            if text:
                documents.append(text)
                file_names.append(filename)

if not documents:
    raise ValueError(f"No text files found in {DOCS_FOLDER}")

print(f"[INFO] Loaded {len(documents)} documents from folder.")

# ---------------------------
# Word-based chunking function
# ---------------------------
def chunk_text(text, chunk_size=100, overlap=20):
    """
    Split text into chunks of approx `chunk_size` words with `overlap`.
    """
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap  # sliding window
    return chunks

# ---------------------------
# Create chunks and metadata
# ---------------------------
all_chunks = []
chunk_metadata = []

for doc, fname in zip(documents, file_names):
    chunks = chunk_text(doc, chunk_size=100, overlap=20)  # adjust numbers if needed
    all_chunks.extend(chunks)
    chunk_metadata.extend([fname] * len(chunks))  # track which file the chunk came from

print(f"[INFO] Created {len(all_chunks)} chunks from {len(documents)} documents.")

# ---------------------------
# Create embeddings
# ---------------------------
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = embedding_model.encode(all_chunks, show_progress_bar=True).astype('float32')
print("[INFO] Created embeddings for all chunks.")

# ---------------------------
# Create FAISS index
# ---------------------------
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)
print(f"[INFO] FAISS index created with {index.ntotal} vectors.")

# ---------------------------
# Save index and metadata
# ---------------------------
os.makedirs(MODEL_FOLDER, exist_ok=True)
faiss.write_index(index, os.path.join(MODEL_FOLDER, "index.faiss"))

with open(os.path.join(MODEL_FOLDER, "metadata.pkl"), "wb") as f:
    pickle.dump({
        "chunks": all_chunks,
        "chunk_metadata": chunk_metadata
    }, f)

print(f"[INFO] Saved FAISS index and metadata to '{MODEL_FOLDER}'.")
