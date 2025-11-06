# create_embeddings.py

from sentence_transformers import SentenceTransformer
import faiss
import pickle
import os

# ---------------------------
# Folders (absolute paths, Windows-safe)
# ---------------------------
import os

DOCS_FOLDER = os.path.join("D:/", "institute_project", "backend", "data", "docs")
MODEL_FOLDER = os.path.join("D:/", "institute_project", "models")


# ---------------------------
# Check docs folder exists
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
# Word-based chunking
# ---------------------------
def chunk_text(text, chunk_size=100, overlap=20):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

# ---------------------------
# Create chunks and metadata
# ---------------------------
all_chunks = []
chunk_metadata = []

for doc, fname in zip(documents, file_names):
    chunks = chunk_text(doc, chunk_size=100, overlap=20)
    all_chunks.extend(chunks)
    chunk_metadata.extend([fname] * len(chunks))

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
try:
    os.makedirs(MODEL_FOLDER, exist_ok=True)
    index_file = os.path.join(MODEL_FOLDER, "index.faiss")
    metadata_file = os.path.join(MODEL_FOLDER, "metadata.pkl")

    faiss.write_index(index, index_file)
    with open(metadata_file, "wb") as f:
        pickle.dump({"chunks": all_chunks, "chunk_metadata": chunk_metadata}, f)

    print(f"[INFO] Saved FAISS index to '{index_file}'")
    print(f"[INFO] Saved metadata to '{metadata_file}'")
except Exception as e:
    print("[ERROR] Failed to save files:", e)
