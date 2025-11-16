# create_embeddings.py (clean, LangChain 1.0+, CPU-friendly)

import os
import pickle
import faiss
import numpy as np
from langchain_huggingface import HuggingFaceEmbeddings

# ---------------------------
# Folders
# ---------------------------
DOCS_FOLDER = os.path.join("backend", "data", "docs")
MODEL_FOLDER = os.path.join("models")
os.makedirs(MODEL_FOLDER, exist_ok=True)

# ---------------------------
# Load documents
# ---------------------------
docs = []
if not os.path.exists(DOCS_FOLDER):
    raise FileNotFoundError(f"❌ Docs folder not found: {DOCS_FOLDER}")

for filename in sorted(os.listdir(DOCS_FOLDER)):
    if filename.endswith(".txt"):
        path = os.path.join(DOCS_FOLDER, filename)
        with open(path, "r", encoding="utf-8") as f:
            text = f.read().strip()
            if text:
                docs.append(text)

if not docs:
    raise ValueError(f"❌ No text files found in {DOCS_FOLDER}")

print(f"[INFO] ✅ Loaded {len(docs)} documents.")

# ---------------------------
# Create embeddings
# ---------------------------
print("[INFO] Creating embeddings using 'sentence-transformers/sentence-t5-base'...")
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/sentence-t5-base")
embeddings = embedding_model.embed_documents(docs)

# Normalize for cosine similarity
embeddings = np.array(embeddings, dtype="float32")
embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
print(f"[INFO] ✅ Created and normalized {len(embeddings)} embeddings.")

# ---------------------------
# Create FAISS index
# ---------------------------
dimension = embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)  # inner product ≈ cosine similarity
index.add(embeddings)
print(f"[INFO] ✅ FAISS index created with {index.ntotal} vectors (dim={dimension}).")

# ---------------------------
# Save index and metadata
# ---------------------------
index_path = os.path.join(MODEL_FOLDER, "index.faiss")
meta_path = os.path.join(MODEL_FOLDER, "metadata.pkl")

faiss.write_index(index, index_path)
with open(meta_path, "wb") as f:
    pickle.dump(docs, f)

print(f"[INFO] ✅ Saved FAISS index to '{index_path}'")
print(f"[INFO] ✅ Saved metadata to '{meta_path}'")
