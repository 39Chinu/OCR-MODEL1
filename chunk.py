import json
import numpy as np
from sentence_transformers import SentenceTransformer

def recursive_chunk(text: str, chunk_size: int = 800, min_chunk_size: int = 200) -> list:
    def split_recursively(t: str) -> list:
        if len(t) <= chunk_size:
            return [t]

        for sep in ["\n\n", ".", " "]:
            parts = t.split(sep)
            if len(parts) == 1:
                continue

            chunks = []
            current = ""
            for part in parts:
                if current and len(current) + len(part) + len(sep) > chunk_size:
                    chunks.append(current.strip())
                    current = ""
                current += part + sep

            if current.strip():
                chunks.append(current.strip())

            if all(len(c) <= chunk_size for c in chunks):
                return chunks

        return [t[i:i + chunk_size] for i in range(0, len(t), chunk_size)]

    chunks = split_recursively(text)
    return [c for c in chunks if len(c.strip()) >= min_chunk_size]


# === Config ===
file_paths = [
    "markdown files",
    # Add more file paths if needed
]

model = SentenceTransformer("all-MiniLM-L6-v2")

for file_path in file_paths:
    if not file_path.strip():
        continue

    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    chunks = recursive_chunk(text)

    embeddings = model.encode(chunks, convert_to_numpy=True)

    # Save embeddings as .npy
    npy_path = file_path.replace(".md", "_embeddings.npy")
    np.save(npy_path, embeddings)
