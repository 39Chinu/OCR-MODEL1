# OCR-MODEL

A complete pipeline for building a **Retrieval-Augmented Generation (RAG)** system using PDFs as input. This project uses `marker-pdf` for accurate text extraction with layout awareness, `numpy` for data handling, and LLMs for intelligent question-answering.

---

##  Features

- Merge and save multiple PDFs into a single file.
- Load and extract text using `marker-pdf` (with layout preservation).
- Support for both single and batch PDF extraction.
- Convert extracted `.md` files into text chunks using recursive chunking.
- Generate sentence embeddings from chunks and save as `.npy`.
- Perform contextual search and Q&A using RAG (LLM-based).

---

##  Project Structure

```
 pdf-rag
│
├── pdfs/                    # Input PDF files
├── merged.pdf               # Merged PDF file (optional)
├── extracted_markdowns/    # Marker-pdf generated .md files
├── chunks/                 # Chunked text files
├── embeddings/             # Saved numpy arrays (.npy) for chunks
├── rag.py                  # Main script for querying with RAG
├── requirements.txt
└── README.md               # This file
```

---




### Dependencies

- `marker-pdf`
- `numpy`
- `sentence-transformers`
- `langchain` or `llama-index` (based on LLM setup)
- `PyPDF2` (for merging)
- `markdown`, `tiktoken` (for chunking if needed)
- Your preferred LLM backend (e.g., `OpenAI`, `LLama`, `Gemini`, etc.)

---


### 1. Save PDFs into one file

Use a Python script or tool like `PyPDF2` to merge:

```python
from PyPDF2 import PdfMerger

merger = PdfMerger()
for file in os.listdir("pdfs"):
    merger.append(f"pdfs/{file}")
merger.write("merged.pdf")
merger.close()
```

---

### 2. Extract text using marker-pdf

#### Single PDF:

```bash
!marker_single merged.pdf
```

#### Multiple PDFs:

```bash
!marker pdfs/
```

Markdown files will be saved in the current directory or specified output folder.

---

### 3. Chunk the `.md` files

Use a recursive chunking logic (e.g., based on headings, sentences, or token count) and save the output chunks.

---

### 4. Generate embeddings

Convert each chunk into a dense vector using a sentence-transformer model:

```python
from sentence_transformers import SentenceTransformer
import numpy as np
import os

model = SentenceTransformer("all-MiniLM-L6-v2")

chunks_dir = 'chunks/'
embeddings = []

for file in os.listdir(chunks_dir):
    with open(os.path.join(chunks_dir, file), 'r', encoding='utf-8') as f:
        text = f.read()
        emb = model.encode(text)
        embeddings.append(emb)

np.save("embeddings/embeddings.npy", embeddings)
```

---

### 5. Build `rag.py`

script :import os
import requests
import numpy as np
from scipy.spatial import distance
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

txtdir = 'chunk-texts as npy'
embdir = 'Embeddings as npy'

combined_chunks = []
combined_embs = []

for f in os.listdir(txtdir):
    embfile = os.path.join(embdir, f)
    txtfile = os.path.join(txtdir, f)

    if os.path.exists(embfile) and os.path.exists(txtfile):
        try:
            embs = np.load(embfile)
            chunks = np.load(txtfile, allow_pickle=True)
            if len(embs) != len(chunks):
                continue
            combined_chunks.extend(chunks)
            combined_embs.extend(embs)
        except Exception:
            continue
    else:
        continue

Query = 'What is residual prophet inequality?'
try:
    query_emb = model.encode(Query, convert_to_numpy=True)
except Exception:
    query_emb = None

if query_emb is not None and combined_embs:
    try:
        combined_embs_array = np.array(combined_embs)
        distances = distance.cdist([query_emb], combined_embs_array, 'cosine')[0]
        top_3_indices = np.argsort(distances)[:3]

        llm_prompt = f'Answer the user query based on following context. User query = {Query}\nContext:'
        counter = 1
        for idx in top_3_indices:
            ctx = combined_chunks[idx]
            llm_prompt += f'\nContext # {counter} - {ctx}'
            counter += 1

        print(llm_prompt)
    except Exception:
        pass

- Loads embeddings
- Accepts user query
- Finds the most relevant chunk(s)
- Sends combined context + query to an LLM
- Returns an intelligent answer

---

##  Sample Usage/ output

```bash
python rag.py
# Query: "What is residual prophet inequality?"
# Answer: [ The prophet inequality problem, as introduced by [Krengel and Sucheston](#page-29-0) [\(1977\)](#page-29-0), was resolved by using a dynamic program that gave a tight approximation ratio of 1/2. [Samuel-Cahn](#page-30-0) [\(1984\)](#page-30-0) later proved that a single-threshold strategy yields the same guarantee; this also showed that the order in which the variables are observed is immaterial.]
```

---



