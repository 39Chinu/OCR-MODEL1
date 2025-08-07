import os
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
