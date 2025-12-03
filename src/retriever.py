# src/retriever.py
import numpy as np
from src.embeddings import get_embedding_model, embed_texts
from src.indexer import FaissIndexer
import os


DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
INDEX_PATH = os.path.join(DATA_DIR, 'faiss.index')
IDS_PATH = os.path.join(DATA_DIR, 'ids.npy')
CHUNKS_PATH = os.path.join(DATA_DIR, 'chunks.txt')


class Retriever:
    def __init__(self, model_name=None):
        self.model = get_embedding_model(model_name or 'all-MiniLM-L6-v2')
        self.indexer = FaissIndexer().load(INDEX_PATH, IDS_PATH)
        # load chunks (must be same order as index ids)
        with open(CHUNKS_PATH, 'r', encoding='utf-8') as f:
            self.chunks = [line.strip() for line in f.readlines()]


    def retrieve(self, query, top_k=3):
        q_vec = embed_texts(self.model, [query])
        D, I = self.indexer.search(q_vec, k=top_k)
        results = []
        for idx in I[0]:
            results.append(self.chunks[idx])
        return results