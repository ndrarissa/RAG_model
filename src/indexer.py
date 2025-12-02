import faiss
import numpy as np
import pandas as pd
from src.embeddings import get_embedding_model, embed_texts

class FaissIndexer:
    def __init__(self, dimension=None):
        self.index = None
        self.dimension = dimension
        self.ids = [] # map row -> original id

    def build(self, texts, ids=None, model_name=None):
        model = get_embedding_model(model_name or "all-MiniLM-L6-v2")
        vectors = embed_texts(model, texts)
        if self.dimension is None:
            self.dimension = vectors.shape[1]
        self.index = faiss.IndexFlatL2(self.dimension)
        self.index.add(vectors.astype('float32'))
        self.ids = ids or list(range(len(texts)))
        return self

    def save(self, index_path: str, ids_path: str):
        faiss.write_index(self.index, index_path)
        np.save(ids_path, np.array(self.ids))

    def load(self, index_path: str, ids_path: str):
        self.index = faiss.read_index(index_path)
        self.ids = np.load(ids_path).tolist()
        self.dimension = self.index.d
        return self

    def search(self, query_vector, k=4):
        if self.index is None:
            raise ValueError("Index not built or loaded")
        D, I = self.index.search(query_vector.astype('float32'), k)
        return D, I
