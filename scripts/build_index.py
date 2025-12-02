import pandas as pd
from src.embeddings import get_embedding_model, embed_texts
from src.indexer import FaissIndexer
import numpy as np
import os

import sys, os

# Get project root (folder above scripts/)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)



DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'MedQuAD_Dataset_RAG_Scenario.csv')
INDEX_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'faiss.index')
IDS_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'ids.npy')
TEXTS_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'chunks.txt')

if __name__ == '__main__':
    df = pd.read_csv(DATA_PATH)
    # Create chunk text combining Q+A
    df['chunk'] = df.apply(lambda r: f"Question: {r['question']}\nAnswer: {r['answer']}", axis=1)
    texts = df['chunk'].tolist()
    ids = df['id'].tolist()

    indexer = FaissIndexer()
    indexer.build(texts, ids=ids)
    indexer.save(INDEX_PATH, IDS_PATH)

# save texts to recover later
    with open(TEXTS_PATH, 'w', encoding='utf-8') as f:
        for t in texts:
            f.write(t.replace('\n', ' ') + '\n')

    print('Index built and saved.')
