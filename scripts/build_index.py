import sys, os

# Get project root (folder above scripts/)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.indexer import FaissIndexer
from src.embeddings import get_embedding_model, embed_texts
import pandas as pd
import numpy as np

DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'MedQuAD_Dataset_RAG_Scenario.csv')
INDEX_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'faiss.index')
IDS_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'ids.npy')
TEXTS_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'chunks.txt')

if __name__ == '__main__':
    df = pd.read_csv(DATA_PATH)

    # Create combined chunk text
    df['chunk'] = df.apply(lambda r:
        f"Question: {r['question']}\nAnswer: {r['answer']}",
        axis=1
    )
    
    texts = df['chunk'].tolist()
    ids = list(range(len(df)))  # AUTO-GENERATE IDs

    indexer = FaissIndexer()
    indexer.build(texts, ids=ids)
    indexer.save(INDEX_PATH, IDS_PATH)

    # save text chunks
    with open(TEXTS_PATH, 'w', encoding='utf-8') as f:
        for t in texts:
            f.write(t.replace('\n', ' ') + '\n')

    print("Index built and saved.")
