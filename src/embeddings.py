from sentence_transformers import SentenceTransformer

def get_embedding_model(model_name: str = "all-MiniLM-L6-v2"):
    model = SentenceTransformer(model_name)
    return model

def embed_texts(model, texts):
    # returns numpy array
    return model.encode(texts, convert_to_numpy=True, show_progress_bar=False)

