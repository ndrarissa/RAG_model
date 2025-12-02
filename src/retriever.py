def retrieve_context(query, top_k=3):
    query_emb = model.encode([query])
    D, I = index.search(query_emb, top_k)

    results = []
    for idx in I[0]:
        results.append(metadata[idx])
    return results
