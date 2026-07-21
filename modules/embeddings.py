from sentence_transformers import SentenceTransformer
from functools import lru_cache

@lru_cache(maxsize=1)
def get_embedding_model():

    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(
        "BAAI/bge-small-en-v1.5"
    )


class EmbeddingModel:

    def __init__(self):
        self.model = get_embedding_model()

    def embed_documents(self, chunks):
        return self.model.encode(chunks, show_progress_bar=True)

    def embed_query(self, query):
        return self.model.encode(query)