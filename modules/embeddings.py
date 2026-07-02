from sentence_transformers import SentenceTransformer


class EmbeddingModel:

    def __init__(self):
        self.model = SentenceTransformer("BAAI/bge-small-en-v1.5")

    def embed_documents(self, chunks):
        return self.model.encode(chunks, show_progress_bar=True)

    def embed_query(self, query):
        return self.model.encode(query)