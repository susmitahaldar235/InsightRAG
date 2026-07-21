import chromadb
from config import CHROMA_DIR


class VectorDB:

    def __init__(self):

        self.client = chromadb.PersistentClient(path=str(CHROMA_DIR))

        self.collection = self.client.get_or_create_collection(
            name="knowledge_base"
        )

    def add_documents(self, chunks, embeddings, source):

        ids = []
        metadatas = []

        for i in range(len(chunks)):

            ids.append(f"{source}_{i}")

            metadatas.append({
                "source": source,
                "chunk_id": i
            })

        self.collection.add(
            ids=ids,
            documents=chunks,
            embeddings=embeddings.tolist(),
            metadatas=metadatas
        )
    def search(self, query_embedding, k=5):
    
        return self.collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=k
    )

    def count(self):

        return self.collection.count()