import pickle
from pathlib import Path

from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from modules.embeddings import EmbeddingModel
from modules.vectordb import VectorDB
from functools import lru_cache


# ============================================================
# Vector Retriever (ChromaDB)
# ============================================================

class VectorRetriever:
    
    def __init__(self, embedder, db):
        self.embedder = embedder
        self.db = db

    def retrieve(self, query, k=5):

        query_embedding = self.embedder.embed_query(query)

        results = self.db.search(query_embedding, k)

        return results


# ============================================================
# BM25 Retriever
# ============================================================

class BM25Retriever:
    
    def __init__(self):

        self.corpus_path = Path("data/bm25_corpus.pkl")

        self.documents = []
        self.tokenized_documents = []
        self.bm25 = None

        if self.corpus_path.exists():
            self.load()

    def add_documents(self, chunks, source):

        # Append new chunks instead of replacing old ones
        for chunk_id, chunk in enumerate(chunks):

            self.documents.append(
                {
                    "text": chunk,
                    "source": source,
                    "chunk_id": chunk_id
                }
            )

        # Rebuild tokenized corpus from ALL documents
        self.tokenized_documents = [
            doc["text"].lower().split()
            for doc in self.documents
        ]

        # Rebuild BM25 index
        self.bm25 = BM25Okapi(
            self.tokenized_documents
        )

        self.save()

    def search(self, query, top_k=5):
    
        if self.bm25 is None:
            return []

        tokenized_query = query.lower().split()

        # BM25 score for every document
        scores = self.bm25.get_scores(tokenized_query)

        # Pair each document with its score
        ranked = list(zip(self.documents, scores))

        # Sort by score (highest first)
        ranked.sort(
            key=lambda x: x[1],
            reverse=True
        )

        # Return top-k document dictionaries
        return [
            doc
            for doc, score in ranked[:top_k]
        ]
    def save(self):
    
        with open(self.corpus_path, "wb") as f:

            pickle.dump(
                (
                    self.documents,
                    self.tokenized_documents
                ),
                f
            )

    def load(self):

        with open(self.corpus_path, "rb") as f:

            self.documents, self.tokenized_documents = pickle.load(f)

        self.bm25 = BM25Okapi(
            self.tokenized_documents
        )


@lru_cache(maxsize=1)
def get_reranker_model():

    return CrossEncoder(
        "BAAI/bge-reranker-base"
    )
# ============================================================
# Cross Encoder Reranker
# ============================================================

class Reranker:
    
    def __init__(self):

        self.model = get_reranker_model()

    def rerank(self, query, items, top_k=5):

        if len(items) == 0:
            return []

        pairs = [
            (query, item["document"])
            for item in items
        ]

        scores = self.model.predict(pairs)

        for item, score in zip(items, scores):
            item["score"] = float(score)

        items.sort(
            key=lambda x: x["score"],
            reverse=True
        )

        return items[:top_k]


# ============================================================
# Hybrid Retriever
# ============================================================

class HybridRetriever:
    
    def __init__(self, embedder, db, bm25):

        self.vector = VectorRetriever(embedder, db)
        self.bm25 = bm25
        self.reranker = Reranker()

    def retrieve(self, question, top_k=5):

        vector_results = self.vector.retrieve(question)

        vector_docs = vector_results["documents"][0]
        vector_meta = vector_results["metadatas"][0]

        bm25_docs = self.bm25.search(question, top_k)

        combined = []
        seen = set()

        # Vector Results
        for doc, meta in zip(vector_docs, vector_meta):

            if doc not in seen:

                combined.append(
                    {
                        "document": doc,
                        "metadata": meta,
                        "retriever": "vector"
                    }
                )

                seen.add(doc)

        # BM25 Results
        for doc in bm25_docs:
    
            if doc["text"] not in seen:

                combined.append(
                    {
                        "document": doc["text"],
                        "metadata": {
                            "source": doc["source"],
                            "chunk_id": doc["chunk_id"]
                        },
                        "retriever": "bm25"
                    }
                )

        seen.add(doc["text"])

        return self.reranker.rerank(
            question,
            combined,
            top_k
        )
