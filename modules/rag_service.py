from pathlib import Path

from modules.loader import DocumentLoader
from modules.chunker import DocumentChunker
from modules.embeddings import EmbeddingModel
from modules.vectordb import VectorDB
from modules.retrieval import VectorRetriever
from modules.generator import Generator
from modules.retrieval import HybridRetriever, BM25Retriever
from modules.corrective_rag import CorrectiveRAG
from modules.rag_graph import RAGGraph


class RAGService:

    def __init__(self):
        self.loader = DocumentLoader()
        self.chunker = DocumentChunker()
        self.embedder = EmbeddingModel()
        self.db = VectorDB()
        self.bm25 = BM25Retriever()
        self.retriever = HybridRetriever(
            self.embedder,
            self.db,
            self.bm25
        )
        self.generator = Generator()
        self.corrective_rag = CorrectiveRAG()
        self.graph = RAGGraph(
            self.retriever,
            self.generator,
            self.corrective_rag,
        )

    def index_document(self, pdf_path: Path):

        text = self.loader.load_pdf(pdf_path)

        chunks = self.chunker.chunk_text(text)

        self.bm25.add_documents(chunks,pdf_path.name)

        embeddings = self.embedder.embed_documents(chunks)

        self.db.add_documents(
            chunks=chunks,
            embeddings=embeddings,
            source=pdf_path.name
        )

        return len(chunks)
    
    def answer(self, question):
    
        state = {
            "question": question,
            "retrieved_docs": [],
            "rewritten_question": "",
            "retry": False,
            "answer": "",
        }

        result = self.graph.graph.invoke(state)

        return result["answer"], result["retrieved_docs"]
    
    def baseline_answer(self, question):
    
        retrieved_docs = self.retriever.retrieve(question)

        context = [
            item["document"]
            for item in retrieved_docs
        ]

        metadata = [
            item["metadata"]
            for item in retrieved_docs
        ]

        answer = self.generator.generate_answer(
            question,
            context,
            metadata
        )

        return answer, retrieved_docs