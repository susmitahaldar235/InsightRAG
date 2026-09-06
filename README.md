# InsightRAG

> A self-correcting Retrieval-Augmented Generation (RAG) system for question answering over PDF documents.

InsightRAG is a PDF-based question-answering system that combines **semantic retrieval, keyword-based retrieval, reranking, and corrective retrieval** to improve the relevance of retrieved context before generating an answer. The system uses **PyMuPDF** for PDF text extraction, **LangChain RecursiveCharacterTextSplitter** for chunking, **BAAI/bge-small-en-v1.5** for embeddings, **ChromaDB** for persistent vector storage, **BM25** for lexical retrieval, **BAAI/bge-reranker-base** for cross-encoder reranking, **LangGraph** for orchestrating the corrective RAG workflow, and **Gemini 2.5 Flash** for answer generation. A **Streamlit** interface is used for PDF uploading, question answering, source display, and execution tracing.

---

## Features

- PDF document ingestion and text extraction
- Recursive text chunking with overlapping chunks
- Semantic vector retrieval using BGE embeddings
- Keyword-based retrieval using BM25
- Hybrid retrieval combining vector and lexical search
- CrossEncoder-based reranking
- Corrective RAG workflow using LangGraph
- Query rewriting when retrieved context is considered insufficient
- Grounded answer generation using Gemini 2.5 Flash
- Source information displayed with generated answers
- Execution/reasoning trace for the RAG workflow
- Persistent ChromaDB vector storage
- Persistent BM25 corpus
- Streamlit-based interactive interface
- Baseline vs corrective RAG evaluation using Ragas

---

## Architecture

InsightRAG has two main flows: **document indexing** and **question answering**.

### High-Level Architecture

```text
                         ┌───────────────────┐
                         │    Streamlit UI   │
                         │                   │
                         │ Upload PDF        │
                         │ Ask Question      │
                         └─────────┬─────────┘
                                   │
                                   ▼
                         ┌───────────────────┐
                         │    RAGService     │
                         └─────────┬─────────┘
                                   │
                  ┌────────────────┴────────────────┐
                  │                                 │
               INDEXING                           QUERY
                  │                                 │
                  ▼                                 ▼
          ┌──────────────┐                 ┌────────────────┐
          │   PyMuPDF    │                 │   LangGraph    │
          │ PDF Loader   │                 │  Orchestrator  │
          └──────┬───────┘                 └───────┬────────┘
                 │                                 │
                 ▼                                 ▼
          ┌──────────────┐                 ┌────────────────┐
          │   Chunking   │                 │ Hybrid         │
          │  LangChain   │                 │ Retrieval      │
          └──────┬───────┘                 └───────┬────────┘
                 │                                 │
                 ▼                          ┌──────┴──────┐
          ┌──────────────┐                  ▼             ▼
          │ BGE          │             Vector Search    BM25
          │ Embeddings   │             (ChromaDB)       Index
          └──────┬───────┘                  │             │
                 │                          └──────┬──────┘
                 ▼                                 ▼
       ┌───────────────────┐                Combine Results
       │ ChromaDB          │                       │
       │ + BM25 Index      │                       ▼
       └───────────────────┘                ┌──────────────┐
                                            │ CrossEncoder │
                                            │  Reranker    │
                                            └──────┬───────┘
                                                   │
                                                   ▼
                                            ┌──────────────┐
                                            │    Grade     │
                                            └──────┬───────┘
                                                   │
                                           ┌───────┴───────┐
                                           │               │
                                         Good            Poor
                                           │               │
                                           ▼               ▼
                                      Generate          Rewrite
                                                           │
                                                           ▼
                                                   Hybrid Retrieval
                                                           │
                                                           ▼
                                                       Generate
                                                           │
                                                           ▼
                                                   Gemini 2.5 Flash
                                                           │
                                                           ▼
                                                   Answer + Sources ```
## Document Indexing Flow

When a PDF is uploaded, InsightRAG prepares the document for efficient future retrieval.

PDF
 │
 ▼
PyMuPDF
 │
 ▼
Extract Text
 │
 ▼
RecursiveCharacterTextSplitter
 │
 ▼
Text Chunks
 │
 ├───────────────┐
 ▼               ▼
BGE Embeddings   BM25 Corpus
 │               │
 ▼               ▼
ChromaDB         BM25 Index

The document is indexed only during the ingestion phase. The resulting searchable structures are then reused when users ask questions.

Chunking

The project uses LangChain's RecursiveCharacterTextSplitter with:

Chunk size: 500
Chunk overlap: 100
Separators:
\n\n
\n
.
space
empty string

The overlap helps preserve context between neighboring chunks.

Query Processing Flow

When a user asks a question, the question enters the LangGraph workflow.

Question
   │
   ▼
Retrieve
   │
   ▼
Hybrid Retrieval
   │
   ├── Vector Search → ChromaDB
   │
   └── BM25 Search
          │
          ▼
     Combine Results
          │
          ▼
     CrossEncoder
       Reranking
          │
          ▼
        Grade
       /     \
    Good     Poor
      │        │
      ▼        ▼
  Generate   Rewrite
                │
                ▼
       Hybrid Retrieval Again
                │
                ▼
             Generate
                │
                ▼
             Answer
Hybrid Retrieval

InsightRAG uses two complementary retrieval methods.

1. Vector Retrieval

The question is converted into an embedding using:

BAAI/bge-small-en-v1.5

The query embedding is compared against the document embeddings stored in ChromaDB.

This helps retrieve chunks that are semantically similar to the question even when the exact words are different.

2. BM25 Retrieval

BM25 performs lexical/keyword-based retrieval over the indexed document chunks.

This is useful when the question contains:

Exact terms
Names
Technical terminology
Specific keywords
Why Hybrid Retrieval?

Vector retrieval and BM25 have complementary strengths.

Vector Search
→ understands semantic similarity

BM25
→ captures exact keyword matches

Combining both gives the system a broader candidate set before reranking.

CrossEncoder Reranking

After vector and BM25 retrieval, the candidate chunks are combined and passed to:

BAAI/bge-reranker-base

The CrossEncoder evaluates the relationship between:

Question + Retrieved Chunk

and assigns a relevance score.

The candidates are then sorted based on these scores and the top relevant chunks are selected for generation.

The purpose of reranking is to improve the precision of the final context without applying the more expensive CrossEncoder to the entire document collection.

Corrective RAG

A key component of InsightRAG is its corrective retrieval workflow.

Instead of immediately generating an answer after the first retrieval, the system checks whether the retrieved context is sufficient.

```text Retrieve
   ↓
Grade
   ↓
Is the retrieved context sufficient?
   │
   ├── Yes → Generate
   │
   └── No
        ↓
      Rewrite
        ↓
  Hybrid Retrieval Again
        ↓
     Generate ```

The current implementation performs one bounded correction.

It does not repeatedly loop through grading and rewriting indefinitely.

LangGraph Workflow

LangGraph is used as the workflow orchestrator.

The graph contains the following main nodes:

Retrieve

Calls the hybrid retriever to obtain relevant document chunks.

Grade

Checks whether the retrieved context is sufficient and determines whether correction is required.

Rewrite

If the retrieval is considered poor, the question is rewritten and passed through retrieval again.

Generate

The selected document chunks and metadata are passed to the generator.

The workflow can therefore follow either:

Retrieve → Grade → Generate

or:

Retrieve → Grade → Rewrite → Generate

The rewritten query is used for the second retrieval, while the original question is retained for final answer generation.

Answer Generation

The final retrieved chunks are inserted into a grounded prompt.

The prompt instructs the model to:

Answer only from the provided context
Avoid unsupported information
State when enough information cannot be found
Include source document names

The generated response is produced using:

Gemini 2.5 Flash
Storage

InsightRAG maintains two searchable data structures.

ChromaDB

ChromaDB is used for persistent vector storage.

It stores:

Document chunks
Embedding vectors
Source information
Chunk IDs

The vector database uses a persistent ChromaDB client.

BM25

The BM25 corpus is persisted at:

data/bm25_corpus.pkl

It stores the indexed documents and their tokenized representations.

Both indexes are created during document indexing and are later accessed by the HybridRetriever during question answering.

Project Structure
InsightRAG/
│
├── modules/
│   ├── loader.py
│   ├── chunker.py
│   ├── embeddings.py
│   ├── vectordb.py
│   ├── retrieval.py
│   ├── generator.py
│   ├── llm.py
│   ├── prompts.py
│   ├── corrective_rag.py
│   ├── rag_nodes.py
│   ├── rag_graph.py
│   ├── rag_state.py
│   └── rag_service.py
│
├── data/
│   └── bm25_corpus.pkl
│
├── app2.py
├── requirements.txt
├── .env
└── README.md
Module Responsibilities
Module	Responsibility
loader.py	Extracts text from PDFs using PyMuPDF
chunker.py	Splits extracted text into overlapping chunks
embeddings.py	Generates BGE document/query embeddings
vectordb.py	Stores and searches embeddings using ChromaDB
retrieval.py	Implements vector, BM25, hybrid retrieval, and reranking
generator.py	Builds the generation pipeline
llm.py	Interfaces with Gemini 2.5 Flash
prompts.py	Builds grounded prompts using retrieved context
corrective_rag.py	Determines whether retrieval needs correction
rag_nodes.py	Defines LangGraph node operations
rag_graph.py	Defines and compiles the LangGraph workflow
rag_state.py	Defines the state passed through the workflow
rag_service.py	Connects indexing, retrieval, generation, and graph components
app2.py	Streamlit user interface
Technology Stack
Component	Technology
Language	Python
UI	Streamlit
PDF Processing	PyMuPDF
Text Chunking	LangChain RecursiveCharacterTextSplitter
Embeddings	BAAI/bge-small-en-v1.5
Vector Database	ChromaDB
Lexical Retrieval	BM25
Reranking	BAAI/bge-reranker-base
Workflow	LangGraph
LLM	Gemini 2.5 Flash
Evaluation	Ragas
Evaluation

The system was evaluated by comparing a baseline RAG pipeline against the corrective RAG pipeline.

Baseline
Answer Correctness: 0.2647
Faithfulness:       1.0000
Corrective RAG
Answer Correctness: 0.5137
Faithfulness:       1.0000

The evaluation showed an improvement in answer correctness after introducing the corrective retrieval workflow.

The evaluation setup used generated expected answers and Ragas-based evaluation. Reference contexts were not available in the evaluation dataset, so context-based metrics were not used.

Example

A user can upload a research paper and ask:

What is the sample size used in the regression analysis?

The system:

1. Searches the document using vector retrieval.
2. Searches the same indexed content using BM25.
3. Combines the candidates.
4. Reranks them using the CrossEncoder.
5. Grades the retrieved context.
6. Rewrites the question if the context is insufficient.
7. Retrieves again when correction is required.
8. Passes the relevant context to Gemini.
9. Displays the generated answer and sources in Streamlit.
Installation

Clone the repository:

git clone <your-repository-url>
cd InsightRAG

Create a virtual environment:

python -m venv venv

Activate it on Windows:

venv\Scripts\activate

Install dependencies:

pip install -r requirements.txt

Create a .env file:

GEMINI_API_KEY=your_api_key
Running the Application

Start the Streamlit application:

streamlit run app2.py

Then:

Upload a PDF.
Index the document.
Ask a question.
InsightRAG retrieves relevant context.
The corrective workflow evaluates the retrieval.
Gemini generates the grounded answer.
The UI displays the answer and sources.
Design Decisions
Why use embeddings?

Embeddings allow the system to retrieve semantically related information rather than relying only on exact keyword matches.

Why use BM25 as well?

BM25 provides strong lexical retrieval and is useful for exact terms and technical keywords that semantic retrieval may miss.

Why use a CrossEncoder?

Initial retrieval needs to be relatively efficient and broad. The CrossEncoder is more computationally expensive, so it is applied only to the smaller candidate set to improve ranking precision.

Why use LangGraph?

The project contains conditional workflow logic. LangGraph makes the retrieval, grading, rewriting, and generation steps explicit and stateful instead of implementing the entire workflow through deeply nested conditional code.

Why corrective RAG?

A normal RAG pipeline assumes the first retrieval is good enough. InsightRAG adds a correction step so that poor retrieval can trigger query rewriting and another retrieval attempt before generation.

Current Limitations

The current implementation is designed as a project/research prototype rather than a production-scale service.

The corrective workflow currently performs only one correction.
PDF page-level metadata is not preserved by the current loader.
BM25 tokenization uses simple lowercase whitespace splitting.
API retry, timeout, and fallback handling are not extensively implemented.
The current application is designed around a Streamlit interface.
Persistent indexes require explicit reset/management when changing document sets.
The evaluation uses generated expected answers rather than a fully human-annotated benchmark.
The current implementation does not provide production-grade multi-user isolation or distributed scaling.
Future Improvements

Potential improvements include:

Multi-step bounded corrective retrieval
Page-level metadata and citations
Better BM25 preprocessing
More robust evaluation with human-verified reference answers
Retrieval metrics such as Recall@K and MRR
API retry and exponential backoff
Rate limiting
Structured logging and monitoring
User/document-level access isolation
Asynchronous document processing
Scalable external vector storage
Background processing for large PDF files
Production API separation from the UI
Key Concept

The core idea behind InsightRAG is:

Don't immediately ask the LLM to answer.

First:
Retrieve → Combine → Rerank → Check

If retrieval is poor:
Rewrite → Retrieve Again

Then:
Generate a grounded answer.

This makes the system more robust than a basic:

Question → Vector Search → LLM

pipeline by introducing hybrid retrieval, reranking, and retrieval correction before answer generation.

End-to-End Summary

InsightRAG follows two distinct phases.

Indexing Phase
PDF
 ↓
PyMuPDF
 ↓
Text
 ↓
Chunking
 ↓
BGE Embeddings
 ↓
ChromaDB

and in parallel:

Chunks
 ↓
BM25
 ↓
Persistent BM25 Corpus
Query Phase
Question
 ↓
LangGraph
 ↓
Hybrid Retrieval
 ├── ChromaDB Vector Search
 └── BM25 Search
 ↓
Combine
 ↓
CrossEncoder Reranking
 ↓
Grade
 ↓
 ├── Good → Generate
 │
 └── Poor → Rewrite
              ↓
        Hybrid Retrieval Again
              ↓
           Generate
              ↓
       Gemini 2.5 Flash
              ↓
       Answer + Sources
Author

Susmita Haldar

Built as a project exploring Retrieval-Augmented Generation, hybrid information retrieval, reranking, corrective workflows, and LLM-based question answering over documents.
