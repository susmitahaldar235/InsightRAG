# InsightRAG — Corrective Hybrid RAG System

InsightRAG is a **Retrieval-Augmented Generation (RAG)** system for question answering over PDF documents.

Unlike a basic RAG pipeline that directly sends the first retrieved results to an LLM, InsightRAG combines:

- **Semantic retrieval** using BGE embeddings + ChromaDB
- **Lexical retrieval** using BM25
- **CrossEncoder reranking** for improved retrieval precision
- **Corrective RAG** to detect poor retrieval and rewrite the query
- **LangGraph** for stateful workflow orchestration
- **Gemini 2.5 Flash** for grounded answer generation

The goal is to improve the quality and reliability of answers by **checking retrieval quality before generation**.

---

## Architecture

```text
                         ┌───────────────────┐
                         │    Streamlit UI   │
                         │                   │
                         │    Upload PDF     │
                         │    Ask Question   │
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
          ┌───────────────────┐              Combine Results
          │     ChromaDB      │                      │
          │   + BM25 Index    │                      ▼
          └───────────────────┘              ┌──────────────┐
                                             │ CrossEncoder │
                                             │   Reranker   │
                                             └──────┬───────┘
                                                    │
                                                    ▼
                                             ┌──────────────┐
                                             │     Grade    │
                                             └──────┬───────┘
                                                    │
                                             ┌──────┴──────┐
                                             │             │
                                           Good          Poor
                                             │             │
                                             ▼             ▼
                                         Generate       Rewrite
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
                                                   Answer + Sources
```

---

## How It Works

InsightRAG operates in two main phases:

1. **Document Indexing**
2. **Question Answering**

---

## 1. Document Indexing

When a PDF is uploaded, the document is processed and converted into searchable representations.

```text
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
 ├───────────────────┐
 ▼                   ▼
BGE Embeddings      BM25 Corpus
 │                   │
 ▼                   ▼
ChromaDB            BM25 Index
```

The document is indexed during the ingestion phase. The resulting vector and lexical indexes are then reused during question answering.

### Chunking

The project uses LangChain's `RecursiveCharacterTextSplitter`.

| Parameter | Value |
|---|---:|
| Chunk Size | 500 |
| Chunk Overlap | 100 |

The configured separators allow the splitter to preserve larger textual boundaries before falling back to smaller ones.

The overlap helps preserve context between neighboring chunks.

---

# 2. Query Processing

When a user asks a question, it enters the LangGraph workflow.

```text
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
```

---

## Hybrid Retrieval

InsightRAG uses two complementary retrieval strategies.

### 1. Vector Retrieval

The question is converted into an embedding using:

**`BAAI/bge-small-en-v1.5`**

The query embedding is compared with document embeddings stored in ChromaDB.

This allows the system to retrieve chunks based on **semantic similarity**, even when the exact words in the query do not appear in the document.

### 2. BM25 Retrieval

BM25 performs lexical/keyword-based retrieval over the indexed document chunks.

It is particularly useful for queries containing:

- Exact terms
- Names
- Technical terminology
- Specific keywords

### Why Use Both?

The two retrieval approaches have complementary strengths:

```text
Vector Search
     │
     └── Semantic similarity

BM25
     │
     └── Exact keyword matching
```

Combining their results provides a broader candidate set before the reranking step.

---

# CrossEncoder Reranking

After vector and BM25 retrieval, the candidate chunks are combined and passed to:

**`BAAI/bge-reranker-base`**

The CrossEncoder evaluates the relationship between:

```text
Question + Retrieved Chunk
```

and assigns a relevance score.

The candidates are then sorted according to their relevance scores, and the highest-ranked chunks are selected for generation.

The CrossEncoder is applied only to the retrieved candidate set rather than the entire document collection, reducing the computational cost of reranking.

---

# Corrective RAG

A key feature of InsightRAG is its **corrective retrieval workflow**.

Instead of assuming that the first retrieval is sufficient, the system evaluates the retrieved context before generating an answer.

```text
Retrieve
   │
   ▼
 Grade
   │
   ├───────────────┐
   │               │
  Good            Poor
   │               │
   ▼               ▼
Generate         Rewrite
                   │
                   ▼
            Hybrid Retrieval
                   │
                   ▼
                Generate
```

If the retrieved context is considered insufficient:

1. The query is rewritten.
2. Hybrid retrieval is performed again.
3. The resulting context is passed to the generation stage.

The current implementation performs **one bounded correction** rather than repeatedly looping through retrieval and rewriting.

---

# LangGraph Workflow

LangGraph is used to orchestrate the retrieval and generation pipeline.

The main workflow nodes are:

### Retrieve

Retrieves relevant document chunks using the hybrid retriever.

### Grade

Evaluates whether the retrieved context is sufficient for answering the question.

### Rewrite

If the retrieved context is considered poor, the original question is rewritten and sent through retrieval again.

### Generate

The selected document chunks and associated metadata are passed to the generation pipeline.

The workflow can therefore follow either:

```text
Retrieve → Grade → Generate
```

or:

```text
Retrieve → Grade → Rewrite → Generate
```

The rewritten query is used for the second retrieval, while the original question is retained for final answer generation.

---

# Answer Generation

The final retrieved chunks are inserted into a grounded prompt.

The prompt instructs the model to:

- Answer only from the provided context
- Avoid unsupported information
- State when sufficient information cannot be found
- Include source document names

The final response is generated using:

**Gemini 2.5 Flash**

---

# Storage

InsightRAG maintains two searchable data structures.

## ChromaDB

ChromaDB is used for persistent vector storage.

It stores:

- Document chunks
- Embedding vectors
- Source information
- Chunk IDs

The project uses a persistent ChromaDB client.

## BM25

The BM25 corpus is persisted at:

```text
data/bm25_corpus.pkl
```

It stores the indexed documents and their tokenized representations.

Both indexes are created during document ingestion and are later accessed by the `HybridRetriever` during question answering.

---

# Project Structure

```text
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
```

## Module Responsibilities

| Module | Responsibility |
|---|---|
| `loader.py` | Extracts text from PDFs using PyMuPDF |
| `chunker.py` | Splits extracted text into overlapping chunks |
| `embeddings.py` | Generates BGE document/query embeddings |
| `vectordb.py` | Stores and searches embeddings using ChromaDB |
| `retrieval.py` | Implements vector, BM25, hybrid retrieval, and reranking |
| `generator.py` | Builds the generation pipeline |
| `llm.py` | Interfaces with Gemini 2.5 Flash |
| `prompts.py` | Builds grounded prompts using retrieved context |
| `corrective_rag.py` | Determines whether retrieval needs correction |
| `rag_nodes.py` | Defines LangGraph node operations |
| `rag_graph.py` | Defines and compiles the LangGraph workflow |
| `rag_state.py` | Defines the state passed through the workflow |
| `rag_service.py` | Connects indexing, retrieval, generation, and graph components |
| `app2.py` | Streamlit user interface |

---

# Technology Stack

| Component | Technology |
|---|---|
| Language | Python |
| UI | Streamlit |
| PDF Processing | PyMuPDF |
| Text Chunking | LangChain `RecursiveCharacterTextSplitter` |
| Embeddings | `BAAI/bge-small-en-v1.5` |
| Vector Database | ChromaDB |
| Lexical Retrieval | BM25 |
| Reranking | `BAAI/bge-reranker-base` |
| Workflow | LangGraph |
| LLM | Gemini 2.5 Flash |
| Evaluation | Ragas |

---

# Evaluation

The system was evaluated by comparing a baseline RAG pipeline against the corrective RAG pipeline.

| Metric | Baseline RAG | Corrective RAG |
|---|---:|---:|
| Answer Correctness | 0.2647 | **0.5137** |
| Faithfulness | 1.0000 | **1.0000** |

The evaluation showed an improvement in **answer correctness** after introducing the corrective retrieval workflow.

The evaluation used generated expected answers and Ragas-based evaluation. Reference contexts were not available in the evaluation dataset, so context-based metrics were not used.

---

# Example

A user can upload a research paper and ask:

> **What is the sample size used in the regression analysis?**

InsightRAG processes the question through the following pipeline:

```text
User Question
     │
     ▼
Vector Retrieval ──────┐
                       │
BM25 Retrieval ────────┤
                       ▼
                Combine Candidates
                       │
                       ▼
                 CrossEncoder
                   Reranking
                       │
                       ▼
                     Grade
                    /     \
                 Good     Poor
                  │         │
                  │      Rewrite
                  │         │
                  │         ▼
                  │   Hybrid Retrieval
                  │         │
                  └────┬────┘
                       ▼
                   Gemini 2.5
                       │
                       ▼
                Answer + Sources
```

---

# Installation

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd InsightRAG
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

### 3. Activate the environment

**Windows:**

```bash
venv\Scripts\activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Configure the API key

Create a `.env` file:

```env
GEMINI_API_KEY=your_api_key
```

---

# Running the Application

Start the Streamlit application:

```bash
streamlit run app2.py
```

Then:

1. Upload a PDF.
2. Index the document.
3. Ask a question.
4. InsightRAG performs hybrid retrieval.
5. The retrieved context is reranked and graded.
6. If necessary, the question is rewritten and retrieval is repeated.
7. Gemini generates a grounded answer.
8. The UI displays the answer and sources.

---

# Design Decisions

## Why embeddings?

Embeddings allow the system to retrieve semantically related information rather than relying only on exact keyword matches.

## Why BM25?

BM25 provides strong lexical retrieval and is particularly useful for exact terms, names, and technical keywords that semantic retrieval may miss.

## Why a CrossEncoder?

Initial retrieval needs to be relatively efficient and broad. Since CrossEncoder scoring is more computationally expensive, it is applied only to the smaller retrieved candidate set to improve ranking precision.

## Why LangGraph?

The project contains conditional workflow logic involving retrieval, grading, rewriting, and generation.

LangGraph makes these steps explicit and stateful instead of implementing the entire workflow through deeply nested conditional code.

## Why Corrective RAG?

A standard RAG pipeline assumes that the first retrieval is good enough.

InsightRAG introduces a correction step:

```text
Retrieve
   ↓
Grade
   ↓
Rewrite if necessary
   ↓
Retrieve Again
   ↓
Generate
```

This provides an additional opportunity to improve retrieval before the LLM generates the final response.

---

# Current Limitations

InsightRAG is currently a **project/research prototype rather than a production-scale service**.

Current limitations include:

- The corrective workflow performs only one correction.
- PDF page-level metadata is not preserved by the current loader.
- BM25 tokenization uses simple lowercase whitespace splitting.
- API retry, timeout, and fallback handling are not extensively implemented.
- The current application is designed around a Streamlit interface.
- Persistent indexes require explicit reset/management when changing document sets.
- Evaluation uses generated expected answers rather than a fully human-annotated benchmark.
- The current implementation does not provide production-grade multi-user isolation or distributed scaling.

---

# Future Improvements

Potential improvements include:

- Multi-step bounded corrective retrieval
- Page-level metadata and citations
- Better BM25 preprocessing
- Human-verified evaluation datasets
- Retrieval metrics such as Recall@K and MRR
- API retry and exponential backoff
- Rate limiting
- Structured logging and monitoring
- User/document-level access isolation
- Asynchronous document processing
- Scalable external vector storage
- Background processing for large PDF files
- Separation of the production API from the UI

---

# Core Idea

The central idea behind InsightRAG is:

> **Don't immediately ask the LLM to answer. First make sure the retrieved context is good enough.**

Instead of:

```text
Question
   ↓
Vector Search
   ↓
LLM
```

InsightRAG uses:

```text
Question
   ↓
Hybrid Retrieval
   ↓
Combine
   ↓
Rerank
   ↓
Grade
   │
   ├── Good ──→ Generate
   │
   └── Poor ──→ Rewrite
                    ↓
               Retrieve Again
                    ↓
                 Generate
```

This combines **hybrid retrieval, reranking, and corrective retrieval** before answer generation.

---

# End-to-End Summary

## Indexing Phase

```text
PDF
 ↓
PyMuPDF
 ↓
Text Extraction
 ↓
Chunking
 ↓
 ┌───────────────┐
 │               │
 ▼               ▼
BGE Embeddings  BM25
 │               │
 ▼               ▼
ChromaDB        BM25 Corpus
```

## Query Phase

```text
Question
 ↓
LangGraph
 ↓
Hybrid Retrieval
 ├── ChromaDB Vector Search
 └── BM25 Search
 ↓
Combine Results
 ↓
CrossEncoder Reranking
 ↓
Grade
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
```

---

## Author

**Susmita Haldar**
