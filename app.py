import streamlit as st
from pathlib import Path
from modules.rag_service import RAGService

@st.cache_resource
def get_rag_service():
    return RAGService()

st.set_page_config(
    page_title="InsightRAG",
    page_icon="🐧",
    layout="wide"
)

st.title(" InsightRAG")
st.caption("Self-Correcting RAG Assistant")

service = get_rag_service()
if "indexed_files" not in st.session_state:
    st.session_state.indexed_files = set()

uploaded_files = st.file_uploader(
    "Upload PDF(s)",
    type=["pdf"],
    accept_multiple_files=True
)

if uploaded_files:
    
    st.success(f"{len(uploaded_files)} PDF(s) selected.")

    if st.button("Index Documents"):

        total_chunks = 0

        with st.spinner("Indexing documents..."):

            for uploaded_file in uploaded_files:
                if uploaded_file.name in st.session_state.indexed_files:
                    continue

                save_path = Path("uploads") / uploaded_file.name

                with open(save_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                chunks = service.index_document(save_path)
                total_chunks += chunks
                st.session_state.indexed_files.add(uploaded_file.name)

        st.success("All documents indexed successfully!")

        st.write(f"Total Chunks Indexed: {total_chunks}")
        st.write(f"Total Indexed Documents: {service.db.count()}")

    st.divider()

    st.subheader("Ask Questions")

    question = st.text_input("Enter your question")

    if st.button("Retrieve Context"):

        answer, results = service.answer(question)
        st.subheader(" Answer")
        st.write(answer)
        st.divider()

        documents = [
            item["document"]
            for item in results
        ]
        metadatas = [
            item["metadata"]
            for item in results
        ]
        st.success(f"Retrieved {len(documents)} chunks")

        for i, (doc, meta) in enumerate(zip(documents, metadatas), start=1):

            with st.expander(f"Chunk {i}"):

                st.write("Source:", meta["source"])
                st.write("Chunk ID:", meta["chunk_id"])

                st.write(doc)