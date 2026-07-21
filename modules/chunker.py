from langchain_text_splitters import RecursiveCharacterTextSplitter


class DocumentChunker:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
            separators=["\n\n", "\n", ".", " ", ""]
        )

    def chunk_text(self, text: str):
        """
        Split text into overlapping chunks.
        """
        return self.text_splitter.split_text(text)