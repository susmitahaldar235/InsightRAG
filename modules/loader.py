from pathlib import Path
import fitz  # PyMuPDF


class DocumentLoader:
    def load_pdf(self, pdf_path: Path) -> str:
        """
        Extract text from a PDF.

        Args:
            pdf_path: Path to the PDF file.

        Returns:
            Extracted text as a string.
        """
        document = fitz.open(pdf_path)

        text = ""

        for page in document:
            text += page.get_text()

        document.close()

        return text