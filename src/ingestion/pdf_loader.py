from io import BytesIO

from langchain_core.documents import Document
from pypdf import PdfReader

from src.config import MAX_TEXT_CHARS
from src.errors import FinSightError
from .validation import validate_upload


def load_pdf(data: bytes, filename: str) -> list[Document]:
    filename, _ = validate_upload(data, filename, {".pdf"})
    try:
        reader = PdfReader(BytesIO(data))
        if reader.is_encrypted and not reader.decrypt(""):
            raise FinSightError("Password-protected PDF is unsupported. Upload a decrypted copy you are authorized to use.")
        if len(reader.pages) > 500:
            raise FinSightError("PDF exceeds the 500-page limit.")
        documents = []
        total = 0
        empty_pages = 0
        for number, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            if not text:
                empty_pages += 1
                continue
            total += len(text)
            if total > MAX_TEXT_CHARS:
                raise FinSightError("Extracted text exceeds the one-million-character limit. Split the document.")
            documents.append(Document(page_content=text, metadata={"filename": filename, "page": number, "kind": "pdf"}))
        if not documents:
            raise FinSightError("No extractable text found. Scanned/image-only PDFs are unsupported; OCR is not included.")
        for document in documents:
            document.metadata.update(total_pages=len(reader.pages), empty_pages=empty_pages)
        return documents
    except FinSightError:
        raise
    except Exception as exc:
        raise FinSightError("Cannot read this PDF. It may be corrupt, encrypted, or unsupported.") from exc
