from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import CHUNK_OVERLAP, CHUNK_SIZE
from src.errors import FinSightError


def split_documents(documents: list[Document], chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP) -> list[Document]:
    if chunk_size <= 0 or not 0 <= chunk_overlap < chunk_size:
        raise FinSightError("Chunk overlap must be nonnegative and smaller than chunk size.")
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, add_start_index=True)
    chunks = splitter.split_documents(documents)
    for number, chunk in enumerate(chunks):
        chunk.metadata["chunk_number"] = number
    return chunks
