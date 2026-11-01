"""Validated document ingestion; independent from portfolio price ingestion."""
from .pdf_loader import load_pdf
from .tabular_loader import load_table
from .validation import validate_upload


def load_document(data: bytes, filename: str):
    name, suffix = validate_upload(data, filename, {".pdf", ".csv", ".xlsx"})
    return load_pdf(data, name) if suffix == ".pdf" else load_table(data, name)
