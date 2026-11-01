from io import BytesIO, StringIO

import pandas as pd
from langchain_core.documents import Document
from openpyxl import load_workbook

from src.config import MAX_TABLE_ROWS, MAX_TEXT_CHARS
from src.errors import FinSightError
from .validation import read_csv_text, validate_headers, validate_upload, validate_xlsx_archive


def load_table(data: bytes, filename: str) -> list[Document]:
    filename, extension = validate_upload(data, filename, {".csv", ".xlsx"})
    try:
        if extension == ".csv":
            text, headers = read_csv_text(data)
            frames = {"CSV": pd.read_csv(StringIO(text), dtype=str, keep_default_na=False)}
            frames["CSV"].columns = headers
        else:
            validate_xlsx_archive(data)
            workbook = load_workbook(BytesIO(data), read_only=True, data_only=True)
            try:
                headers_by_sheet = {
                    sheet.title: validate_headers(next(sheet.iter_rows(values_only=True), ()))
                    for sheet in workbook.worksheets
                }
            finally:
                workbook.close()
            frames = pd.read_excel(BytesIO(data), sheet_name=None, dtype=str, keep_default_na=False, engine="openpyxl")
            for sheet, frame in frames.items():
                frame.columns = headers_by_sheet[sheet]
        if sum(len(frame) for frame in frames.values()) > MAX_TABLE_ROWS:
            raise FinSightError("Document exceeds the 20,000-row limit. Split the table.")
        documents = []
        total = 0
        for sheet, frame in frames.items():
            for row_number, row in enumerate(frame.itertuples(index=False, name=None), start=2):
                if not any(str(value).strip() for value in row):
                    continue
                fields = " | ".join(f"{header}: {value}" for header, value in zip(frame.columns, row))
                text = f"File: {filename}; Sheet: {sheet}; Row: {row_number}\n{fields}"
                total += len(text)
                if total > MAX_TEXT_CHARS:
                    raise FinSightError("Extracted table text exceeds the one-million-character limit.")
                documents.append(Document(page_content=text, metadata={"filename": filename, "sheet": sheet, "row_start": row_number, "row_end": row_number, "kind": "table"}))
        if not documents:
            raise FinSightError("The document contains no nonempty data rows.")
        return documents
    except FinSightError:
        raise
    except Exception as exc:
        raise FinSightError("Cannot read this table. Check the format, header row, and workbook contents.") from exc
