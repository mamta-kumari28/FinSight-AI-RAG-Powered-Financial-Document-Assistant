import csv
from io import BytesIO, StringIO
from pathlib import PurePosixPath
from zipfile import BadZipFile, ZipFile

from src.config import MAX_UPLOAD_BYTES
from src.errors import FinSightError


def validate_upload(data: bytes, filename: str, extensions: set[str]) -> tuple[str, str]:
    name = PurePosixPath(filename.replace("\\", "/")).name
    if not name or len(name) > 180 or any(ord(c) < 32 for c in name):
        raise FinSightError("Use a valid filename of at most 180 characters.")
    suffix = PurePosixPath(name).suffix.lower()
    if suffix not in extensions:
        raise FinSightError("Unsupported format. Allowed formats: " + ", ".join(sorted(extensions)))
    if not data or not data.strip():
        raise FinSightError("The uploaded file is empty.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise FinSightError("File exceeds the 20 MB upload limit.")
    return name, suffix


def validate_headers(headers) -> list[str]:
    values = [str(v).strip() if v is not None else "" for v in headers]
    if not values or any(not h for h in values):
        raise FinSightError("Every column must have a nonempty header.")
    if len(set(values)) != len(values):
        raise FinSightError("Duplicate column headers are not allowed.")
    return values


def read_csv_text(data: bytes) -> tuple[str, list[str]]:
    try:
        text = data.decode("utf-8-sig")
        rows = csv.reader(StringIO(text), strict=True)
        headers = validate_headers(next(rows))
        for row in rows:
            if len(row) != len(headers):
                raise FinSightError("CSV rows must all have the same number of fields as the header.")
        return text, headers
    except (UnicodeError, csv.Error, StopIteration) as exc:
        raise FinSightError("Unreadable CSV. Use a UTF-8, comma-separated file with a header row.") from exc


def validate_xlsx_archive(data: bytes) -> None:
    try:
        with ZipFile(BytesIO(data)) as archive:
            if sum(item.file_size for item in archive.infolist()) > 80 * 1024 * 1024:
                raise FinSightError("Excel workbook expands beyond the 80 MB safety limit.")
    except BadZipFile as exc:
        raise FinSightError("Unreadable Excel workbook. Upload a valid .xlsx file.") from exc
