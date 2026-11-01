from io import BytesIO

import pandas as pd
import pytest
from langchain_core.documents import Document
from pypdf import PdfWriter

from src.errors import FinSightError
from src.ingestion import load_document
from src.ingestion.pdf_loader import load_pdf
from src.ingestion.tabular_loader import load_table
from src.ingestion.text_splitter import split_documents
from src.ingestion.validation import validate_upload


def test_pdf_facts_and_pages(samples):
    docs = load_pdf((samples / 'abc_financial_report.pdf').read_bytes(), 'abc.pdf')
    assert len(docs) == 1
    assert 'Revenue in 2024: Rs. 100 crore' in docs[0].page_content
    assert 'Revenue in 2025: Rs. 120 crore' in docs[0].page_content
    assert 'Net profit in 2025: Rs. 15 crore' in docs[0].page_content
    assert docs[0].metadata['page'] == 1
    assert docs[0].metadata['filename'] == 'abc.pdf'


def test_multipage_pdf_and_chunk_metadata(samples):
    docs = load_pdf((samples / 'abc_long_report.pdf').read_bytes(), 'long.pdf')
    assert [d.metadata['page'] for d in docs] == [1, 2, 3, 4]
    chunks = split_documents(docs)
    assert len(chunks) > len(docs)
    assert {c.metadata['page'] for c in chunks} == {1, 2, 3, 4}
    assert all(len(c.page_content) <= 1000 and c.metadata['filename'] == 'long.pdf' for c in chunks)
    assert all('start_index' in c.metadata for c in chunks)


def test_chunk_overlap_and_no_metadata_mutation():
    original = Document(page_content=' '.join(f'token{i:04d}' for i in range(400)), metadata={'page': 9, 'filename': 'file.pdf'})
    chunks = split_documents([original], 1000, 200)
    assert len(chunks) > 2
    assert chunks[1].metadata['start_index'] < len(chunks[0].page_content)
    assert chunks[0].page_content[-150:] in chunks[1].page_content
    assert original.metadata == {'page': 9, 'filename': 'file.pdf'}


@pytest.mark.parametrize('size,overlap', [(0, 0), (100, 100), (100, -1)])
def test_bad_chunk_configuration(size, overlap):
    with pytest.raises(FinSightError):
        split_documents([], size, overlap)


def test_encrypted_pdf():
    writer = PdfWriter()
    writer.add_blank_page(100, 100)
    writer.encrypt('private-password')
    stream = BytesIO()
    writer.write(stream)
    with pytest.raises(FinSightError, match='Password-protected'):
        load_pdf(stream.getvalue(), 'locked.pdf')


def test_image_only_pdf():
    writer = PdfWriter()
    writer.add_blank_page(100, 100)
    stream = BytesIO()
    writer.write(stream)
    with pytest.raises(FinSightError, match='OCR'):
        load_pdf(stream.getvalue(), 'scan.pdf')


@pytest.mark.parametrize('data,name', [(b'garbage', 'bad.pdf'), (b'', 'empty.pdf'), (b'a,b\n1,2', 'fake.exe'), (b'not a zip', 'bad.xlsx')])
def test_rejected_documents(data, name):
    with pytest.raises(FinSightError):
        load_document(data, name)


def test_upload_size_and_safe_filename():
    with pytest.raises(FinSightError, match='20 MB'):
        validate_upload(b'a' * (20 * 1024 * 1024 + 1), 'big.csv', {'.csv'})
    assert validate_upload(b'a,b', '../../secret.csv', {'.csv'})[0] == 'secret.csv'


def test_csv_references(samples):
    docs = load_table((samples / 'financial_summary.csv').read_bytes(), 'financial_summary.csv')
    assert len(docs) == 3
    assert docs[0].metadata['row_start'] == 2
    assert 'Year: 2024' in docs[0].page_content
    assert 'Amount (Rs. crore): 100' in docs[0].page_content
    assert docs[-1].metadata['row_end'] == 4


def test_excel_sheets_and_rows(samples):
    docs = load_table((samples / 'financial_summary.xlsx').read_bytes(), 'financial_summary.xlsx')
    assert {d.metadata['sheet'] for d in docs} == {'Financials', 'Notes'}
    note = next(d for d in docs if d.metadata['sheet'] == 'Notes')
    assert note.metadata['row_start'] == 2
    assert 'Disclosure:' in note.page_content


@pytest.mark.parametrize('data', [b'a,a\n1,2', b'a,b\n1,2,3', b'a,b\n', b'\xff\xfe\x01', b'a,\n1,2'])
def test_bad_csv(data):
    with pytest.raises(FinSightError):
        load_table(data, 'bad.csv')


def test_excel_duplicate_headers():
    stream = BytesIO()
    pd.DataFrame([[1, 2]], columns=['A', 'A']).to_excel(stream, index=False)
    with pytest.raises(FinSightError, match='Duplicate'):
        load_table(stream.getvalue(), 'bad.xlsx')
