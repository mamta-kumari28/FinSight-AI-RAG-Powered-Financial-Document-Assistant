"""Regenerate only public fictional samples; no network/API calls."""
from datetime import datetime
from pathlib import Path
import re
from zipfile import ZipFile, ZipInfo, ZIP_DEFLATED
from io import BytesIO

import numpy as np
import pandas as pd
from openpyxl import Workbook
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import simpleSplit
from reportlab.pdfgen import canvas

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'data/samples'


def pdf(filename, pages):
    stream = BytesIO()
    doc = canvas.Canvas(stream, pagesize=A4, invariant=1)
    doc.setTitle('ABC Limited - Fictional Financial Report')
    for number, paragraphs in enumerate(pages, 1):
        doc.setFont('Helvetica-Bold', 16)
        doc.drawString(48, 790, 'ABC Limited | Fictional Financial Report')
        doc.setFont('Helvetica', 10)
        y = 760
        for paragraph in paragraphs:
            for line in simpleSplit(paragraph, 'Helvetica', 10, 495):
                doc.drawString(48, y, line)
                y -= 15
            y -= 12
        doc.drawString(48, 35, f'FICTIONAL EDUCATIONAL SAMPLE - Page {number}')
        doc.showPage()
    doc.save()
    (OUT / filename).write_bytes(stream.getvalue())


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    summary = [
        'Company: ABC Limited. This is a fictional company used only for educational research.',
        'Revenue in 2024: Rs. 100 crore.',
        'Revenue in 2025: Rs. 120 crore.',
        'Net profit in 2025: Rs. 15 crore.',
        'All figures are fictional. The report contains historical illustrative figures for 2024 and 2025 only. No revenue forecast is provided.',
    ]
    pdf('abc_financial_report.pdf', [summary])
    pages = [summary + [
        'Reporting basis: This fictional financial review describes a small illustrative business. Amounts are in Indian rupees crore unless stated otherwise. These are not audited financial statements and do not relate to a listed company.',
        'Scope: Revenue means gross sales recorded during the specified illustrative year. Net profit refers to the bottom-line earnings figure for 2025. Comparisons must use the year labels attached to each figure. Readers should cite the page containing the source figures.',
        'The report has no information about revenue in 2030. Future performance cannot be inferred from these historical numbers. Any question requiring an undisclosed figure should receive an insufficient-information response.',
    ]]
    topics = [
        ('Operations and working capital', 'Inventory planning, supplier lead times, and customer collections influence working capital. Management reviews receivables at each month end. This description contains no quantified working-capital forecast.'),
        ('Risk and governance', 'The fictional business faces demand changes, input-cost volatility, and customer concentration. The board reviews risk registers quarterly. These controls do not guarantee any financial outcome or investment return.'),
        ('Accounting and research notes', 'The figures in this educational report are illustrative and use consistent year labels. The document contains no share-price information, investment recommendation, or model training data. Verify all calculations against the original source page.'),
    ]
    for topic, text in topics:
        paragraphs = [topic]
        for section in range(1, 9):
            paragraphs.append(f'Research note {section}: {text} A reader should distinguish a qualitative description from a disclosed numeric fact and should not invent missing amounts.')
        pages.append(paragraphs)
    pdf('abc_long_report.pdf', pages)
    frame = pd.DataFrame({'Company': ['ABC Limited'] * 3, 'Year': [2024, 2025, 2025],
                          'Metric': ['Revenue', 'Revenue', 'Net profit'], 'Amount (Rs. crore)': [100, 120, 15]})
    frame.to_csv(OUT / 'financial_summary.csv', index=False)
    wb = Workbook()
    wb.active.title = 'Financials'
    wb.active.append(list(frame.columns))
    for row in frame.itertuples(index=False, name=None):
        wb.active.append(row)
    notes = wb.create_sheet('Notes')
    notes.append(['Company', 'Disclosure'])
    notes.append(['ABC Limited', 'Fictional educational data. No forecasts supplied.'])
    wb.properties.created = wb.properties.modified = datetime(2025, 1, 1)
    temporary = BytesIO()
    wb.save(temporary)
    # Stabilize ZIP member timestamps for repeatable sample regeneration.
    with ZipFile(temporary) as source, ZipFile(OUT / 'financial_summary.xlsx', 'w', ZIP_DEFLATED) as target:
        for name in source.namelist():
            info = ZipInfo(name, date_time=(2025, 1, 1, 0, 0, 0))
            info.compress_type = ZIP_DEFLATED
            content = source.read(name)
            if name == 'docProps/core.xml':
                content = re.sub(rb'(<dcterms:modified[^>]*>).*?(</dcterms:modified>)', rb'\g<1>2025-01-01T00:00:00Z\g<2>', content)
            target.writestr(info, content)
    rng = np.random.Generator(np.random.PCG64(2025))
    noise = rng.normal(size=(252, 3)) @ np.linalg.cholesky(np.array([[1, .25, .10], [.25, 1, .15], [.10, .15, 1]])).T
    noise -= noise.mean(axis=0)
    returns = noise * np.array([.012, .016, .007]) + np.array([.12, .17, .08]) / 252
    values = np.vstack([np.ones(3), np.cumprod(1 + returns, axis=0)]) * np.array([100, 120, 80])
    prices = pd.DataFrame(values, index=pd.bdate_range('2024-01-02', periods=253), columns=['SIM_Alpha', 'SIM_Beta', 'SIM_Gamma'])
    prices.index.name = 'Date'
    prices.to_csv(OUT / 'synthetic_prices.csv', float_format='%.8f', date_format='%Y-%m-%d')
    print('Generated 2 fictional PDFs, CSV, Excel, and 253 simulated daily price observations.')


if __name__ == '__main__':
    main()
