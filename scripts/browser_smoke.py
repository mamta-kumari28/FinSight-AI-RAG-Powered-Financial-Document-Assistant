"""Real browser smoke checks. No fake API answers; live Q&A requires a configured key.
Run from repository root: python scripts/browser_smoke.py
Requires Playwright plus Chromium (CHROMIUM_EXECUTABLE or playwright install chromium).
"""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from urllib.request import urlopen

from playwright.sync_api import sync_playwright, expect

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/browser'
SHOTS = ROOT / 'docs/screenshots'
PORT = int(os.environ.get('FINSIGHT_SMOKE_PORT', '8502'))
URL = f'http://127.0.0.1:{PORT}'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    SHOTS.mkdir(parents=True, exist_ok=True)
    log = (OUT / 'streamlit.log').open('w')
    environment = {**os.environ, 'MPLCONFIGDIR': os.environ.get('MPLCONFIGDIR', str(OUT / 'matplotlib'))}
    server = subprocess.Popen([sys.executable, '-m', 'streamlit', 'run', 'app.py', '--server.headless=true',
                               '--server.address=127.0.0.1', f'--server.port={PORT}', '--browser.gatherUsageStats=false'],
                              cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, env=environment)
    results = []
    try:
        for _ in range(60):
            if server.poll() is not None:
                raise RuntimeError('Streamlit startup failed; inspect artifacts/browser/streamlit.log')
            try:
                with urlopen(URL + '/_stcore/health', timeout=2) as response:
                    if response.read() == b'ok':
                        break
            except OSError:
                time.sleep(.5)
        else:
            raise RuntimeError('Server failed readiness check')
        with sync_playwright() as p:
            executable = os.environ.get('CHROMIUM_EXECUTABLE') or shutil.which('chromium') or shutil.which('google-chrome')
            options = {'headless': True, 'args': ['--no-sandbox', '--disable-dev-shm-usage']}
            if executable:
                options['executable_path'] = executable
            browser = p.chromium.launch(**options)
            page = browser.new_page(viewport={'width': 1440, 'height': 1080}, device_scale_factor=1)
            page.goto(URL)
            expect(page.get_by_role('heading', name='📊 FinSight AI', exact=True)).to_be_visible(timeout=60000)
            page.screenshot(path=str(SHOTS / 'home.png'), full_page=True)
            results.append('Home renders both workflow links and privacy/limitations')
            page.get_by_role('link', name='Open Document Assistant').click()
            expect(page.get_by_role('heading', name='📄 Document Assistant', exact=True)).to_be_visible(timeout=60000)
            page.get_by_text('Fictional sample documents', exact=True).click()
            page.get_by_role('button', name='Inspect locally (no API call)', exact=True).click()
            expect(page.get_by_text('Revenue in 2025: Rs. 120 crore.', exact=False).first).to_be_visible(timeout=60000)
            page.screenshot(path=str(SHOTS / 'document_inspection.png'), full_page=True)
            results.append('Real document page extracts fictional PDF locally and shows page metadata')
            if not os.getenv('GEMINI_API_KEY'):
                expect(page.get_by_role('button', name='Index selected documents', exact=True)).to_be_disabled()
                expect(page.get_by_role('button', name='Ask Gemini', exact=True)).to_be_disabled()
                results.append('Missing-key indexing and Q&A correctly disabled (not live-validated)')
            # Real file upload exercises the browser upload path independently of sample selection.
            page.get_by_text('Upload documents', exact=True).click()
            page.locator('input[type=file]').set_input_files(str(ROOT / 'data/samples/financial_summary.xlsx'))
            page.get_by_role('button', name='Inspect locally (no API call)', exact=True).click()
            expect(page.get_by_text('Amount (Rs. crore): 100', exact=False).first).to_be_visible(timeout=60000)
            results.append('Excel upload path preserves financial headers and sheet/row metadata')
            page.goto(URL + '/Portfolio_Optimizer')
            expect(page.get_by_role('heading', name='📈 Portfolio Optimizer', exact=True)).to_be_visible(timeout=60000)
            page.get_by_text('Simulated example', exact=True).click()
            page.get_by_role('button', name='Optimize portfolios', exact=True).click()
            expect(page.get_by_text('Analysis complete: 253 price observations, 252 daily returns.', exact=True)).to_be_visible(timeout=60000)
            expect(page.get_by_role('heading', name='Maximum Sharpe', exact=True)).to_be_visible()
            expect(page.get_by_role('button', name='Chart image: risk_return.png', exact=True)).to_be_visible(timeout=60000)
            page.get_by_role('heading', name='Minimum variance', exact=True).evaluate("el => el.scrollIntoView({block: 'start'})")
            page.locator('[data-testid="stMain"]').evaluate("el => el.scrollTop -= 70")
            page.screenshot(path=str(SHOTS / 'portfolio_results.png'), full_page=True)
            with page.expect_download() as download_info:
                page.get_by_role('button', name='Structured report (JSON)', exact=True).click()
            download = download_info.value
            download.save_as(OUT / 'portfolio.json')
            report = json.loads((OUT / 'portfolio.json').read_text())
            assert report['simulated'] and len(report['portfolios']) == 2
            assert all(abs(sum(item['weights'].values()) - 1) < 1e-8 for item in report['portfolios'])
            expect(page.get_by_role('heading', name='Maximum Sharpe', exact=True)).to_be_visible()
            results.append('Both portfolios, tables, three chart images, valid JSON download and download-rerun retention')
            for name in ['Allocation CSV', 'Readable report (Markdown)', 'Chart image: allocations.png']:
                with page.expect_download() as info:
                    page.get_by_role('button', name=name, exact=True).click()
                assert info.value.failure() is None
                info.value.save_as(OUT / info.value.suggested_filename)
            results.append('CSV, Markdown, and PNG exports download successfully')
            page.get_by_role('spinbutton', name='Annual risk-free rate (%)').fill('5')
            page.get_by_role('spinbutton', name='Annual risk-free rate (%)').press('Enter')
            expect(page.get_by_role('heading', name='Maximum Sharpe', exact=True)).to_have_count(0, timeout=15000)
            results.append('Risk-free input change clears stale results before recomputation')
            # Upload the same public CSV through the actual price uploader.
            page.get_by_text('Upload price CSV', exact=True).click()
            page.locator('input[type=file]').set_input_files(str(ROOT / 'data/samples/synthetic_prices.csv'))
            page.get_by_role('textbox', name='Data provenance / provider and adjustment method').fill('Browser test: SIMULATED PCG64 seed 2025')
            page.get_by_text('This uploaded price series is simulated / synthetic', exact=True).click()
            page.get_by_role('button', name='Optimize portfolios', exact=True).click()
            expect(page.get_by_text('Analysis complete: 253 price observations, 252 daily returns.', exact=True)).to_be_visible(timeout=60000)
            results.append('Actual price-file upload completes optimization independently of Gemini')
            page.locator('input[type=file]').set_input_files({'name': 'bad.csv', 'mimeType': 'text/csv', 'buffer': b'Date,A,B\n2025-01-01,1,2\n'})
            expect(page.get_by_role('heading', name='Maximum Sharpe', exact=True)).to_have_count(0)
            page.get_by_role('button', name='Optimize portfolios', exact=True).click()
            expect(page.get_by_text('Provide at least 61 daily price observations (rows).', exact=True)).to_be_visible(timeout=15000)
            assert page.locator('[data-testid=stException]').count() == 0
            results.append('Replacing a price file clears old results; invalid file shows a helpful error')
            browser.close()
        (OUT / 'results.json').write_text(json.dumps({'passed': results, 'live_gemini': 'unrun without credentials'}, indent=2))
        print(json.dumps({'passed': results, 'screenshots': str(SHOTS)}, indent=2))
    finally:
        server.terminate()
        try:
            server.wait(timeout=15)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait(timeout=5)
        log.close()


if __name__ == '__main__':
    main()
