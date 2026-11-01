# Demo Guide

Run all commands from the repository root using Python 3.12. Start with `python -m streamlit run app.py` from an environment with `requirements.txt` installed. See the README for direct Windows/macOS/Linux virtual-environment commands.

## Offline demo: available without Gemini

1. Open the home page. Explain that document research and portfolio research are independent.
2. Open Document Assistant, choose **Fictional sample documents**, and click **Inspect locally (no API call)**. The one-page ABC report shows page metadata and extracted figures. Choose the longer PDF to demonstrate multiple source pages, or CSV/Excel to show headers/sheets/rows. This does not index or generate answers.
3. Open Portfolio Optimizer and choose **Simulated example**. Read the simulated-data warning. Use a 4% annual risk-free rate and click **Optimize portfolios**.
4. Inspect both allocation columns and expected-return/volatility/Sharpe metrics, the allocation chart, rebased price chart, and historical risk–return comparison.
5. Download allocation CSV, Markdown research report, JSON report, and PNG charts. Results should remain visible after downloads.
6. Change the risk-free rate: old results must clear before recomputation. Upload an invalid or too-short price CSV: no old results should be shown for it.
7. Explain that prices are simulated and the illustration is in-sample, not a backtest or forecast.

## Live document demo: prerequisites

Configure your own key securely in `.env` or deployment secrets. Use `GEMINI_API_KEY`, `GEMINI_CHAT_MODEL`, and `GEMINI_EMBEDDING_MODEL`. Permit HTTPS to `generativelanguage.googleapis.com` if network access is restricted. Documented defaults are suggestions; current access must be checked for your key. Do not show or reuse any key from a screenshot.

1. In **Gemini Settings**, list/validate available models. Both required API actions must be available. Metadata alone does not prove inference will work.
2. Choose `abc_financial_report.pdf`, click **Index selected documents**, and wait for a real completion message/chunk count. This sends fictional document text for embedding and consumes quota.
3. Leave the new index selected. Ask: “How much did revenue increase from 2024 to 2025, in rupees crore and percent? Show the calculation.” Expected document-supported arithmetic is 120 − 100 = 20 crore, 20 / 100 × 100 = 20%.
4. Ask: “What was net profit in 2025?” Expected fact: Rs. 15 crore. Open the retrieved evidence and verify the source page.
5. Ask: “What is revenue in 2030?” The document has no figure. Expect the insufficient-information response, not an extrapolated number.
6. Download Q&A Markdown/JSON with question, answer, sources and retrieved passages.
7. Index the same file again: it should reuse the existing collection without document re-embedding. New questions still consume API requests.
8. Use the four-page report for multiple chunks/page citations, or the workbook for sheet/row citations. Select which indexes to query. Remove an index or confirm/reset all session indexes and verify it is no longer selectable.

If a model returns 404, use the actual model catalog and change settings. If quota, authorization or network access fails, report the live demo as blocked; do not replace it with a canned answer. The application does not implement a fake “demo Gemini” mode.

## Validation commands

```sh
python -m pip install -r requirements-dev.txt
python -m pip check
python -m pytest --cov=src --cov-report=term-missing
# Explicit opt-in: real API usage, only with secure usable configuration
python -m pytest -m live --live -v tests/test_live_gemini.py
# Optional real-browser smoke test and actual screenshots
python scripts/browser_smoke.py
```

For browser smoke tests, install Chromium via Playwright or set `CHROMIUM_EXECUTABLE`. The script uses an isolated local server on port 8502 by default (`FINSIGHT_SMOKE_PORT` overrides it), tests actual upload/download paths, and stops only its own server. Generated browser artifacts are ignored. Screenshots under `docs/screenshots/` contain only public fictional/simulated samples. Do not capture private documents or secrets.

## Honest demo wording

Say “historical estimated return,” “simulated prices,” “retrieved evidence,” and “API validation pending” where applicable. Do not claim guaranteed returns, OCR, fine-tuning, production security, or live success based on mocked tests. See the validation report for what was actually executed in this implementation.
