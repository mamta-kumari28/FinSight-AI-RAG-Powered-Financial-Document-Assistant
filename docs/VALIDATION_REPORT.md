# Validation Report

Validation date: **2 November 2026 (Asia/Kolkata)**. Execution host timestamps use UTC. This report covers the implementation built in this checkout, starting from `ac6a82ebbf878a705c2a8ea94243b6e2c3b0e6b9`. No earlier workspace, old implementation, recovery bundle, or previous “49 tests” claim was used.

## Environment and installation

- Linux x86-64, Python **3.12.14**.
- A fresh repository-local `.venv` was created with `python3.12 -m venv .venv`.
- ` .venv/bin/python -m pip install --no-cache-dir -r requirements-dev.txt` completed successfully with the exact pinned runtime/development versions.
- Installation was repeated successfully; `.venv/bin/python -m pip check` reports **No broken requirements found**.
- The cloud environment also retains `/workspace/.venvs/finsight` with the same application/test stack. Linux-only `uvloop` and Windows-only `colorama` use platform markers. Windows and macOS execution have not been tested on those operating systems.
- Major installed versions: Streamlit 1.66.1, PyPDF 6.21.4, Pandas 3.1.0, NumPy 2.5.4, openpyxl 3.1.5, LangChain 1.4.7, langchain-google-genai 4.4.2, google-genai 2.34.0, ChromaDB 1.5.9, CVXPY 1.9.3, Matplotlib 3.11.2, pytest 9.2.0. Full transitive pins are in the requirement files.

## Final pytest run

From the repository root, in the fresh `.venv`:

```sh
MPLCONFIGDIR=/tmp/finsight-matplotlib .venv/bin/python -m pytest --cov=src --cov-report=term-missing --junitxml=/workspace/finsight-setup/final-pytest.xml
```

**86 passed, 2 skipped, 0 failed (88 collected), 19.86 seconds.** `src/` statement coverage: **93%** (652 statements, 45 uncovered). This is source-module coverage, not a model-accuracy measure or proof of live provider availability.

The two skipped tests are the separately marked live Gemini integration checks. A second explicit `-m live --live` invocation also skipped both because a usable `GEMINI_API_KEY` was not configured; no real API inference was performed.

Checks exercised:

- Text PDF extraction, sample financial facts and exact page references; multipage chunking; corrupt/encrypted/image-only PDF handling; upload limits.
- CSV headers/shape/row references; Excel sheets and rows; duplicate headers and unreadable/empty inputs.
- Chunk overlap, offsets and source metadata without mutating original documents.
- **Real Chroma** indexing, cosine retrieval, reopen/persistence, duplicate-index reuse, content/configuration identity, per-session isolation, removal/reset, failed partial build cleanup, stale indexes and dimension mismatch handling. These tests use a clearly labeled deterministic local **test embedder**, not live Gemini and not a downloaded model.
- Gemini SDK adapter construction and mocked retrieval-task/dimension/normalization/error contracts; actual installed SDK fields were inspected. These are not successful embedding calls.
- Structured answer citation guards, prompt untrusted-data boundary, insufficient-information behavior, and safe provider errors with clearly labeled mocked generation.
- Price validation (dates, duplicate assets, missing/nonnumeric/infinite/nonpositive prices, minimum observations, provenance); annualized return/covariance calculations.
- Real CVXPY minimum variance and maximum Sharpe against known analytic solutions and a simplex grid comparison, long-only/full-investment constraints, regularization, missing solvers, solver exceptions/nonconvergence, invalid solver output, transformation residuals, and unsupported excess-return cases.
- Full portfolio pipeline, reports/provenance/simulated labels, finite JSON, CSV formula-cell protection, and valid exported PNG signatures.
- Actual Streamlit page execution; input fingerprints; download-style rerun retention; clearing old results when inputs change. A document-page indexing/Q&A/reset integration test uses explicit API test doubles and real Chroma; it is not a live model demo.

Initial test failures were diagnosed and fixed: Chroma collection configuration was moved out of mutable metadata, and newly indexed documents are selected for Q&A before the selection widget renders. These failures are not hidden or described as live-provider successes. The final suite above has no unresolved offline failures.

## Actual application and browser checks

`python scripts/browser_smoke.py` started a real Streamlit process on an isolated local port, opened system Chromium through Playwright, exercised the UI and downloaded output bytes, then stopped only the server it started. Its final run succeeded with these nine check groups:

1. Home workflow navigation and privacy/limitations render.
2. The actual document page locally extracts the fictional PDF and shows page metadata.
3. With the key absent, live indexing and Q&A buttons are disabled with a clear configuration message.
4. An actual Excel upload preserves financial headers and sheet/row references.
5. The simulated portfolio example renders both optimized strategies, allocation table, three chart images, and a valid structured-report download; results remain after downloads.
6. Allocation CSV, readable Markdown report, and PNG chart exports download successfully.
7. A risk-free-rate change clears stale results before recomputation.
8. The actual price-file upload path solves both portfolios independently of Gemini.
9. Replacing the price file clears old results; a too-short CSV produces the expected validation error without exposing a raw traceback.

These checks go beyond imports, PIDs, open ports, or mocked HTTP responses. The server also returned its expected `ok` health response.

### Actual screenshots

- `docs/screenshots/home.png`: current home page.
- `docs/screenshots/document_inspection.png`: actual local PDF extraction/configuration state, not live indexing or generated Q&A.
- `docs/screenshots/portfolio_results.png`: actual simulated-data portfolio metrics, allocation table, and chart.

Screenshots were captured from Chromium, not generated or fabricated. A loading-state capture was replaced with a results capture after rendering completed. **Live indexing and Q&A-with-evidence screenshots are unavailable because no usable Gemini credential is configured.** No mock answer is presented as a screenshot of real inference.

## Reproducible public sample check

The generator was run repeatedly under pinned dependencies. SHA-256 hashes matched byte-for-byte for all five generated inputs: both PDFs, CSV, Excel, and simulated prices. XLSX archive/core-property timestamps are fixed to make regeneration repeatable. The price file contains 253 observations from 2024-01-02 to 2024-12-19. It has no real market data.

At a 4% annual risk-free rate, the actual solver outputs on that public simulated sample were approximately:

| Strategy | Estimated annual return | Annual volatility | Sharpe |
| --- | ---: | ---: | ---: |
| Minimum variance | 9.6285% | 10.2030% | 0.55165 |
| Maximum Sharpe | 11.8843% | 12.0757% | 0.65291 |

Both used CLARABEL and returned `optimal`, nonnegative weights summing to one. These are historical in-sample estimates on fictional data, not investment performance or guaranteed future returns.

## Live checks: unrun / blocked

- `GEMINI_API_KEY`, `GEMINI_CHAT_MODEL`, and `GEMINI_EMBEDDING_MODEL` had no injected binding at inspection; no local `.env` or Streamlit secrets file existed. The key remains unset. No key from any historical screenshot is used.
- The official Google Gen AI SDK README was retrieved from its official GitHub repository. It currently documents `gemini-flash-latest`, `gemini-embedding-001`, model listing and embedding calls. The installed SDK's model metadata fields, paired embedding tasks, constructor parameters and structured-output integration were inspected/constructed offline.
- Direct access to the Gemini documentation/API endpoint was denied by the current restricted proxy policy. Required domains were saved to the environment draft; saving a draft does not apply or publish it.
- No real account-specific model catalog, embeddings, retrieval over real Gemini vectors, generated answers, generated citation evidence, or missing-information response has been validated live. Offline mocked tests do not substitute for these checks.

To finish live verification: securely configure `GEMINI_API_KEY` in environment/deployment settings or a private local `.env`, allow HTTPS to `generativelanguage.googleapis.com`, inspect the actual model catalog, configure available model names, and run:

```sh
.venv/bin/python -m scripts.check_models
.venv/bin/python -m pytest -m live --live -v tests/test_live_gemini.py
```

These calls consume quota and may incur charges. Metadata support alone does not guarantee permission, dimensional support, successful inference, or quota. Live screenshots should be captured only after real success. If a model returns 404, use the catalog instead of guessing a historical model ID.

## Git and configuration boundaries

- Work was performed in the specified Mamta repository on local `main`; `.git` and existing history were retained.
- The original committed version is preserved by local branch `backup/pre-finsight-implementation-20261102`. The initial worktree was clean, with no uncommitted user files or credentials to move aside.
- GitHub HTTPS Git reads succeed. The GitHub account-identity API was initially forbidden by the network policy; token-variable presence is not used as proof of account identity or write permission. The final handoff reports the actual commit/push outcome.
- Source/private-data staging checks and normal fast-forward-only delivery are performed before upload. GitHub Actions is configured; no unobserved remote CI success is claimed.
- Reusable installation/start instructions and missing secure Gemini requirements are saved to the environment configuration draft. Review/save and publication are user-controlled actions; draft persistence is not publication or a new-task restoration test.

The code and offline development workflow are implemented and verified as above. End-to-end live document inference remains an external prerequisite, so this report does not claim that every requested live check is complete.
