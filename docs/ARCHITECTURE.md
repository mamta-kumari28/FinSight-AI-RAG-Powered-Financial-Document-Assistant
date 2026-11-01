# Architecture

FinSight AI is a single-process Streamlit research app with two independent workflows. It calls the Gemini Developer API only for document research. It does not need a database service or portfolio API.

## Modules actually used by the pages

| Module | Responsibility |
| --- | --- |
| `app.py` | Home, workflow navigation, privacy and research limitations |
| `pages/1_Document_Assistant.py` | Local inspection, secure settings, indexing/removal, evidence-grounded Q&A and report downloads |
| `pages/2_Portfolio_Optimizer.py` | Price input, provenance, risk-free input, analysis/results, chart/report downloads |
| `src/config.py` | Limits, defaults, `.env` / process / deployment-secret configuration; key excluded from object representation |
| `src/errors.py` | Expected user-facing errors and safe provider-error classification |
| `src/ingestion/validation.py` | Type/size checks, safe filenames, duplicate/missing headers, CSV shape checks, XLSX expansion limit |
| `src/ingestion/pdf_loader.py` | Page-wise PyPDF extraction, encrypted/corrupt/image-only handling, page/size limits |
| `src/ingestion/tabular_loader.py` | Pandas CSV/Excel extraction; each nonempty data row retains filename, sheet, and row number |
| `src/ingestion/text_splitter.py` | LangChain recursive splitting; 1,000 characters, 200 overlap, original source metadata and start offsets |
| `src/rag/embeddings.py` | Actual SDK model catalog, action validation, compatible retrieval-task embeddings and normalization |
| `src/rag/vector_store.py` | Explicit-embedding Chroma collections, index identities, reuse, persistence, retrieval, isolation, deletion/reset |
| `src/rag/prompts.py` | Evidence-only policy, untrusted-document boundary, citations, missing-information instructions |
| `src/rag/qa_chain.py` | LangChain structured Gemini output, evidence ID validation, safe insufficient-information behavior |
| `src/portfolio/price_loader.py` | Independent strict price-CSV input boundary; no silent material data cleaning |
| `src/portfolio/metrics.py` | Simple daily returns, annualized estimates, sample covariance, ridge and portfolio metrics |
| `src/portfolio/optimizer.py` | Real CVXPY problems, solver selection, status/residual/weight checks |
| `src/portfolio/analysis.py` | Orchestrates estimates, independent strategy outcomes, charts and reports |
| `src/reporting/charts.py` | Thread-locked Matplotlib/Agg PNG chart rendering; figures are closed after export |
| `src/reporting/report_generator.py` | Allocation CSV, readable Markdown and finite-value JSON; no credential fields |
| `src/state.py` | Length-delimited SHA-256 fingerprints and stale-result invalidation |

## Document data flow

1. Validate extensions (`.pdf`, `.csv`, `.xlsx`), nonempty content, 20 MB per-file size, and safe display filename. The page caps a batch at ten files / 40 MB. Text is capped at one million characters; PDFs at 500 pages; tables at 20,000 rows. XLSX ZIP contents are size-checked before reading.
2. Extract one LangChain `Document` per PDF page or nonempty table row. PDF pages are 1-based. Excel sheet names and rows are preserved; CSV references count logical records with the header as row 1. Header/value strings remain attached to table text.
3. Split within source units, never concatenate pages or sheets into a chunk that loses provenance. Short table rows remain short; long rows can produce multiple chunks with the same row reference.
4. Build a content/configuration identity from bytes SHA-256, sanitized filename, embedding signature, chunk size/overlap and schema version. The embedding signature covers provider/API, model, dimension, document/query task types and normalization.
5. Under a per-session file lock, reuse a valid existing collection without any document embedding request. New collections start as `building`; batched inserts use explicitly supplied Gemini vectors. Set `ready` only after all batches succeed. Failed builds are removed. An interrupted build is detected on the next attempt and requires reset.
6. The Chroma persistent client uses cosine HNSW configuration and disables anonymized telemetry. It has no default embedding function; it cannot silently download/use another embedding model. Text, source metadata and vectors live in the session's local SQLite/index files.
7. A question uses `RETRIEVAL_QUERY`; chunks use `RETRIEVAL_DOCUMENT`, both with the same configured model/dimension and L2 normalization. Query selected compatible collections, merge their results by cosine distance, retain at most four passages with distance <= 0.8. This cutoff is a heuristic, **not a calibrated relevance/confidence guarantee**. A rerun alone performs no embedding request.
8. Send the selected evidence as JSON data in a human message and the grounding/untrusted-input policy as a system message. LangChain's Google integration requests structured `AnswerPayload` output. Answer IDs must belong to retrieved evidence and appear in the answer; otherwise return the canonical insufficient-information response. No retrieved passages means no generation request.
9. Display the answer, cited source labels and expandable retrieved text, including uncited retrieved passages with their own evidence IDs. Download Q&A Markdown/JSON with question, answer, sources, model names, evidence and UTC timestamp. No API key appears in these reports.

Prompt instructions and citation validation cannot prove that a model's reasoning is sound. A plausible answer can still be wrong or misread evidence. Users must verify figures and calculations. No agent executes document instructions, tools, SQL, or code from the retrieved content.

## State and storage boundaries

- A random 128-bit browser-session namespace identifies `data/chroma/<session-id>`. The store validates its format before resolving the path. It is a convenience separation on a trusted local machine, not authentication, encryption, or tenant security.
- Streamlit holds uploads and result objects in session memory. Content/settings/selection/question changes invalidate stale answers. Price bytes, filename, provenance, simulation declaration, or risk-free rate changes invalidate portfolio results. Download clicks retain current results.
- Removing an index or resetting a session deletes its Chroma collections logically. SQLite/index remnants, backups, or old session directories may remain. Close the server before manually removing `data/chroma/` for full local cleanup. No forensic erasure claim is made.
- A new browser/server session may start a new namespace. Persistence means the store survives/reopens at the same namespace; it does not promise automatic document recovery across new sessions.
- API keys are read from a private environment/`.env` or Streamlit deployment secrets. Default models come from official SDK examples, and actual metadata validation is required before use. Provider errors are mapped to fixed helpful messages instead of reflecting potentially sensitive exception payloads.

## Portfolio mathematics

Let `P[t,i]` be an adjusted daily closing price. The application rejects incomplete, nonpositive or nonfinite observations, duplicate asset headers, invalid/duplicate/non-increasing dates, and fewer than 61 rows by default. Prices are not imputed, forward-filled, sorted silently, or fetched from a market provider.

`r[t,i] = P[t,i] / P[t-1,i] - 1`

`mu = 252 * mean(r)` and `Sigma_sample = 252 * cov(r, ddof=1)`.

The number of observations is N prices and N−1 returns. Annualized arithmetic mean is not compounded growth/CAGR. Volatility annualizes via the square root of annualized covariance. The 252 assumption requires a meaningful daily trading dataset; corporate-action adjustment and market calendars must be checked by the researcher.

A scale-aware ridge improves conditioning:

`epsilon = max(trace(Sigma_sample)/n, 1e-6) * 1e-8`

`Sigma = (Sigma_sample + Sigma_sample.T)/2 + epsilon*I`

Both solver and risk reporting use the adjusted matrix. Raw/regularized covariance and epsilon are exported so the adjustment is visible. The ridge stabilizes a singular estimate; it does not add information or cure poor data.

### Minimum variance

Minimize `w.T @ Sigma @ w` subject to `w >= 0` and `sum(w) = 1`. This is a convex quadratic program. Expected return does not enter this objective; changing the risk-free rate changes reported Sharpe but not the unconstrained minimum-variance allocation.

### Maximum Sharpe: positive-excess convex transformation

Let `a = mu - rf * 1`. For feasible portfolios with `a.T @ w > 0`, define `y = w / (a.T @ w)`. Then `a.T @ y = 1`, `y >= 0`. Maximizing Sharpe is equivalent to minimizing `y.T @ Sigma @ y` under those constraints. Recover `w = y / sum(y)`.

The derivation follows from `Sharpe(w) = (a.T w) / sqrt(w.T Sigma w)`; with this substitution `Sharpe = 1/sqrt(y.T Sigma y)`. At least one asset must have positive estimated excess return to make the transformed problem feasible. The application explicitly declines the unsupported all-nonpositive case and still shows minimum variance if it solved. It never submits a non-DCP ratio objective to CVXPY.

Available CLARABEL, OSQP and SCS solvers are checked in that order. Only `optimal` is accepted (not `optimal_inaccurate`). Outputs must be finite, nonnegative within tolerance, satisfy the investment/transformation residuals and normalize to one. Tiny negative roundoff within 1e-7 is clipped; there is no fallback portfolio on failure.

`expected_return = mu.T @ w`; `volatility = sqrt(w.T @ Sigma @ w)`; `Sharpe = (expected_return - rf) / volatility`.

Risk-free input is annual and in arithmetic-return units. Models assume long-only, full investment, no leverage/shorting, no trading costs/taxes/slippage, and no out-of-sample evaluation. Neither portfolio is a backtest or forecast.

## Validation design

Offline tests exercise source extraction, metadata, chunk overlap, real Chroma indexing/persistence/isolation/reset using an explicit test embedder, API adapters with clearly labeled mocks, analytic optimizer solutions and grid comparisons, solver failures, reports, input identities and actual Streamlit page execution. Real browser checks cover uploads, download bytes, rendering, error handling and stale results. Live tests are separate opt-in Gemini calls; skipped live tests never count as live validation. See `docs/VALIDATION_REPORT.md`.
