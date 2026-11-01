# Interview Guide

Use these short, simple explanations as speaking notes. Describe only work you can demonstrate in this repository. The main explanations are in English; the optional Roman-Hindi introduction helps practice a Hinglish delivery.

## A 60-second introduction

“FinSight AI is an educational financial research application with two independent workflows. The first extracts text from PDFs, CSVs, and Excel files, splits it with LangChain, embeds it with Gemini, and retrieves relevant passages from a local Chroma index. Gemini answers using those passages, and the UI shows source pages or table rows so a reader can verify the answer. The second workflow validates daily historical prices and uses CVXPY to compare long-only minimum-variance and maximum-Sharpe portfolios. It exports allocations, risk/return metrics, charts, and research reports. I focused on source metadata, index reuse, input validation, numerical checks, and honest verification. This is not model training, trading, or a production multi-user service. Live Gemini results depend on a securely configured key, available models, quota, and network access.”

Optional Hinglish practice: “Is project mein do independent flows hain: financial documents se evidence ke saath Q&A, aur historical prices se portfolio analysis. RAG mein model ko train nahi kiya; relevant document passages retrieve karke context diya hai. Har answer ke sources dekh sakte hain. Portfolio ke liye CVXPY use hota hai, aur future return guarantee nahi kiya gaya.”

## Explain the actual workflow

**Document path:** `pages/1_Document_Assistant.py` → `src/ingestion/` → `src/rag/embeddings.py` → `src/rag/vector_store.py` → `src/rag/qa_chain.py` → `src/reporting/report_generator.py`.

PyPDF extracts one text unit per PDF page. Pandas reads tables; each row carries the filename, sheet, and row reference. LangChain makes overlapping chunks without losing that metadata. Gemini document embeddings go to the browser session's Chroma namespace. A query uses a compatible query embedding, retrieves the nearest passages, and sends them to Gemini as untrusted evidence. Structured output contains an answer and evidence IDs. The application rejects unknown/missing citations and displays sources and expandable passages.

**Portfolio path:** `pages/2_Portfolio_Optimizer.py` → `price_loader.py` → `metrics.py` → `optimizer.py` → `analysis.py` → `reporting/`.

A CSV is validated before calculation. The modules compute returns and annualized estimates, regularize covariance, solve the two optimization problems, validate weights, and build memory-only reports and PNG charts. Gemini is not used in this path.

## RAG versus fine-tuning

RAG means retrieval-augmented generation: provide relevant external passages at question time. It does not change model weights. Fine-tuning changes a model using a training process and dataset. This project calls an existing Gemini model; it does not train or fine-tune it. RAG makes it easier to update documents and attach references, but retrieval and generation can still fail.

## Chunking and overlap

The default is 1,000 characters with 200 overlap using `RecursiveCharacterTextSplitter`. Splitting keeps individual evidence units manageable. Overlap reduces the chance that a boundary separates connected context. It increases storage and embedding costs. These are initial engineering settings, not a benchmark-proven optimum. Chunks never merge different source pages or table rows. Short rows may produce a single small chunk.

## Embeddings and retrieval

An embedding is a numeric representation of text. Similar meaning can yield nearby vectors. The adapter uses the same configurable Gemini embedding model and output dimension (768) for both document and query vectors, with `RETRIEVAL_DOCUMENT` / `RETRIEVAL_QUERY` task types. It L2-normalizes vectors and verifies shape/finite values. Chroma uses cosine distance. Four best passages across selected indexes are retained with a heuristic maximum distance of 0.8. Distance is not a probability or correctness score. There is no local embedding-model download in the application or offline tests.

## Evidence, hallucination, and prompt injection

A citation identifies where a retrieved passage came from; it does not prove every sentence is supported. Wrong retrieval, missing evidence, arithmetic mistakes, or a model ignoring instructions can still produce errors. The prompt restricts answers to evidence and explicitly treats document text and questions as untrusted data. JSON evidence is placed in the human message, separate from system instructions. No tools execute document commands. Structured citation IDs are validated against actual retrieved IDs; missing/invalid support yields a canonical insufficient-information response. There is no claim of perfect hallucination prevention or production adversarial security.

For the fictional ABC sample, revenue moves from 100 to 120 crore. The expected calculation is `(120 - 100) = 20` crore and `20 / 100 * 100 = 20%`. Those are sample facts/calculations for testing, not rules hard-coded in the Q&A engine. The document does not disclose 2030 revenue.

## Chroma persistence and isolation

The collection identity combines a hash of the actual file bytes, filename, embedding configuration, chunk settings, and schema version. A ready matching index is reused without re-embedding its documents. A model/dimension/configuration mismatch is not silently reused. Failed partial builds are deleted; interrupted/stale builds require reset. Per-session file locks serialize index changes. Each browser session has a random namespace under `data/chroma/`. This is useful local separation, not authenticated multi-tenancy. Session restarts can leave old local indexes; removal is logical deletion, not secure erasure.

## Returns, covariance, Markowitz, and Sharpe

A daily simple return is `P[t]/P[t-1] - 1`. The historical arithmetic mean × 252 estimates annual expected return. Sample covariance × 252 describes annualized variation and joint movement under the daily-data assumption. Correlation describes standardized co-movement; covariance keeps return-scale units.

Markowitz mean–variance analysis balances expected return and risk using a return vector and covariance matrix. The minimum-variance portfolio here minimizes `wᵀΣw`, subject to nonnegative weights summing to one. Diversification can reduce risk when assets are not perfectly correlated, but historical diversification does not eliminate uncertainty.

Sharpe ratio is `(estimated annual return - annual risk-free rate) / annual volatility`. It measures estimated excess return per unit of risk. It is not guaranteed performance and is sensitive to historical estimates and the selected risk-free rate.

## The maximum-Sharpe mathematics used here

A raw Sharpe ratio is not a valid CVXPY DCP objective. The implementation uses a positive-excess-return transformation:

1. Set `a = mu - rf * 1`.
2. For a portfolio with `aᵀw > 0`, set `y = w / (aᵀw)`.
3. Solve `minimize yᵀΣy`, subject to `aᵀy = 1`, `y >= 0`.
4. Recover `w = y / sum(y)`.

Under this transformation Sharpe is `1 / sqrt(yᵀΣy)`, so minimizing the denominator maximizes the positive ratio. At least one asset must have positive estimated excess return. If none does, the app explicitly declines this maximum-Sharpe case; it does not manufacture equal weights or pretend the convex transformation solved a negative-excess case. Minimum variance can still be reported.

The app checks installed CLARABEL/OSQP/SCS solvers, accepts only optimal status, verifies finite output, nonnegative weights within tolerance, full investment and transformation residuals, then clips only tiny solver roundoff. There is no failure fallback allocation.

## Annualization and numerical stability

A daily dataset uses 252 trading days per year. Mean and covariance multiply by 252; volatility comes from the square root of annualized variance. Arithmetic annualization is not CAGR. The risk-free rate is entered annually in percent, then converted to decimal.

A small diagonal ridge is added: `max(trace(Sigma_sample)/n, 1e-6) * 1e-8`. This helps singular/highly correlated covariance estimates become numerically usable. The actual ridge, raw covariance, and adjusted covariance are reported; optimization and displayed risk use the same adjusted matrix. A ridge cannot fix unrealistic prices, insufficient observations, or unstable expected-return estimates. Near-constant series can make Sharpe highly sensitive to tiny variance; do not interpret numerical precision as economic confidence.

## Test evidence and tradeoffs

Read the current `docs/VALIDATION_REPORT.md` before quoting numbers. Tests exercise PDF/table metadata, overlapping chunks, real Chroma persistence/isolation/removal, input validation, known analytic optimizer solutions, grid comparisons, solver failures, report contents, and state invalidation. Some offline tests use clearly labeled API doubles. They validate integration logic, not live model quality or credentials. Real Gemini tests are opt-in and marked separately. Chromium checks exercise actual uploads, downloads and page rendering.

Tradeoffs: local single-user-style storage is simple to run but lacks authentication; character-based chunking is lightweight but not table-layout-aware; top-k retrieval has no reranker; historical arithmetic means are easy to explain but unstable; strict input validation avoids hidden data changes but requires users to prepare clean files. These choices are explicit rather than production claims.

## Likely questions and accurate answers

| Question | Answer tied to this code |
| --- | --- |
| Did you train Gemini? | No. The app uses existing Gemini embeddings and generation through supported APIs. |
| Why use RAG? | To give the model relevant uploaded evidence at query time and retain traceable references. |
| Why Chroma? | It provides local vector persistence and cosine retrieval without a separate database service. |
| How do reruns avoid duplicate API costs? | Index IDs include content/configuration hashes; a ready matching collection is reused. The index button, not each rerun, performs embedding. A new question still needs a query embedding/generation request. |
| Can I ask about scanned PDFs? | No OCR is implemented. Image-only PDFs are rejected; textless pages in mixed PDFs are disclosed/skipped. |
| What if a document says “ignore instructions”? | It is treated as untrusted evidence. The system policy forbids executing it; citations are validated. These are mitigations, not a proof against every attack. |
| What if evidence is missing? | Empty retrieval, an insufficient model response, or invalid/missing citations yields a canonical insufficient-information answer. |
| How are Excel references handled? | Per-row documents contain header/value text and filename, sheet, and 1-based row metadata. |
| Why not optimize Sharpe directly? | A ratio of affine return to a norm is not a valid DCP maximization. The implemented positive-excess transformation is a convex quadratic program. |
| Are prices fetched or adjusted automatically? | No. The user supplies adjusted closing prices and provenance; the app cannot independently verify that declaration. |
| What happens when a solver fails? | Show an explicit strategy error. Keep other successfully solved strategies; never substitute an allocation silently. |
| Is this a backtest? | No. It estimates and optimizes on the uploaded historical window, without an out-of-sample trading simulation. |
| Why 61 prices? | It gives at least 60 daily returns as a modest initial minimum, not statistical certainty. More observations and out-of-sample evaluation may be needed. |
| How is privacy handled? | Keys are read securely and ignored by Git; document text goes to Gemini and persists locally in ignored Chroma storage. This is not a private offline document model. |
| What could improve? | OCR with clear limits, layout-aware parsing, reranking, retrieval evaluation, provider/model migration testing, robust/shrinkage estimates, explicit market-calendar validation, and separate out-of-sample evaluation. These are future work, not current claims. |

Do not invent deployment scale, experience, accuracy percentages, live Gemini results, or a successful push. Show the actual modules, test output, sample data, and limitations.
