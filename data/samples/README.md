# Public fictional and simulated samples

All sample data is educational and contains no real company disclosures or market prices.

- **ABC Limited PDFs:** revenue in 2024 Rs. 100 crore; revenue in 2025 Rs. 120 crore; net profit in 2025 Rs. 15 crore. No 2030 revenue figure is disclosed. The longer report adds fictional qualitative notes across four pages.
- **CSV / Excel documents:** the same fictional financial facts, with a separate Notes sheet in Excel.
- **synthetic_prices.csv:** 253 business-day observations from 2024-01-02 through 2024-12-19 for `SIM_Alpha`, `SIM_Beta`, and `SIM_Gamma`. Generated with NumPy PCG64 seed 2025, centered correlated shocks, positive illustrative drift, and compounded daily simple returns. These are simulated price series, not adjusted exchange quotes.

Regenerate with `python scripts/generate_samples.py` after installing `requirements-dev.txt`. PDF bytes and numeric sample content are deterministic for the pinned stack; sample generation has no network calls. These public files are safe to commit. Keep real private documents and reports out of this directory.
