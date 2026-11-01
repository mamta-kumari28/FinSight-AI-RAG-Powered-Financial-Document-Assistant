import streamlit as st

st.set_page_config(page_title="FinSight AI", page_icon="📊", layout="wide")
st.title("📊 FinSight AI")
st.subheader("Financial documents. Traceable answers. Historical portfolio research.")
st.write("Explore evidence in financial documents or compare long-only portfolios using your own historical prices.")
st.divider()
left, right = st.columns(2, gap="large")
with left:
    st.header("📄 Document Assistant")
    st.write("Upload text PDFs, CSVs, or Excel workbooks. Index their text with Gemini embeddings and ask questions with page, sheet, and row references.")
    st.page_link("pages/1_Document_Assistant.py", label="Open Document Assistant", icon="📄")
    st.caption("Requires your Gemini API key and available model names. No OCR.")
with right:
    st.header("📈 Portfolio Optimizer")
    st.write("Upload daily adjusted closing prices. Compare minimum-variance and maximum-Sharpe portfolios, inspect charts, and download research reports.")
    st.page_link("pages/2_Portfolio_Optimizer.py", label="Open Portfolio Optimizer", icon="📈")
    st.caption("Works independently of Gemini. Includes reproducible simulated prices.")
st.divider()
st.header("Start with transparent examples")
st.write("Fictional ABC Limited reports, CSV/Excel documents, and deterministic simulated price data are available on their respective pages. Sample documents contain no real company disclosures.")
with st.expander("Privacy and research limitations", expanded=True):
    st.write("Document text is sent to the embedding provider. Questions and retrieved passages are sent to Gemini for answers. API usage can consume quota and incur costs. Chroma stores text and vectors locally until you remove/reset indexes.")
    st.write("Local browser-session separation is intended for personal research; this is not production multi-user security. Portfolio estimates use historical data and exclude transaction costs and taxes. No trading, model training, or guaranteed returns.")
st.caption("Educational financial research only. Verify source evidence and assumptions before drawing conclusions.")
