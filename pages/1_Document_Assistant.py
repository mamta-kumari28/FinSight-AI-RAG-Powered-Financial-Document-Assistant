from dataclasses import replace
from uuid import uuid4

import streamlit as st

from src.config import ROOT, load_settings
from src.errors import FinSightError
from src.ingestion import load_document
from src.rag.embeddings import GeminiEmbeddings, model_catalog, validate_models
from src.rag.qa_chain import answer_question, chat_client
from src.rag.vector_store import DocumentStore
from src.reporting.report_generator import json_bytes, qa_report
from src.state import fingerprint, sync_inputs

st.set_page_config(page_title="Document Assistant | FinSight AI", page_icon="📄", layout="wide")
st.title("📄 Document Assistant")
st.write("Ask evidence-grounded questions about text PDFs, CSVs, and Excel workbooks. Portfolio price files belong on the Portfolio Optimizer page.")
st.info("Privacy & API costs: document text is sent for Gemini embeddings; questions and retrieved passages are sent to Gemini for answers. API quota/billing applies. Use only documents you are authorized to share.")
try:
    secret_values = dict(st.secrets)
except (FileNotFoundError, st.errors.StreamlitSecretNotFoundError):
    secret_values = {}
settings = load_settings(secret_values)
with st.sidebar:
    st.header("Gemini Settings")
    st.caption("Set GEMINI_API_KEY in a local .env file or deployment secrets. Never commit it.")
    chat_model = st.text_input("Chat model", value=settings.chat_model, key="gemini_chat_model")
    embedding_model = st.text_input("Embedding model", value=settings.embedding_model, key="gemini_embedding_model")
    settings = replace(settings, chat_model=chat_model.strip(), embedding_model=embedding_model.strip())
    st.caption("Documented defaults are suggestions; availability is checked against your key's model metadata before use.")
    if not settings.api_key:
        st.warning("GEMINI_API_KEY is not configured. Inspect samples locally; live indexing and Q&A are unavailable.")
    model_identity = fingerprint(settings.api_key, settings.chat_model, settings.embedding_model)
    if st.button("Validate models / list available models", disabled=not settings.api_key):
        try:
            with st.spinner("Reading model metadata from Gemini…"):
                catalog = model_catalog(settings)
                st.json(catalog)
                validate_models(settings)
                st.session_state["validated_models"] = model_identity
            st.success("Configured models are listed with the required actions. Live inference still needs verification.")
        except FinSightError as exc:
            st.error(str(exc))


def ensure_models():
    if st.session_state.get("validated_models") != model_identity:
        validate_models(settings)
        st.session_state["validated_models"] = model_identity


st.session_state.setdefault("document_session", uuid4().hex)
try:
    store = DocumentStore(settings.storage_root, st.session_state["document_session"])
except Exception:
    st.error("Local document storage is unavailable. Check disk space/permissions and restore or remove a damaged local Chroma directory while the app is stopped.")
    st.stop()
samples = ROOT / "data" / "samples"
with st.expander("Download fictional sample documents"):
    for filename in ("abc_financial_report.pdf", "abc_long_report.pdf", "financial_summary.csv", "financial_summary.xlsx"):
        st.download_button(f"Download {filename}", (samples / filename).read_bytes(), file_name=filename, key=f"sample_{filename}")
st.subheader("1. Select and inspect documents")
source = st.radio("Document source", ["Upload documents", "Fictional sample documents"], horizontal=True)
inputs = []
if source == "Upload documents":
    uploads = st.file_uploader("Financial documents (20 MB per file; PDF, CSV, XLSX)", type=["pdf", "csv", "xlsx"], accept_multiple_files=True)
    inputs = [(file.name, file.getvalue()) for file in uploads]
else:
    chosen = st.multiselect("Fictional documents", ["abc_financial_report.pdf", "abc_long_report.pdf", "financial_summary.csv", "financial_summary.xlsx"], default=["abc_financial_report.pdf"])
    inputs = [(name, (samples / name).read_bytes()) for name in chosen]
if len(inputs) > 10 or sum(len(data) for _, data in inputs) > 40 * 1024 * 1024:
    st.error("Choose at most 10 files and 40 MB total per batch.")
    inputs = []
input_identity = fingerprint([(name, fingerprint(data)) for name, data in inputs], model_identity)
sync_inputs(st.session_state, "document_input_identity", input_identity, ("qa_result", "inspection"))
if st.button("Inspect locally (no API call)", disabled=not inputs):
    st.session_state.pop("inspection", None)
    try:
        inspection = []
        for name, data in inputs:
            documents = load_document(data, name)
            inspection.append({"filename": name, "units": len(documents), "metadata": documents[0].metadata, "preview": documents[0].page_content[:1200]})
        st.session_state["inspection"] = inspection
    except FinSightError as exc:
        st.error(str(exc))
if "inspection" in st.session_state:
    for item in st.session_state["inspection"]:
        with st.expander(f"{item['filename']} — {item['units']} pages/rows", expanded=True):
            st.json(item["metadata"])
            if item["metadata"].get("empty_pages", 0):
                st.warning("Some PDF pages have no extractable text and were skipped. OCR is unsupported.")
            st.text(item["preview"])
if st.button("Index selected documents", disabled=not inputs or not settings.configured, type="primary"):
    st.session_state.pop("qa_result", None)
    try:
        with st.status("Validating models and indexing documents…", expanded=True) as status:
            ensure_models()
            embedder = GeminiEmbeddings(settings)
            for name, data in inputs:
                record, created = store.index(data, name, embedder)
                st.session_state["selected_document_indexes"] = list(dict.fromkeys([*st.session_state.get("selected_document_indexes", []), record.id]))
                st.write(f"{record.filename}: {record.chunks} chunks — {'indexed' if created else 'reused existing index; no document re-embedding'}")
            status.update(label="Indexing complete", state="complete", expanded=True)
    except FinSightError as exc:
        st.error(str(exc))
    except Exception:
        st.error("Local indexing failed. Check storage permissions, remove stale indexes, and retry. No credentials are shown.")
st.subheader("2. Manage this session's indexes")
records = store.list_indexes()
names = {record.id: f"{record.filename} · {record.chunks} chunks · {record.id[-8:]}" for record in records}
st.caption("Indexes persist locally in this browser session's namespace. A new browser session starts a separate namespace; this is not an authentication boundary.")
selected = st.multiselect("Use these indexed documents for Q&A", list(names), default=list(names), format_func=lambda value: names[value], key="selected_document_indexes")
left, right = st.columns(2)
with left:
    remove_id = st.selectbox("Document index to remove", [""] + list(names), format_func=lambda value: names.get(value, "Select an index"))
    if st.button("Remove document index", disabled=not remove_id):
        store.remove(remove_id)
        st.session_state.pop("qa_result", None)
        st.session_state.pop("selected_document_indexes", None)
        st.rerun()
with right:
    confirm_reset = st.checkbox("Confirm removal of all indexes in this session")
    if st.button("Reset session indexes", disabled=not confirm_reset):
        store.reset()
        st.session_state.pop("qa_result", None)
        st.session_state.pop("selected_document_indexes", None)
        st.rerun()
    st.caption("Removal is logical deletion, not guaranteed secure disk erasure. Private uploads are not saved as raw files.")
st.subheader("3. Ask a question")
question = st.text_area("Question", placeholder="How did revenue change from 2024 to 2025?", max_chars=2000)
qa_identity = fingerprint(input_identity, sorted(selected), question)
sync_inputs(st.session_state, "qa_identity", qa_identity, ("qa_result",))
if st.button("Ask Gemini", disabled=not selected or not question.strip() or not settings.configured, type="primary"):
    st.session_state.pop("qa_result", None)
    try:
        with st.spinner("Retrieving evidence and generating a grounded answer…"):
            ensure_models()
            evidence = store.retrieve(selected, question, GeminiEmbeddings(settings))
            answer = answer_question(question, evidence, chat_client(settings))
            markdown, structured = qa_report(answer, {"chat": settings.chat_model, "embedding": settings.embedding_model})
            st.session_state["qa_result"] = (answer, markdown, json_bytes(structured))
    except FinSightError as exc:
        st.error(str(exc))
    except Exception:
        st.error("The local retrieval request failed. Check your indexes and settings, then retry.")
if "qa_result" in st.session_state:
    answer, markdown, structured = st.session_state["qa_result"]
    st.subheader("Answer")
    st.write(answer.text)
    for item in answer.sources:
        st.text(f"{item['id']}: {item['source']}")
    st.caption("Citations identify retrieved evidence, not a guarantee of factual correctness. Verify the original figures.")
    for number, item in enumerate(answer.evidence, 1):
        with st.expander(f"E{number} — {item.source}"):
            st.text(item.text)
            st.caption(f"Cosine distance: {item.distance:.3f} (lower is closer; not a confidence score)")
    st.download_button("Download Q&A report (Markdown)", markdown, file_name="finsight_qa.md", mime="text/markdown")
    st.download_button("Download Q&A report (JSON)", structured, file_name="finsight_qa.json", mime="application/json")
