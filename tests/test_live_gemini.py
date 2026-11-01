"""Real integration checks. Opt in explicitly; these consume API quota and send fictional text."""
import re
from uuid import uuid4

import pytest

from src.config import load_settings
from src.rag.embeddings import GeminiEmbeddings, validate_models
from src.rag.qa_chain import answer_question, chat_client
from src.rag.vector_store import DocumentStore

pytestmark = pytest.mark.live


@pytest.fixture(scope='module')
def live_context(tmp_path_factory):
    settings = load_settings()
    if not settings.configured:
        pytest.skip('GEMINI_API_KEY and available models must be configured securely.')
    validate_models(settings)
    embeddings = GeminiEmbeddings(settings)
    store = DocumentStore(tmp_path_factory.mktemp('live-chroma'), uuid4().hex)
    from src.config import ROOT
    index, _ = store.index((ROOT / 'data/samples/abc_financial_report.pdf').read_bytes(), 'abc_financial_report.pdf', embeddings)
    yield store, index, embeddings, chat_client(settings)
    store.reset()


def test_live_embedding_index_and_citations(live_context):
    store, record, embedder, client = live_context
    evidence = store.retrieve([record.id], 'What was net profit in 2025?', embedder)
    assert evidence and any('15 crore' in item.text for item in evidence)
    answer = answer_question('What was net profit in 2025?', evidence, client)
    assert not answer.insufficient and '15' in answer.text
    assert answer.sources and all('page 1' in source['source'] for source in answer.sources)


def test_live_growth_and_missing_information(live_context):
    store, record, embedder, client = live_context
    question = 'How much did revenue increase from 2024 to 2025, both in Rs. crore and percentage? Show arithmetic.'
    answer = answer_question(question, store.retrieve([record.id], question, embedder), client)
    assert not answer.insufficient and '20' in answer.text and '%' in answer.text and answer.sources
    question = 'What is revenue in 2030?'
    missing = answer_question(question, store.retrieve([record.id], question, embedder), client)
    assert missing.insufficient
